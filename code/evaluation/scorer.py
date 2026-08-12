import sys
import csv
from typing import Dict, Any, List
from pathlib import Path

from config import SAMPLE_MESSAGES_CSV
from pipeline.context import DatasetLoader, load_csv
from pipeline.graph import build_pipeline_graph
from checkpoints.manager import CheckpointManager
from pipeline.evidence import select_evidence
from pipeline.extraction import process_media_item
from pipeline.gate import apply_safety_gate
from pipeline.confidence import compute_confidence
from schemas import RoutingDecision, ActionEnum, MessageTypeEnum


def run_evaluation(
    loader: DatasetLoader,
    checkpoint_mgr: CheckpointManager,
    yes: bool = False,
    dry_run: bool = False,
):
    print("=" * 60)
    print("Evaluation Suite Scoped to sample_messages.csv")
    print("=" * 60)

    sample_rows = load_csv(SAMPLE_MESSAGES_CSV)
    if not sample_rows:
        print("Error: dataset/sample_messages.csv not found or empty.")
        return

    eval_data = checkpoint_mgr.load("eval")
    needs_llm_run = False

    for s_row in sample_rows:
        s_id = s_row.get("message_id", "")
        if not checkpoint_mgr.is_done("eval", s_id):
            needs_llm_run = True
            break

    # Confirmation Guard on Cache Miss
    if needs_llm_run and not dry_run and not yes:
        user_input = input(
            "No eval cache found — this will call the LLM on sample rows (real cost). Continue? [y/N]: "
        )
        if user_input.strip().lower() not in ["y", "yes"]:
            print("Evaluation cancelled by user.")
            return

    # Phase 1: Populate / Update eval checkpoint
    graph = build_pipeline_graph(loader, checkpoint_mgr)

    for s_row in sample_rows:
        s_id = s_row.get("message_id", "")
        if not s_id:
            continue

        if not checkpoint_mgr.is_done("eval", s_id):
            print(f"Eval LLM call for {s_id}...", end="", flush=True)
            try:
                state = {
                    "msg_row": s_row,
                    "context": {},
                    "evidence_ids": "none",
                    "evidence_dicts": [],
                    "media_extraction": None,
                    "routing_decision": None,
                    "final_decision": None,
                    "is_gate_override": False,
                    "confidence": 0.0,
                    "iteration_count": 0,
                    "dry_run": dry_run,
                    "force_media": False,
                }
                res_state = graph.invoke(state)
                f_dec: RoutingDecision = res_state["final_decision"]
                conf = res_state["confidence"]
                ev_ids = res_state["evidence_ids"]

                record = {
                    "message_id": s_id,
                    "action": f_dec.action.value if hasattr(f_dec.action, "value") else str(f_dec.action),
                    "message_type": f_dec.message_type.value if hasattr(f_dec.message_type, "value") else str(f_dec.message_type),
                    "reason": f_dec.reason,
                    "confidence": str(conf),
                    "evidence_message_ids": ev_ids,
                }
                checkpoint_mgr.set_record("eval", s_id, "done", result=record)
                print(f" -> {record['action']} ({record['message_type']})")
            except Exception as e:
                print(f" -> ERROR: {e}")

    # Phase 2: Score predictions vs Ground Truth (apply current config.py)
    eval_cache = checkpoint_mgr.load("eval")

    action_correct = 0
    type_correct = 0
    evidence_valid = 0
    total_rows = len(sample_rows)

    print("\n" + "-" * 60)
    print("FIELD-BY-FIELD EVALUATION REPORT")
    print("-" * 60)

    for s_row in sample_rows:
        s_id = s_row.get("message_id", "")
        gt_action = s_row.get("action", "")
        gt_type = s_row.get("message_type", "")
        gt_evidence = s_row.get("evidence_message_ids", "none")

        rec = eval_cache.get(s_id, {}).get("result", {})
        ctx = loader.get_context_for_message(s_row)
        ev_ids, _ = select_evidence(ctx, loader)
        media_extr = process_media_item(ctx, checkpoint_mgr, dry_run=True)

        raw_dec = RoutingDecision(
            action=ActionEnum(rec.get("action", "digest")),
            message_type=MessageTypeEnum(rec.get("message_type", "unknown")),
            reason=rec.get("reason", ""),
            potential_prompt_injection=rec.get("potential_prompt_injection", False),
            injection_confidence=rec.get("injection_confidence", 0.0),
        )

        final_dec, is_override = apply_safety_gate(raw_dec, ctx, media_extr)
        conf = compute_confidence(final_dec, ctx, is_override)

        pred_action = final_dec.action.value
        pred_type = final_dec.message_type.value

        is_act_match = pred_action == gt_action
        is_type_match = pred_type == gt_type
        # Evidence match: check actual overlap between predicted and ground-truth IDs
        gt_ev_set = set(gt_evidence.split(";")) if gt_evidence != "none" else set()
        pred_ev_set = set(ev_ids.split(";")) if ev_ids != "none" else set()
        if gt_ev_set == pred_ev_set:
            is_ev_match = True  # Exact match (including both "none")
        elif gt_ev_set and pred_ev_set:
            is_ev_match = bool(gt_ev_set & pred_ev_set)  # Partial overlap counts
        else:
            is_ev_match = False  # One has evidence, the other doesn't

        if is_act_match:
            action_correct += 1
        if is_type_match:
            type_correct += 1
        if is_ev_match:
            evidence_valid += 1

        print(
            f"ID: {s_id:15} | Act: {pred_action:6} (GT: {gt_action:6}) {'[OK]' if is_act_match else '[FAIL]'} | Type: {pred_type:15} (GT: {gt_type:15}) {'[OK]' if is_type_match else '[FAIL]'}"
        )

    act_acc = (action_correct / total_rows) * 100 if total_rows > 0 else 0
    type_acc = (type_correct / total_rows) * 100 if total_rows > 0 else 0
    ev_acc = (evidence_valid / total_rows) * 100 if total_rows > 0 else 0

    print("=" * 60)
    print(f"Action Accuracy:       {act_acc:.1f}% ({action_correct}/{total_rows})")
    print(f"Message Type Accuracy: {type_acc:.1f}% ({type_correct}/{total_rows})")
    print(f"Evidence Relevance:    {ev_acc:.1f}% ({evidence_valid}/{total_rows})")
    print("=" * 60)
