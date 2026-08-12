import sys
import csv
import argparse
from pathlib import Path

from config import (
    MESSAGES_CSV,
    OUTPUT_CSV,
    SAMPLE_MESSAGES_CSV,
)
from pipeline.context import DatasetLoader, load_csv
from pipeline.graph import build_pipeline_graph
from checkpoints.manager import CheckpointManager
from pipeline.evidence import select_evidence
from pipeline.extraction import process_media_item
from pipeline.gate import apply_safety_gate
from pipeline.confidence import compute_confidence
from schemas import RoutingDecision, ActionEnum, MessageTypeEnum


def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp Message Notification Router"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip completed records, retry error/in_progress (default)",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Wipe all checkpoints and restart from scratch",
    )
    parser.add_argument(
        "--force-media",
        action="store_true",
        help="Wipe media checkpoint only",
    )
    parser.add_argument(
        "--force-routing",
        action="store_true",
        help="Wipe routing and evaluation checkpoints",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ALL checkpoints and output.csv, then exit (full reset)",
    )
    parser.add_argument(
        "--recompute-gate",
        action="store_true",
        help="Re-run evidence, safety gate, and confidence over cached routing outputs ($0 cost)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation against sample_messages.csv ground truth",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run context loading + safety gate only, zero API calls",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Bypass interactive confirmation prompts",
    )

    args = parser.parse_args()

    checkpoint_mgr = CheckpointManager()

    # --reset: wipe everything and exit
    if args.reset:
        checkpoint_mgr.handle_flags(force_clean=True, force_media=False, force_routing=False)
        if OUTPUT_CSV.exists():
            OUTPUT_CSV.unlink()
            print("Deleted dataset/output.csv")
        print("=" * 60)
        print("Full reset complete. All checkpoints and output wiped.")
        print("=" * 60)
        sys.exit(0)

    # Force flags are modifiers: wipe the relevant caches, then continue to pipeline
    checkpoint_mgr.handle_flags(
        force_clean=args.force_clean,
        force_media=args.force_media,
        force_routing=args.force_routing,
    )

    loader = DatasetLoader()

    # Route 1: Evaluation Flag
    if args.evaluate:
        from evaluation.scorer import run_evaluation

        run_evaluation(loader, checkpoint_mgr, yes=args.yes, dry_run=args.dry_run)
        return

    # Route 2: Recompute Gate Only ($0 cost)
    if args.recompute_gate:
        run_recompute_gate(loader, checkpoint_mgr)
        return

    # Route 3: Standard Pipeline Execution (or --dry-run)
    run_main_pipeline(
        loader,
        checkpoint_mgr,
        dry_run=args.dry_run,
        force_media=args.force_media,
    )


def run_main_pipeline(
    loader: DatasetLoader,
    checkpoint_mgr: CheckpointManager,
    dry_run: bool = False,
    force_media: bool = False,
):
    print("=" * 60)
    print(f"Starting Message Notification Router (dry_run={dry_run})")
    print("=" * 60)

    messages = load_csv(MESSAGES_CSV)
    if not messages:
        print("No messages found in dataset/messages.csv")
        return

    graph = build_pipeline_graph(loader, checkpoint_mgr)
    output_rows = []

    for idx, msg_row in enumerate(messages, 1):
        msg_id = msg_row.get("message_id", "")
        if not msg_id:
            continue

        # Skip if done (unless force re-run)
        if checkpoint_mgr.is_done("routing", msg_id):
            cached = checkpoint_mgr.get_record("routing", msg_id)
            if cached and "result" in cached:
                output_rows.append(cached["result"])
                continue

        print(f"[{idx}/{len(messages)}] Processing {msg_id}...", end="", flush=True)

        try:
            initial_state = {
                "msg_row": msg_row,
                "context": {},
                "evidence_ids": "none",
                "evidence_dicts": [],
                "media_extraction": None,
                "routing_decision": None,
                "final_decision": None,
                "is_gate_override": False,
                "confidence": 0.0,
                "context_evidence_summary": "None",
                "iteration_count": 0,
                "dry_run": dry_run,
                "force_media": force_media,
            }

            result_state = graph.invoke(initial_state)
            final_dec: RoutingDecision = result_state["final_decision"]
            conf: float = result_state["confidence"]
            ev_ids: str = result_state["evidence_ids"]

            row_dict = {
                "message_id": msg_id,
                "action": final_dec.action.value if hasattr(final_dec.action, "value") else str(final_dec.action),
                "message_type": final_dec.message_type.value if hasattr(final_dec.message_type, "value") else str(final_dec.message_type),
                "reason": final_dec.reason,
                "confidence": str(conf),
                "evidence_message_ids": ev_ids,
                "potential_prompt_injection": final_dec.potential_prompt_injection,
                "injection_confidence": final_dec.injection_confidence,
            }

            # If the routing hit an API error and yielded a mock fallback, mark it as "error"
            # so the CheckpointManager skips it and retries on the next --resume run.
            status = "error" if "[MOCK_FALLBACK]" in final_dec.reason else "done"
            checkpoint_mgr.set_record("routing", msg_id, status, result=row_dict)
            output_rows.append(row_dict)
            print(f" -> {row_dict['action']} ({row_dict['message_type']}) [conf={conf}]")

        except Exception as e:
            print(f" -> ERROR: {e}")
            fallback_row = {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "unknown",
                "reason": "Processing failed — routed to digest as safe default.",
                "confidence": "0.30",
                "evidence_message_ids": "none",
            }
            checkpoint_mgr.set_record(
                "routing", msg_id, "error", result=fallback_row, error=str(e)
            )
            output_rows.append(fallback_row)

    _write_output_csv(output_rows)
    print("=" * 60)
    print(f"Pipeline Execution Complete. Written {len(output_rows)} predictions to dataset/output.csv")
    print("=" * 60)


def run_recompute_gate(loader: DatasetLoader, checkpoint_mgr: CheckpointManager):
    print("=" * 60)
    print("Running Gate-Only Recompute (--recompute-gate, $0 Cost)")
    print("=" * 60)

    routing_data = checkpoint_mgr.load("routing")
    messages = load_csv(MESSAGES_CSV)
    output_rows = []

    for msg_row in messages:
        msg_id = msg_row.get("message_id", "")
        if not msg_id:
            continue

        cached_rec = routing_data.get(msg_id, {})
        if not cached_rec or "result" not in cached_rec:
            # Fallback if un-cached
            ctx = loader.get_context_for_message(msg_row)
            ev_ids, _ = select_evidence(ctx, loader)
            output_rows.append(
                {
                    "message_id": msg_id,
                    "action": "digest",
                    "message_type": "unknown",
                    "reason": "Uncached record in recompute gate.",
                    "confidence": "0.30",
                    "evidence_message_ids": ev_ids,
                }
            )
            continue

        ctx = loader.get_context_for_message(msg_row)
        ev_ids, _ = select_evidence(ctx, loader)
        media_extr = process_media_item(ctx, checkpoint_mgr, dry_run=True)

        cached_res = cached_rec["result"]
        raw_dec = RoutingDecision(
            action=ActionEnum(cached_res["action"]),
            message_type=MessageTypeEnum(cached_res["message_type"]),
            reason=cached_res["reason"],
            potential_prompt_injection=cached_res.get("potential_prompt_injection", False),
            injection_confidence=cached_res.get("injection_confidence", 0.0),
        )

        final_dec, is_override = apply_safety_gate(raw_dec, ctx, media_extr)
        conf = compute_confidence(final_dec, ctx, is_override)

        updated_row = {
            "message_id": msg_id,
            "action": final_dec.action.value,
            "message_type": final_dec.message_type.value,
            "reason": final_dec.reason,
            "confidence": str(conf),
            "evidence_message_ids": ev_ids,
        }
        output_rows.append(updated_row)

    _write_output_csv(output_rows)
    print(f"Recompute Gate Complete. Updated {len(output_rows)} records in dataset/output.csv")


def _write_output_csv(rows):
    fieldnames = [
        "message_id",
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
