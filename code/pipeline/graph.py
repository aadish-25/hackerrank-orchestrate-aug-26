from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from schemas import RoutingDecision, MediaExtraction
from pipeline.context import DatasetLoader
from pipeline.evidence import select_evidence
from pipeline.extraction import process_media_item
from pipeline.routing import run_routing_llm
from pipeline.gate import apply_safety_gate
from pipeline.confidence import compute_confidence
from checkpoints.manager import CheckpointManager


class GraphState(TypedDict):
    msg_row: Dict[str, str]
    context: Dict[str, Any]
    evidence_ids: str
    evidence_dicts: List[Dict[str, Any]]
    media_extraction: Optional[MediaExtraction]
    routing_decision: Optional[RoutingDecision]
    final_decision: Optional[RoutingDecision]
    is_gate_override: bool
    confidence: float
    context_evidence_summary: str
    iteration_count: int
    dry_run: bool
    force_media: bool


def context_node(state: GraphState, loader: DatasetLoader) -> Dict[str, Any]:
    msg_row = state["msg_row"]
    ctx = loader.get_context_for_message(msg_row)
    ev_ids, ev_dicts = select_evidence(ctx, loader)

    evidence_summary = "None"
    if ev_dicts:
        summaries = [
            f"- Past Msg {e['message_id']}: '{e['text'][:60]}' (Event: opened={e['event'].get('message_opened')}, replied={e['event'].get('message_replied')}, dismissed={e['event'].get('notification_dismissed')})"
            for e in ev_dicts
        ]
        evidence_summary = "\n".join(summaries)

    return {
        "context": ctx,
        "evidence_ids": ev_ids,
        "evidence_dicts": ev_dicts,
        "context_evidence_summary": evidence_summary,
    }


def media_node(state: GraphState, checkpoint_mgr: CheckpointManager) -> Dict[str, Any]:
    ctx = state["context"]
    dry_run = state.get("dry_run", False)
    force_media = state.get("force_media", False)
    media_type = ctx.get("media_type", "")
    msg_id = ctx.get("message_id", "")

    if media_type:
        print(f"  [{media_type.upper()}] Processing media for {msg_id}...", end=" ", flush=True)

    media_extr = process_media_item(
        ctx, checkpoint_mgr, force_media=force_media, dry_run=dry_run
    )

    if media_type:
        cached_label = "(cached)" if not force_media and not dry_run else ""
        print(f"done {cached_label}")

    return {"media_extraction": media_extr}


def routing_node(state: GraphState) -> Dict[str, Any]:
    ctx = state["context"]
    ev_summary = state.get("context_evidence_summary", "None")
    media_extr = state.get("media_extraction")
    dry_run = state.get("dry_run", False)

    iter_cnt = state.get("iteration_count", 0) + 1

    decision = run_routing_llm(
        ctx, ev_summary, media_extraction=media_extr, dry_run=dry_run
    )
    return {"routing_decision": decision, "iteration_count": iter_cnt}


def gate_node(state: GraphState) -> Dict[str, Any]:
    raw_decision = state["routing_decision"]
    ctx = state["context"]
    media_extr = state.get("media_extraction")

    final_decision, is_override = apply_safety_gate(raw_decision, ctx, media_extr)
    conf = compute_confidence(final_decision, ctx, is_override)

    return {
        "final_decision": final_decision,
        "is_gate_override": is_override,
        "confidence": conf,
    }


def build_pipeline_graph(loader: DatasetLoader, checkpoint_mgr: CheckpointManager):
    workflow = StateGraph(GraphState)

    def _ctx(state):
        return context_node(state, loader)

    def _media(state):
        return media_node(state, checkpoint_mgr)

    workflow.add_node("context_step", _ctx)
    workflow.add_node("media_step", _media)
    workflow.add_node("routing_step", routing_node)
    workflow.add_node("gate_step", gate_node)

    workflow.set_entry_point("context_step")
    workflow.add_edge("context_step", "media_step")
    workflow.add_edge("media_step", "routing_step")
    workflow.add_edge("routing_step", "gate_step")
    workflow.add_edge("gate_step", END)

    return workflow.compile()
