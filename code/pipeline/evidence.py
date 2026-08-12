from typing import Dict, Any, List, Tuple
from config import (
    MUTE_SIGNAL_WEIGHTS,
    NOTIFY_SIGNAL_WEIGHTS,
    REACTION_TIME_FAST_THRESHOLD_MINUTES,
    EVIDENCE_MAX_COUNT,
)


def select_evidence(
    context: Dict[str, Any], loader: Any
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Selects top-3 historical evidence message IDs pre-LLM using pure Python.
    Returns (evidence_message_ids_str, list_of_evidence_dicts).
    """
    user_id = context.get("user_id", "")
    sender_id = context.get("sender_user_id", "")
    group_id = context.get("group_id", "")
    business_id = context.get("business_id", "")

    if not user_id:
        return "none", []

    # Step 1: Filter candidates in message_history
    candidates = []
    for h_msg in loader.message_history:
        if h_msg.get("user_id") != user_id:
            continue

        h_sender = h_msg.get("sender_user_id", "")
        h_group = h_msg.get("group_id", "")
        h_biz = h_msg.get("business_id", "")

        matches_sender = bool(sender_id and h_sender == sender_id)
        matches_group = bool(group_id and h_group == group_id)
        matches_biz = bool(business_id and h_biz == business_id)

        if matches_sender or matches_group or matches_biz:
            candidates.append(h_msg)

    if not candidates:
        return "none", []

    # Step 2: Determine initial lean direction for pool scoring
    lean_mute = (
        context.get("group_muted", False)
        or context.get("is_domain_fraud", False)
        or context.get("regex_injection_detected", False)
        or context.get("forwarded_count", 0) >= 6
    )

    weights = MUTE_SIGNAL_WEIGHTS if lean_mute else NOTIFY_SIGNAL_WEIGHTS

    # Step 3: Score candidates
    scored_candidates = []
    for cand in candidates:
        h_id = cand.get("message_id", "")
        event = loader.message_events.get((user_id, h_id), {})
        score = 0.0

        if lean_mute:
            if event.get("notification_dismissed") == "1":
                score += weights.get("notification_dismissed", 0.35)
            if event.get("muted_after_message") == "1":
                score += weights.get("muted_after_message", 0.35)
            if event.get("message_reported") == "1":
                score += weights.get("message_reported", 0.30)
        else:
            if event.get("message_replied") == "1":
                score += weights.get("message_replied", 0.40)
            if event.get("message_opened") == "1":
                score += weights.get("message_opened", 0.25)
            try:
                rx_time = float(event.get("reaction_time_minutes", 999) or 999)
                if rx_time <= REACTION_TIME_FAST_THRESHOLD_MINUTES:
                    score += weights.get("fast_reaction_time", 0.35)
            except Exception:
                pass

        if score > 0:
            scored_candidates.append((score, h_id, cand, event))

    if not scored_candidates:
        return "none", []

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:EVIDENCE_MAX_COUNT]

    evidence_ids = [item[1] for item in top_candidates]
    evidence_dicts = [
        {"message_id": item[1], "text": item[2].get("message_text", ""), "event": item[3]}
        for item in top_candidates
    ]

    return ";".join(evidence_ids), evidence_dicts
