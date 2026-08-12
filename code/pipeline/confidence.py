from typing import Dict, Any
from config import CONFIDENCE_WEIGHTS, GATE_OVERRIDE_CONFIDENCE
from schemas import RoutingDecision, ActionEnum


def compute_confidence(
    decision: RoutingDecision,
    context: Dict[str, Any],
    is_gate_override: bool,
) -> float:
    """
    Computes code-driven confidence score (0.0 - 1.0) using weights in config.py.
    """
    if is_gate_override:
        return GATE_OVERRIDE_CONFIDENCE

    biz_info = context.get("business_info", {})
    user_biz = context.get("user_biz_info", {})
    group_member = context.get("group_member_info", {})
    sender_member = context.get("sender_member_info", {})

    total_score = 0.0

    # 1. Business Verified
    if biz_info.get("verified") == "1":
        total_score += CONFIDENCE_WEIGHTS.get("business_verified", 0.20)

    # 2. Opt-In Active
    if user_biz.get("allows_promotions") == "1":
        total_score += CONFIDENCE_WEIGHTS.get("opt_in_active", 0.15)

    # 3. Prior Engagement Positive
    try:
        opened = int(user_biz.get("messages_opened_30d", 0) or 0)
        replied = int(user_biz.get("messages_replied_30d", 0) or 0)
        if opened > 2 or replied > 0:
            total_score += CONFIDENCE_WEIGHTS.get("prior_engagement_positive", 0.20)
    except Exception:
        pass

    # 4. Absence of Injection Flags
    if not decision.potential_prompt_injection and not context.get("regex_injection_detected", False):
        total_score += CONFIDENCE_WEIGHTS.get("no_injection_flag", 0.15)

    # 5. Domain Match
    off_domain = biz_info.get("official_domain", "").strip()
    sender_domain = biz_info.get("domain_used_by_sender", "").strip()
    if not off_domain or (off_domain and sender_domain and off_domain == sender_domain):
        total_score += CONFIDENCE_WEIGHTS.get("domain_match", 0.15)

    # 6. Sender Trust (Group Admin or Known Personal Contact)
    if sender_member.get("role") == "admin" or context.get("conversation_type") == "personal":
        total_score += CONFIDENCE_WEIGHTS.get("sender_trust", 0.15)

    # Base baseline boost so confidence is bounded between 0.60 and 0.95
    final_conf = 0.60 + (total_score * 0.35)
    return round(min(max(final_conf, 0.50), 0.95), 2)
