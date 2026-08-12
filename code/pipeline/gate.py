from typing import Dict, Any, Optional, Tuple

from config import (
    INJECTION_CONFIDENCE_THRESHOLD,
    FORWARD_COUNT_MUTE_THRESHOLD,
)
from schemas import RoutingDecision, ActionEnum, MessageTypeEnum, MediaExtraction


def apply_safety_gate(
    decision: RoutingDecision,
    context: Dict[str, Any],
    media_extraction: Optional[MediaExtraction] = None,
) -> Tuple[RoutingDecision, bool]:
    """
    Applies deterministic safety gate logic in pure Python (zero model calls).
    Returns (final_decision, is_gate_override).
    """
    action = decision.action
    msg_type = decision.message_type
    reason = decision.reason

    msg_text = context.get("message_text", "")
    forwarded_count = context.get("forwarded_count", 0)
    group_muted = context.get("group_muted", False)
    user_mentioned = context.get("user_mentioned", False)
    is_dnd = context.get("is_dnd", False)
    is_domain_fraud = context.get("is_domain_fraud", False)

    # Media observations if present
    media_text = media_extraction.visible_text if media_extraction else ""
    media_transcript = media_extraction.transcript if media_extraction else ""
    media_payment = media_extraction.payment_request_present if media_extraction else False
    media_otp = media_extraction.otp_request_present if media_extraction else False
    media_inj = media_extraction.potential_prompt_injection if media_extraction else False
    media_inj_conf = media_extraction.injection_confidence if media_extraction else 0.0

    # Rule 1: Injection Defense (Layer 1 OR Layer 2)
    layer1_inj = (
        decision.potential_prompt_injection
        and decision.injection_confidence >= INJECTION_CONFIDENCE_THRESHOLD
    ) or (
        media_inj and media_inj_conf >= INJECTION_CONFIDENCE_THRESHOLD
    )

    layer2_inj = (
        context.get("regex_injection_detected", False)
        or _contains_regex_injection(media_text)
        or _contains_regex_injection(media_transcript)
    )

    if layer1_inj or layer2_inj:
        return (
            RoutingDecision(
                action=ActionEnum.mute,
                message_type=MessageTypeEnum.scam,
                reason="Gate Override: System prompt injection attempt detected.",
                potential_prompt_injection=True,
                injection_confidence=0.95,
            ),
            True,
        )

    # Rule 2: Domain Fraud Detection
    if is_domain_fraud:
        return (
            RoutingDecision(
                action=ActionEnum.mute,
                message_type=MessageTypeEnum.scam,
                reason="Gate Override: Sender domain is newly registered or mismatched with official brand domain.",
                potential_prompt_injection=decision.potential_prompt_injection,
                injection_confidence=decision.injection_confidence,
            ),
            True,
        )

    # Rule 3: Unverified Business + Payment/OTP Pressure
    biz_info = context.get("business_info", {})
    if biz_info.get("verified") == "0" and (media_payment or media_otp):
        return (
            RoutingDecision(
                action=ActionEnum.mute,
                message_type=MessageTypeEnum.scam,
                reason="Gate Override: Unverified business account requesting sensitive payment or OTP verification.",
                potential_prompt_injection=decision.potential_prompt_injection,
                injection_confidence=decision.injection_confidence,
            ),
            True,
        )

    # Rule 4: High Forward Chain Messages
    if forwarded_count >= FORWARD_COUNT_MUTE_THRESHOLD:
        if action != ActionEnum.mute:
            return (
                RoutingDecision(
                    action=ActionEnum.mute,
                    message_type=MessageTypeEnum.forward if msg_type != MessageTypeEnum.spam else msg_type,
                    reason=f"Gate Override: Viral forwarded message (forwarded_count={forwarded_count}).",
                    potential_prompt_injection=decision.potential_prompt_injection,
                    injection_confidence=decision.injection_confidence,
                ),
                True,
            )

    # Rule 5: Promotion Opt-Out Enforced
    user_biz = context.get("user_biz_info", {})
    opted_out = bool(user_biz.get("promotions_opted_out_at")) or user_biz.get("allows_promotions") == "0"
    if opted_out and msg_type == MessageTypeEnum.promotion:
        return (
            RoutingDecision(
                action=ActionEnum.mute,
                message_type=MessageTypeEnum.promotion,
                reason="Gate Override: User has explicitly opted out of marketing promotions from this sender.",
                potential_prompt_injection=decision.potential_prompt_injection,
                injection_confidence=decision.injection_confidence,
            ),
            True,
        )

    # Rule 6: Group Muted by User (no @mention override)
    if group_muted and not user_mentioned:
        if action == ActionEnum.notify:
            return (
                RoutingDecision(
                    action=ActionEnum.digest,
                    message_type=msg_type,
                    reason=f"Gate Override: Group is muted by user and contains no direct mention. Downgraded notify -> digest. Original reason: {reason}",
                    potential_prompt_injection=decision.potential_prompt_injection,
                    injection_confidence=decision.injection_confidence,
                ),
                True,
            )

    # Rule 7: DND Window Active
    if is_dnd and action == ActionEnum.notify:
        return (
            RoutingDecision(
                action=ActionEnum.digest,
                message_type=msg_type,
                reason=f"Gate Override: Message arrived during user DND window. Downgraded notify -> digest. Original reason: {reason}",
                potential_prompt_injection=decision.potential_prompt_injection,
                injection_confidence=decision.injection_confidence,
            ),
            True,
        )

    # Model Decision Stands
    return decision, False


def _contains_regex_injection(text: str) -> bool:
    if not text:
        return False
    from config import INJECTION_REGEX_PATTERNS
    import re
    return any(re.search(p, text) for p in INJECTION_REGEX_PATTERNS)
