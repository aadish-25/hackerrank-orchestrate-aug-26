import csv
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from config import (
    MESSAGES_CSV,
    SAMPLE_MESSAGES_CSV,
    USERS_CSV,
    GROUPS_CSV,
    GROUP_MEMBERS_CSV,
    BUSINESS_ACCOUNTS_CSV,
    USER_BUSINESS_HISTORY_CSV,
    MESSAGE_HISTORY_CSV,
    MESSAGE_EVENTS_CSV,
    IMAGES_CSV,
    VOICE_NOTES_CSV,
    DAILY_SUMMARY_CSV,
    INJECTION_REGEX_PATTERNS,
    DOMAIN_AGE_FRAUD_DAYS,
)


def load_csv(path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


class DatasetLoader:
    def __init__(self):
        self.users = {r["user_id"]: r for r in load_csv(USERS_CSV)}
        self.groups = {r["group_id"]: r for r in load_csv(GROUPS_CSV)}
        self.group_members = {
            (r["group_id"], r["user_id"]): r for r in load_csv(GROUP_MEMBERS_CSV)
        }
        self.business_accounts = {
            r["business_id"]: r for r in load_csv(BUSINESS_ACCOUNTS_CSV)
        }
        self.user_business_history = {
            (r["user_id"], r["business_id"]): r
            for r in load_csv(USER_BUSINESS_HISTORY_CSV)
        }
        self.message_history = load_csv(MESSAGE_HISTORY_CSV)
        self.message_events = {
            (r["user_id"], r["message_id"]): r for r in load_csv(MESSAGE_EVENTS_CSV)
        }
        self.images = {r["image_id"]: r["file_path"] for r in load_csv(IMAGES_CSV)}
        self.voice_notes = {
            r["voice_note_id"]: r["file_path"] for r in load_csv(VOICE_NOTES_CSV)
        }
        self.daily_summary = load_csv(DAILY_SUMMARY_CSV)

    def get_context_for_message(self, msg_row: Dict[str, str]) -> Dict[str, Any]:
        msg_id = msg_row.get("message_id", "")
        user_id = msg_row.get("user_id", "")
        group_id = msg_row.get("group_id", "")
        business_id = msg_row.get("business_id", "")
        sender_id = msg_row.get("sender_user_id", "")
        created_at_str = msg_row.get("created_at", "")
        msg_text = msg_row.get("message_text", "")
        media_type = msg_row.get("media_type", "")
        media_id = msg_row.get("media_id", "")
        forwarded_count = int(msg_row.get("forwarded_count", 0) or 0)

        # Context lookups
        user_info = self.users.get(user_id, {})
        group_info = self.groups.get(group_id, {})
        group_member_info = self.group_members.get((group_id, user_id), {})
        sender_member_info = self.group_members.get((group_id, sender_id), {})
        business_info = self.business_accounts.get(business_id, {})
        user_biz_info = self.user_business_history.get((user_id, business_id), {})

        # Media path lookup
        media_file_path = ""
        if media_type == "image":
            media_file_path = self.images.get(media_id, "")
        elif media_type == "voice":
            media_file_path = self.voice_notes.get(media_id, "")

        # Signal 1: DND Window Check
        is_dnd = self._check_dnd(user_info.get("do_not_disturb_window", ""), created_at_str)

        # Signal 2: Group Muted & User Mention
        group_muted = group_member_info.get("group_muted_by_user") == "1"
        mention_pattern = rf"@{re.escape(user_id)}\b"
        user_mentioned = bool(re.search(mention_pattern, msg_text, re.IGNORECASE))

        # Signal 3: Domain Fraud Check
        is_domain_fraud = self._check_domain_fraud(business_info)

        # Signal 4: Layer 2 Regex Injection
        regex_injection_detected = any(
            re.search(pattern, msg_text) for pattern in INJECTION_REGEX_PATTERNS
        )

        return {
            "message_id": msg_id,
            "user_id": user_id,
            "conversation_type": msg_row.get("conversation_type", ""),
            "group_id": group_id,
            "business_id": business_id,
            "sender_user_id": sender_id,
            "created_at": created_at_str,
            "message_text": msg_text,
            "media_type": media_type,
            "media_id": media_id,
            "media_file_path": media_file_path,
            "forwarded_count": forwarded_count,
            "user_info": user_info,
            "group_info": group_info,
            "group_member_info": group_member_info,
            "sender_member_info": sender_member_info,
            "business_info": business_info,
            "user_biz_info": user_biz_info,
            "is_dnd": is_dnd,
            "group_muted": group_muted,
            "user_mentioned": user_mentioned,
            "is_domain_fraud": is_domain_fraud,
            "regex_injection_detected": regex_injection_detected,
        }

    def _check_dnd(self, dnd_window: str, created_at_str: str) -> bool:
        if not dnd_window or not created_at_str:
            return False
        try:
            msg_dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M")
            start_str, end_str = dnd_window.split("-")
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()

            msg_time = msg_dt.time()
            if start_time <= end_time:
                return start_time <= msg_time <= end_time
            else:  # Overnight window (e.g. 22:00-07:00)
                return msg_time >= start_time or msg_time <= end_time
        except Exception:
            return False

    def _check_domain_fraud(self, biz_info: Dict[str, str]) -> bool:
        if not biz_info:
            return False
        off_domain = biz_info.get("official_domain", "").strip()
        sender_domain = biz_info.get("domain_used_by_sender", "").strip()
        try:
            domain_age = int(biz_info.get("domain_used_by_sender_age_days", 999) or 999)
        except Exception:
            domain_age = 999

        unverified = biz_info.get("verified") == "0"
        mismatch = bool(off_domain and sender_domain and off_domain != sender_domain)

        return (mismatch and domain_age < DOMAIN_AGE_FRAUD_DAYS) or (
            unverified and domain_age < DOMAIN_AGE_FRAUD_DAYS
        )
