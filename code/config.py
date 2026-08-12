import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "cache"

MESSAGES_CSV = DATASET_DIR / "messages.csv"
SAMPLE_MESSAGES_CSV = DATASET_DIR / "sample_messages.csv"
USERS_CSV = DATASET_DIR / "users.csv"
GROUPS_CSV = DATASET_DIR / "groups.csv"
GROUP_MEMBERS_CSV = DATASET_DIR / "group_members.csv"
BUSINESS_ACCOUNTS_CSV = DATASET_DIR / "business_accounts.csv"
USER_BUSINESS_HISTORY_CSV = DATASET_DIR / "user_business_history.csv"
MESSAGE_HISTORY_CSV = DATASET_DIR / "message_history.csv"
MESSAGE_EVENTS_CSV = DATASET_DIR / "message_events.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
VOICE_NOTES_CSV = DATASET_DIR / "voice_notes.csv"
DAILY_SUMMARY_CSV = DATASET_DIR / "daily_notification_summary.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"

# Checkpoint Files
CHECKPOINT_MEDIA_FILE = CACHE_DIR / "checkpoint_media.json"
CHECKPOINT_ROUTING_FILE = CACHE_DIR / "checkpoint_routing.json"
CHECKPOINT_EVAL_FILE = CACHE_DIR / "checkpoint_eval.json"

# Models
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama-3.3-70b-versatile")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen2.5-7B-Instruct")
LOCAL_MODEL_URL = os.getenv("LOCAL_MODEL_URL", "http://localhost:1234/v1")

MAX_RETRIES = 3
MAX_ITERATIONS = 5
SOFT_NUDGE_AT_ITERATION = 4

# Evidence Selection Weights & Thresholds
MUTE_SIGNAL_WEIGHTS = {
    "notification_dismissed": 0.35,
    "muted_after_message": 0.35,
    "message_reported": 0.30,
}

NOTIFY_SIGNAL_WEIGHTS = {
    "message_replied": 0.40,
    "fast_reaction_time": 0.35,
    "message_opened": 0.25,
}

REACTION_TIME_FAST_THRESHOLD_MINUTES = 10
EVIDENCE_MAX_COUNT = 3

# Confidence Scoring Weights
CONFIDENCE_WEIGHTS = {
    "business_verified": 0.20,
    "opt_in_active": 0.15,
    "prior_engagement_positive": 0.20,
    "no_injection_flag": 0.15,
    "domain_match": 0.15,
    "sender_trust": 0.15,
}

GATE_OVERRIDE_CONFIDENCE = 0.92

# Safety Gate & Fraud Detection Thresholds
INJECTION_CONFIDENCE_THRESHOLD = 0.75
DOMAIN_AGE_FRAUD_DAYS = 30
FORWARD_COUNT_MUTE_THRESHOLD = 8

# Regex Injection Patterns (Layer 2 Backstop)
INJECTION_REGEX_PATTERNS = [
    r"(?i)assistant\s+instruction",
    r"(?i)routing\s+override",
    r"(?i)set\s+action\s*=",
    r"(?i)ignore\s+(sender\s+risk|previous|all\s+previous)",
    r"(?i)classify\s+as\s+(urgent|notify)",
    r"(?i)mark\s+(this\s+)?as\s+notify",
    r"(?i)system\s+note\s+for\s+(the\s+)?notification\s+router",
    r"(?i)internal\s+router\s+metadata",
]
