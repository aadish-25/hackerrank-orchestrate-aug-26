import json
from pathlib import Path
from typing import Dict, Any, Optional

from config import (
    CACHE_DIR,
    CHECKPOINT_MEDIA_FILE,
    CHECKPOINT_ROUTING_FILE,
    CHECKPOINT_EVAL_FILE,
)


class CheckpointManager:
    """
    Manages three separate JSON checkpoint files:
      1. checkpoint_media.json   (media pre-processing)
      2. checkpoint_routing.json (main routing LLM decisions)
      3. checkpoint_eval.json    (sample_messages.csv ground truth eval runs)
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, checkpoint_type: str) -> Path:
        if checkpoint_type == "media":
            return CHECKPOINT_MEDIA_FILE
        elif checkpoint_type == "routing":
            return CHECKPOINT_ROUTING_FILE
        elif checkpoint_type == "eval":
            return CHECKPOINT_EVAL_FILE
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

    def load(self, checkpoint_type: str) -> Dict[str, Any]:
        path = self._get_path(checkpoint_type)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self, checkpoint_type: str, data: Dict[str, Any]) -> None:
        path = self._get_path(checkpoint_type)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_record(self, checkpoint_type: str, key: str) -> Optional[Dict[str, Any]]:
        data = self.load(checkpoint_type)
        return data.get(key)

    def set_record(
        self,
        checkpoint_type: str,
        key: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        data = self.load(checkpoint_type)
        entry: Dict[str, Any] = {"status": status}
        if metadata is not None:
            entry["metadata"] = metadata
        if result is not None:
            entry["result"] = result
        if error is not None:
            entry["error"] = error
        data[key] = entry
        self.save(checkpoint_type, data)

    def is_done(self, checkpoint_type: str, key: str) -> bool:
        rec = self.get_record(checkpoint_type, key)
        return rec is not None and rec.get("status") == "done"

    def clear(self, checkpoint_type: str) -> None:
        path = self._get_path(checkpoint_type)
        if path.exists():
            path.unlink()

    def handle_flags(
        self, force_clean: bool, force_media: bool, force_routing: bool
    ) -> None:
        """
        Applies checkpoint invalidation rules:
        - force_clean: wipes media, routing, and eval
        - force_media: wipes media
        - force_routing: wipes routing AND eval
        """
        if force_clean:
            self.clear("media")
            self.clear("routing")
            self.clear("eval")
        else:
            if force_media:
                self.clear("media")
            if force_routing:
                self.clear("routing")
                self.clear("eval")
