"""
API 비용 추적 — Claude (Sonnet 4.6) + Voyage AI (multilingual-2)
/data/costs.json 에 누적 저장 (서버 재시작에도 유지)
"""
import json
import threading
from datetime import datetime
from pathlib import Path

# ── 단가 (USD / token) ────────────────────────────────────────────────────────
_CLAUDE_IN  = 3.00  / 1_000_000   # claude-sonnet-4-6 input
_CLAUDE_OUT = 15.00 / 1_000_000   # claude-sonnet-4-6 output
_VOYAGE     = 0.12  / 1_000_000   # voyage-multilingual-2

_lock = threading.Lock()
_data: dict = {
    "since": None,
    "claude": {"input_tokens": 0, "output_tokens": 0},
    "voyage": {"tokens": 0},
}
_path: Path | None = None


def init(persist_path: Path) -> None:
    global _data, _path
    _path = persist_path
    if persist_path.exists():
        try:
            loaded = json.loads(persist_path.read_text(encoding="utf-8"))
            _data.update(loaded)
        except Exception:
            pass
    if not _data.get("since"):
        _data["since"] = datetime.now().isoformat(timespec="seconds")
        _flush()


def record_claude(input_tokens: int, output_tokens: int) -> None:
    with _lock:
        _data["claude"]["input_tokens"]  += input_tokens
        _data["claude"]["output_tokens"] += output_tokens
        _flush()


def record_voyage(tokens: int) -> None:
    with _lock:
        _data["voyage"]["tokens"] += tokens
        _flush()


def reset() -> None:
    with _lock:
        _data["claude"] = {"input_tokens": 0, "output_tokens": 0}
        _data["voyage"] = {"tokens": 0}
        _data["since"]  = datetime.now().isoformat(timespec="seconds")
        _flush()


def _flush() -> None:
    if _path:
        try:
            _path.write_text(json.dumps(_data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def get_summary() -> dict:
    with _lock:
        ci = _data["claude"]["input_tokens"]
        co = _data["claude"]["output_tokens"]
        vt = _data["voyage"]["tokens"]
        claude_cost = ci * _CLAUDE_IN + co * _CLAUDE_OUT
        voyage_cost = vt * _VOYAGE
        total       = claude_cost + voyage_cost
        return {
            "since": _data.get("since"),
            "claude": {
                "input_tokens":  ci,
                "output_tokens": co,
                "cost_usd": round(claude_cost, 4),
            },
            "voyage": {
                "tokens":   vt,
                "cost_usd": round(voyage_cost, 4),
            },
            "total_cost_usd": round(total, 4),
        }
