"""
tools/lead_capture.py

Lead capture tool.
- Stores qualified leads in memory during the session.
- Persists every captured lead to leads.json on disk so data survives restarts.
- Exposes a LangChain @tool wrapper (capture_lead) that the LangGraph agent calls.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

# Write leads.json to /data inside the container (mounted volume) if it exists,
# otherwise fall back to the project root so local runs work out of the box.
_DATA_DIR = Path("/data") if Path("/data").exists() else Path(__file__).parent.parent
_LEADS_FILE = _DATA_DIR / "leads.json"

_captured_leads: list[dict] = []
_last_disk_refresh: float = 0.0
_REFRESH_INTERVAL = 5.0  # Refresh from disk every 5 seconds


def _load_existing_leads() -> list[dict]:
    """Load leads persisted from previous sessions."""
    if _LEADS_FILE.exists():
        try:
            with open(_LEADS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load leads from disk: {e}")
            return []
        except Exception as e:
            print(f"Warning: Unexpected error loading leads: {e}")
            return []
    return []


def _save_leads(leads: list[dict]) -> None:
    """Persist the full lead list to disk atomically."""
    try:
        _LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _LEADS_FILE.with_suffix(".tmp")

        # Write to temporary file
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(leads, f, indent=2)

        # Atomic rename with retry for Windows locking issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tmp.replace(_LEADS_FILE)
                break
            except (OSError, FileExistsError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                else:
                    print(f"Error: Failed to save leads after {max_retries} attempts: {e}")
                    raise
    except Exception as e:
        print(f"Error: Failed to persist leads to disk: {e}")
        raise


def _refresh_leads_from_disk() -> None:
    """Refresh in-memory leads from disk if enough time has passed."""
    global _captured_leads, _last_disk_refresh

    now = time.time()
    if now - _last_disk_refresh >= _REFRESH_INTERVAL:
        try:
            disk_leads = _load_existing_leads()
            if disk_leads and len(disk_leads) > len(_captured_leads):
                # Disk has more leads, sync them
                _captured_leads = disk_leads
            _last_disk_refresh = now
        except Exception as e:
            print(f"Warning: Failed to refresh leads from disk: {e}")


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Capture a qualified lead, store it in memory and persist to leads.json.

    Args:
        name:     Full name of the prospect.
        email:    Email address of the prospect.
        platform: Primary creator platform (YouTube, Instagram, TikTok, etc.).

    Returns:
        dict with success flag, lead_id, message, and full lead record.
    """
    global _captured_leads, _last_disk_refresh

    # Hydrate from disk on first call so we maintain correct sequential IDs
    if not _captured_leads:
        _captured_leads = _load_existing_leads()
        _last_disk_refresh = time.time()

    lead = {
        "id": f"LEAD-{len(_captured_leads) + 1:04d}",
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "status": "new",
    }
    _captured_leads.append(lead)
    _save_leads(_captured_leads)

    # Required print per assignment spec
    print(f"Lead captured successfully: {name}, {email}, {platform}")

    return {
        "success": True,
        "lead_id": lead["id"],
        "message": f"Lead captured successfully for {name}.",
        "lead": lead,
    }


def get_all_leads() -> list[dict]:
    """Return all in-memory leads, loading from disk if needed."""
    global _captured_leads

    if not _captured_leads:
        _captured_leads = _load_existing_leads()
    else:
        _refresh_leads_from_disk()

    return list(_captured_leads)


# ── LangChain @tool wrapper ────────────────────────────────────────────────────

@tool
def capture_lead(name: str, email: str, platform: str) -> str:
    """
    Capture a qualified lead after collecting name, email, and creator platform.
    Only call this tool once you have confirmed ALL THREE values from the user.

    Args:
        name:     Full name of the prospect.
        email:    Email address of the prospect.
        platform: Primary creator platform (e.g. YouTube, Instagram, TikTok).

    Returns:
        Plain-text confirmation message.
    """
    result = mock_lead_capture(name=name, email=email, platform=platform)
    return result["message"] if result["success"] else "Lead capture failed — please try again."

