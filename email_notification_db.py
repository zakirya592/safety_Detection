import logging
import os
import re
from contextlib import contextmanager
from typing import Optional

from dotenv import load_dotenv
from prisma import Prisma

load_dotenv()

logger = logging.getLogger(__name__)

SETTINGS_ID = 1
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@contextmanager
def get_db():
    db = Prisma()
    db.connect()
    try:
        yield db
    finally:
        db.disconnect()


def parse_emails(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [email.strip() for email in str(value).split(",") if email.strip()]


def validate_receiving_email(value: str) -> str:
    emails = parse_emails(value)
    if not emails:
        raise ValueError("receivingEmail is required")
    invalid = [email for email in emails if not EMAIL_RE.match(email)]
    if invalid:
        raise ValueError(f"Invalid email address: {', '.join(invalid)}")
    return ", ".join(emails)


def _serialize_setting(setting) -> dict:
    data = setting.model_dump(mode="json")
    return {
        "receivingEmail": data.get("receivingEmail") or "",
        "attachCapturedPhotos": bool(data.get("attachCapturedPhotos", True)),
        "enabled": bool(data.get("enabled", True)),
        "updatedAt": data.get("updatedAt"),
    }


def _env_fallback_settings() -> dict:
    receiving_email = os.environ.get("ALERT_EMAIL_TO", "").strip()
    return {
        "receivingEmail": receiving_email,
        "attachCapturedPhotos": True,
        "enabled": True,
        "updatedAt": None,
    }


def get_email_settings() -> dict:
    """Return saved notification settings, falling back to ALERT_EMAIL_TO."""
    try:
        with get_db() as db:
            setting = db.emailnotificationsetting.find_unique(where={"id": SETTINGS_ID})
            if setting is None:
                return _env_fallback_settings()
            return _serialize_setting(setting)
    except Exception as exc:
        logger.warning("Could not load email notification settings: %s", exc)
        return _env_fallback_settings()


def save_email_settings(
    receiving_email: str,
    attach_captured_photos: Optional[bool] = None,
    enabled: Optional[bool] = None,
) -> dict:
    """Create or update the single notification-settings row."""
    normalized_email = validate_receiving_email(receiving_email)
    current = get_email_settings()
    photos = (
        current.get("attachCapturedPhotos", True)
        if attach_captured_photos is None
        else bool(attach_captured_photos)
    )
    is_enabled = current.get("enabled", True) if enabled is None else bool(enabled)

    with get_db() as db:
        setting = db.emailnotificationsetting.upsert(
            where={"id": SETTINGS_ID},
            data={
                "create": {
                    "id": SETTINGS_ID,
                    "receivingEmail": normalized_email,
                    "attachCapturedPhotos": photos,
                    "enabled": is_enabled,
                },
                "update": {
                    "receivingEmail": normalized_email,
                    "attachCapturedPhotos": photos,
                    "enabled": is_enabled,
                },
            },
        )
        saved = _serialize_setting(setting)
        logger.info("Email notification settings saved for %s", saved["receivingEmail"])
        return saved


def log_email_activity(
    event: str,
    location: str,
    camera: str,
    recipient: str,
    status: str,
    image_url: Optional[str] = None,
) -> None:
    try:
        with get_db() as db:
            db.emailnotificationlog.create(
                data={
                    "event": event[:200],
                    "location": location[:200],
                    "camera": camera[:100],
                    "recipient": recipient[:255],
                    "status": status[:20],
                    "image_url": image_url,
                }
            )
    except Exception as exc:
        logger.warning("Could not save email notification activity: %s", exc)


def get_recent_email_activity(limit: int = 20) -> list[dict]:
    try:
        with get_db() as db:
            logs = db.emailnotificationlog.find_many(
                order={"createdAt": "desc"},
                take=limit,
            )
            return [log.model_dump(mode="json") for log in logs]
    except Exception as exc:
        logger.warning("Could not load email notification activity: %s", exc)
        return []
