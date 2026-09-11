import logging
import threading
from contextlib import contextmanager
from typing import Literal, Optional

from alert_notifier import notify_detection_alerts_async
from dotenv import load_dotenv
from prisma import Prisma

load_dotenv()

logger = logging.getLogger(__name__)

AlertStatus = Literal["New", "Cleared"]

EVENT_LABELS = {
    "NO-Hardhat": "No Helmet Detected",
    "NO-Safety Vest": "No Safety Vest Detected",
    "NO-Mask": "No Mask Detected",
    "no_boots": "No Boots Detected",
    "no_goggle": "No Goggles Detected",
    "no_goggles": "No Goggles Detected",
    "no_glove": "No Gloves Detected",
    "no_gloves": "No Gloves Detected",
    "no_helmet": "No Helmet Detected",
    "Hardhat": "Helmet Detected",
    "Safety Vest": "Safety Vest Detected",
    "Mask": "Mask Detected",
    "boots": "Boots Detected",
    "goggles": "Goggles Detected",
    "glove": "Gloves Detected",
    "gloves": "Gloves Detected",
}


def format_detection_event(label: str) -> str:
    return EVENT_LABELS.get(label, label.replace("_", " ").replace("-", " ").title())


@contextmanager
def get_db():
    db = Prisma()
    db.connect()
    try:
        yield db
    finally:
        db.disconnect()


def create_detection_alert(
    camera: str,
    location: str,
    event: str,
    confidence: float,
    image_url: Optional[str] = None,
    status: AlertStatus = "New",
) -> dict:
    """Create a new detection alert record."""
    with get_db() as db:
        alert = db.detectionalert.create(
            data={
                "camera": camera,
                "location": location,
                "event": event,
                "confidence": confidence,
                "image_url": image_url,
                "status": status,
            }
        )
        return alert.model_dump()


def save_detection_alerts(
    camera: str,
    location: str,
    persons: list[dict],
    image_url: Optional[str] = None,
) -> list[dict]:
    """Save one alert row per detected violation person."""
    saved = []
    for person in persons:
        try:
            alert = create_detection_alert(
                camera=camera,
                location=location,
                event=format_detection_event(person["label"]),
                confidence=round(person["confidence"] * 100, 1),
                image_url=image_url,
            )
            saved.append(alert)
            logger.info(
                "Alert saved: %s | %s | %.1f%% | %s",
                camera,
                alert["event"],
                alert["confidence"],
                image_url or "no image",
            )
        except Exception as exc:
            logger.error("Failed to save alert for %s: %s", person.get("label"), exc)

    if saved:
        notify_detection_alerts_async(
            camera=camera,
            location=location,
            alerts=saved,
            image_url=image_url,
        )
    return saved


def save_detection_alerts_async(
    camera: str,
    location: str,
    persons: list[dict],
    image_url: Optional[str] = None,
) -> threading.Thread:
    """Save alerts in a background thread so detection is not blocked."""

    def _worker():
        save_detection_alerts(camera, location, persons, image_url)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def get_all_alerts(status: Optional[AlertStatus] = None) -> list[dict]:
    """Fetch all detection alerts, optionally filtered by status."""
    with get_db() as db:
        where = {"status": status} if status else {}
        alerts = db.detectionalert.find_many(
            where=where,
            order={"time": "desc"},
        )
        return [alert.model_dump(mode="json") for alert in alerts]


def get_recent_alerts(limit: int = 50, status: Optional[AlertStatus] = None) -> list[dict]:
    """Fetch recent detection alerts, optionally filtered by status."""
    with get_db() as db:
        where = {"status": status} if status else {}
        alerts = db.detectionalert.find_many(
            where=where,
            order={"time": "desc"},
            take=limit,
        )
        return [alert.model_dump() for alert in alerts]


def update_alert_status(alert_id: int, status: AlertStatus) -> dict:
    """Update the status of an existing alert."""
    with get_db() as db:
        alert = db.detectionalert.update(
            where={"id": alert_id},
            data={"status": status},
        )
        return alert.model_dump()


def clear_alert(alert_id: int) -> dict:
    """Mark an alert as Cleared."""
    return update_alert_status(alert_id, "Cleared")
