import logging
import math
import re
import threading
from contextlib import contextmanager
from datetime import datetime
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
    "no_gloves": "No Gloves Detected",
    "no_helmet": "No Helmet Detected",
    "Hardhat": "Helmet Detected",
    "Safety Vest": "Safety Vest Detected",
    "Mask": "Mask Detected",
    "boots": "Boots Detected",
    "goggles": "Goggles Detected",
    "gloves": "Gloves Detected",
}

# Dropdown values for /detection-alerts?class=
# `terms` are substrings so combined rows ("Hardhat & Gloves") still match,
# and a lone missing helmet stored as "Helmet Detected" still matches no_helmet.
ALERT_CLASSES = [
    {
        "class": "no_helmet",
        "label": "No Helmet",
        "terms": ["Helmet", "Hardhat"],
        "aliases": ["no_helmet", "helmet", "hardhat", "NO-Hardhat", "No Helmet Detected", "Helmet Detected"],
    },
    {
        "class": "no_safety_vest",
        "label": "No Safety Vest",
        "terms": ["Vest"],
        "aliases": ["no_safety_vest", "no_vest", "vest", "NO-Safety Vest", "No Safety Vest Detected", "Safety Vest"],
    },
    {
        "class": "no_mask",
        "label": "No Mask",
        "terms": ["Mask"],
        "aliases": ["no_mask", "mask", "NO-Mask", "No Mask Detected"],
    },
    {
        "class": "no_boots",
        "label": "No Boots",
        "terms": ["Boot"],
        "aliases": ["no_boots", "boots", "No Boots Detected"],
    },
    {
        "class": "no_goggles",
        "label": "No Goggles",
        "terms": ["Goggle"],
        "aliases": ["no_goggles", "no_goggle", "goggles", "No Goggles Detected"],
    },
    {
        "class": "no_gloves",
        "label": "No Gloves",
        "terms": ["Glove"],
        "aliases": ["no_gloves", "no_glove", "gloves", "No Gloves Detected"],
    },
]


def format_detection_event(label: str) -> str:
    return EVENT_LABELS.get(label, label.replace("_", " ").replace("-", " ").title())


def _class_token(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.strip().lower())


def list_alert_classes() -> list[dict]:
    """Classes a client can put in a dropdown and pass back as ?class=."""
    return [{"class": item["class"], "label": item["label"]} for item in ALERT_CLASSES]


def resolve_alert_classes(values: list[str]) -> list[str]:
    """Map dropdown ids or raw class names to event substrings to search for."""
    index = {}
    for item in ALERT_CLASSES:
        for alias in [item["class"], item["label"], *item["aliases"], *item["terms"]]:
            index.setdefault(_class_token(alias), item)

    matched = []
    seen = set()
    for value in values:
        token = value.strip()
        if not token:
            continue
        item = index.get(_class_token(token))
        if item:
            candidates = item["terms"]
        else:
            formatted = format_detection_event(token)
            candidates = [formatted] if formatted != token else [token]
        for term in candidates:
            if term not in seen:
                seen.add(term)
                matched.append(term)
    return matched


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


def query_detection_alerts(
    status: Optional[str] = None,
    events: Optional[list[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    page: int = 1,
    limit: int = 50,
) -> dict:
    """Fetch one page of alerts filtered by status, class/event, and time range."""
    where = {}
    if status:
        where["status"] = status
    if events:
        # Substring match so "no_helmet" also finds "Hardhat & Gloves" and "Helmet Detected".
        where["OR"] = [{"event": {"contains": term}} for term in events]

    time_filter = {}
    if start is not None:
        time_filter["gte"] = start
    if end is not None:
        time_filter["lte"] = end
    if time_filter:
        where["time"] = time_filter

    skip = (page - 1) * limit
    with get_db() as db:
        total = db.detectionalert.count(where=where)
        alerts = db.detectionalert.find_many(
            where=where,
            order={"time": "desc"},
            skip=skip,
            take=limit,
        )

    total_pages = math.ceil(total / limit) if total else 0
    data = [alert.model_dump(mode="json") for alert in alerts]
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "totalPages": total_pages,
        "data": data,
    }


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
