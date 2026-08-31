import logging
import os
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class _PyWhatKitSession:
    """Reuse one WhatsApp Web tab and auto-send messages via keyboard automation."""

    _lock = threading.Lock()
    _phone: Optional[str] = None

    @classmethod
    def _normalize_phone(cls, phone: str) -> str:
        normalized = phone.replace(" ", "")
        if not normalized.startswith("+"):
            normalized = f"+{normalized}"
        return normalized

    @classmethod
    def _build_chat_url(cls, phone: str) -> str:
        phone_digits = cls._normalize_phone(phone).lstrip("+")
        return f"https://web.whatsapp.com/send?phone={phone_digits}"

    @classmethod
    def _find_whatsapp_window(cls):
        try:
            import pygetwindow as gw
        except ImportError:
            logger.warning("pygetwindow not available; cannot reuse WhatsApp tab")
            return None

        matches = []
        for window in gw.getAllWindows():
            title = (window.title or "").strip()
            if not title:
                continue
            lower = title.lower()
            if "whatsapp" in lower and window.width > 200 and window.height > 200:
                matches.append(window)

        if not matches:
            return None

        matches.sort(key=lambda window: window.width * window.height, reverse=True)
        return matches[0]

    @classmethod
    def _activate_window(cls, window) -> bool:
        try:
            if window.isMinimized:
                window.restore()
            window.activate()
            time.sleep(0.8)
            return True
        except Exception as exc:
            logger.warning("WhatsApp: could not focus window '%s' (%s)", window.title, exc)
            return False

    @classmethod
    def _navigate_to_url(cls, window, url: str) -> None:
        import pyautogui
        import pyperclip

        cls._activate_window(window)
        logger.info("WhatsApp: navigating to chat URL")

        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.4)
        pyperclip.copy(url)
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.2)
        pyautogui.press("enter")

    @classmethod
    def _click_compose_box(cls, window) -> None:
        import pyautogui

        compose_x = window.left + int(window.width * 0.72)
        compose_y = window.top + window.height - 72
        pyautogui.click(compose_x, compose_y)
        logger.info("WhatsApp: focused compose box")
        time.sleep(0.4)

    @classmethod
    def _click_send_button(cls, window) -> None:
        import pyautogui

        send_x = window.left + window.width - 55
        send_y = window.top + window.height - 48
        pyautogui.click(send_x, send_y)
        logger.info("WhatsApp: clicked send button")

    @classmethod
    def _paste_and_send(cls, window, message: str) -> None:
        import pyautogui
        import pyperclip

        cls._activate_window(window)
        cls._click_compose_box(window)

        pyperclip.copy(message)
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.8)

        pyautogui.press("enter")
        time.sleep(0.4)

        cls._click_send_button(window)
        time.sleep(0.3)
        pyautogui.press("enter")

        logger.info("WhatsApp: message send triggered")

    @classmethod
    def _open_whatsapp_if_needed(cls, wait_time: int):
        window = cls._find_whatsapp_window()
        if window:
            logger.info("WhatsApp: using existing browser window '%s'", window.title)
            return window

        logger.info("WhatsApp: opening WhatsApp Web (first time only)")
        webbrowser.open("https://web.whatsapp.com", new=2)
        logger.info("WhatsApp: waiting %ss for WhatsApp Web to load — scan QR if needed", wait_time)
        time.sleep(wait_time)
        return cls._find_whatsapp_window()

    @classmethod
    def _send_in_window(cls, window, phone: str, message: str, wait_time: int) -> bool:
        chat_url = cls._build_chat_url(phone)
        cls._navigate_to_url(window, chat_url)

        logger.info("WhatsApp: waiting %ss for chat to open", wait_time)
        time.sleep(wait_time)

        window = cls._find_whatsapp_window() or window
        cls._paste_and_send(window, message)
        return True

    @classmethod
    def send(cls, phone: str, message: str, wait_time: int, reuse_tab: bool) -> bool:
        phone = cls._normalize_phone(phone)

        with cls._lock:
            try:
                if reuse_tab:
                    window = cls._find_whatsapp_window()
                    if window:
                        logger.info("WhatsApp: reusing tab in '%s'", window.title)
                        cls._phone = phone
                        return cls._send_in_window(window, phone, message, wait_time)

                window = cls._open_whatsapp_if_needed(wait_time)
                if window is None:
                    logger.error("WhatsApp: browser window not found after opening WhatsApp Web")
                    return False

                cls._phone = phone
                sent = cls._send_in_window(window, phone, message, wait_time)
                if sent:
                    logger.info("WhatsApp: session ready — next alerts reuse this same tab")
                return sent
            except Exception as exc:
                logger.error("WhatsApp: send failed (%s)", exc)
                return False


class WhatsAppNotifier:
    """Send formatted safety alert messages via WhatsApp (PyWhatKit, Twilio, or CallMeBot)."""

    _pywhatkit_lock = threading.Lock()

    def __init__(
        self,
        provider: Optional[str] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        whatsapp_from: Optional[str] = None,
        whatsapp_to: Optional[str] = None,
        callmebot_phone: Optional[str] = None,
        callmebot_api_key: Optional[str] = None,
        pywhatkit_phone: Optional[str] = None,
        pywhatkit_wait_time: Optional[int] = None,
        pywhatkit_reuse_tab: Optional[bool] = None,
    ):
        self.provider = (provider or os.environ.get("WHATSAPP_PROVIDER", "pywhatkit")).lower()
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.whatsapp_from = whatsapp_from or os.environ.get("TWILIO_WHATSAPP_FROM")
        self.whatsapp_to = whatsapp_to or os.environ.get("WHATSAPP_ALERT_TO")
        self.callmebot_phone = callmebot_phone or os.environ.get("CALLMEBOT_PHONE")
        self.callmebot_api_key = callmebot_api_key or os.environ.get("CALLMEBOT_API_KEY")
        self.pywhatkit_phone = (
            pywhatkit_phone
            or os.environ.get("PYWHATKIT_PHONE")
            or self.whatsapp_to
            or self.callmebot_phone
        )
        self.pywhatkit_wait_time = int(
            pywhatkit_wait_time or os.environ.get("PYWHATKIT_WAIT_TIME", "15")
        )
        reuse_tab_env = os.environ.get("PYWHATKIT_REUSE_TAB", "true").lower()
        self.pywhatkit_reuse_tab = (
            pywhatkit_reuse_tab
            if pywhatkit_reuse_tab is not None
            else reuse_tab_env in {"1", "true", "yes"}
        )

        alerts_enabled = os.environ.get("WHATSAPP_ALERTS_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
        }

        if self.provider == "twilio":
            configured = bool(
                self.account_sid and self.auth_token and self.whatsapp_from and self.whatsapp_to
            )
        elif self.provider == "callmebot":
            configured = bool(self.callmebot_phone and self.callmebot_api_key)
        elif self.provider == "pywhatkit":
            configured = bool(self.pywhatkit_phone)
        else:
            self.provider = "pywhatkit"
            configured = bool(self.pywhatkit_phone)

        self.enabled = configured and alerts_enabled

        if self.enabled:
            logger.info(
                "WhatsApp notifier enabled (provider: %s, reuse_tab: %s, wait: %ss)",
                self.provider,
                self.pywhatkit_reuse_tab,
                self.pywhatkit_wait_time,
            )
        elif configured and not alerts_enabled:
            logger.info("WhatsApp notifier paused (WHATSAPP_ALERTS_ENABLED=false)")
        else:
            logger.warning(
                "WhatsApp notifier disabled. For PyWhatKit set PYWHATKIT_PHONE. "
                "For Twilio set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
                "TWILIO_WHATSAPP_FROM, WHATSAPP_ALERT_TO. "
                "For CallMeBot set CALLMEBOT_PHONE and CALLMEBOT_API_KEY."
            )

    @staticmethod
    def _format_time(value) -> str:
        if isinstance(value, datetime):
            return value.strftime("%d %b %Y, %I:%M %p")
        if value:
            return str(value)
        return datetime.now().strftime("%d %b %Y, %I:%M %p")

    def build_message(
        self,
        camera: str,
        location: str,
        alerts: list[dict],
        image_url: Optional[str] = None,
    ) -> str:
        alert_time = self._format_time(alerts[0].get("time") if alerts else None)
        violation_lines = "\n".join(
            f"  • *{alert.get('event', 'Violation')}* — {alert.get('confidence', 0):.1f}% confidence"
            for alert in alerts
        )

        lines = [
            "🚨 *PPE SAFETY ALERT* 🚨",
            "━━━━━━━━━━━━━━━━━━━━",
            "",
            f"📍 *Location:* {location}",
            f"📷 *Camera:* {camera}",
            f"🕐 *Time:* {alert_time}",
            f"⚠️ *Violations:* {len(alerts)} detected",
            "",
            "*Detected Issues:*",
            violation_lines,
        ]

        if image_url:
            lines.extend(["", f"📸 *Evidence:* {image_url}"])

        lines.extend(["", "🔴 _Status: New — Action Required_", "", "_AI Safety Detection System_"])
        return "\n".join(lines)

    def _send_via_twilio(self, message: str) -> bool:
        try:
            from twilio.rest import Client
        except ImportError:
            logger.error("Twilio package not installed. Run: pip install twilio")
            return False

        try:
            client = Client(self.account_sid, self.auth_token)
            client.messages.create(
                body=message,
                from_=self.whatsapp_from,
                to=self.whatsapp_to,
            )
            logger.info("WhatsApp alert sent via Twilio to %s", self.whatsapp_to)
            return True
        except Exception as exc:
            logger.error("Failed to send WhatsApp via Twilio: %s", exc)
            return False

    def _send_via_callmebot(self, message: str) -> bool:
        params = urllib.parse.urlencode(
            {
                "phone": self.callmebot_phone,
                "text": message,
                "apikey": self.callmebot_api_key,
            }
        )
        url = f"https://api.callmebot.com/whatsapp.php?{params}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
            logger.info("WhatsApp alert sent via CallMeBot: %s", body[:120])
            return True
        except Exception as exc:
            logger.error("Failed to send WhatsApp via CallMeBot: %s", exc)
            return False

    def _send_via_pywhatkit(self, message: str) -> bool:
        phone = self.pywhatkit_phone.replace(" ", "")
        if not phone.startswith("+"):
            phone = f"+{phone}"

        logger.info(
            "WhatsApp: preparing alert for %s (%d chars)",
            phone,
            len(message),
        )

        try:
            with self._pywhatkit_lock:
                sent = _PyWhatKitSession.send(
                    phone=phone,
                    message=message,
                    wait_time=self.pywhatkit_wait_time,
                    reuse_tab=self.pywhatkit_reuse_tab,
                )
            if sent:
                logger.info("WhatsApp alert sent to %s", phone)
            else:
                logger.error("WhatsApp alert failed for %s", phone)
            return sent
        except Exception as exc:
            logger.error("Failed to send WhatsApp via PyWhatKit session: %s", exc)
            return False

    def send_detection_alert(
        self,
        camera: str,
        location: str,
        alerts: list[dict],
        image_url: Optional[str] = None,
    ) -> bool:
        if not self.enabled or not alerts:
            return False

        message = self.build_message(camera, location, alerts, image_url)
        logger.info(
            "WhatsApp: sending %d violation(s) from %s at %s",
            len(alerts),
            camera,
            location,
        )

        if self.provider == "twilio":
            return self._send_via_twilio(message)
        if self.provider == "callmebot":
            return self._send_via_callmebot(message)
        return self._send_via_pywhatkit(message)

    def send_detection_alert_async(
        self,
        camera: str,
        location: str,
        alerts: list[dict],
        image_url: Optional[str] = None,
    ) -> threading.Thread:
        def _worker():
            self.send_detection_alert(camera, location, alerts, image_url)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread
