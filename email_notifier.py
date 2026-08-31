import logging
import os
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send formatted HTML safety alert emails via SMTP."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        smtp_from: Optional[str] = None,
        alert_to: Optional[str] = None,
    ):
        self.smtp_host = smtp_host or os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(smtp_port or os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = (smtp_user or os.environ.get("SMTP_USER", "")).strip()
        raw_password = smtp_password or os.environ.get("SMTP_PASSWORD", "")
        # Gmail app passwords are often pasted with spaces — remove them.
        self.smtp_password = raw_password.replace(" ", "").strip()
        self.smtp_from = (smtp_from or os.environ.get("SMTP_FROM") or self.smtp_user).strip()
        recipients = alert_to or os.environ.get("ALERT_EMAIL_TO", "")
        self.alert_to = [email.strip() for email in recipients.split(",") if email.strip()]
        alerts_enabled = os.environ.get("EMAIL_ALERTS_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        configured = bool(self.smtp_user and self.smtp_password and self.alert_to)
        self.enabled = configured and alerts_enabled

        if self.enabled:
            logger.info("Email notifier enabled (%d recipient(s))", len(self.alert_to))
        elif configured and not alerts_enabled:
            logger.info("Email notifier paused (EMAIL_ALERTS_ENABLED=false)")
        else:
            logger.warning(
                "Email notifier disabled. Set SMTP_USER, SMTP_PASSWORD, and ALERT_EMAIL_TO."
            )

    @staticmethod
    def _format_time(value) -> str:
        if isinstance(value, datetime):
            return value.strftime("%d %b %Y, %I:%M:%S %p")
        if value:
            return str(value)
        return datetime.now().strftime("%d %b %Y, %I:%M:%S %p")

    def build_subject(self, camera: str, location: str, alerts: list[dict]) -> str:
        events = ", ".join({alert.get("event", "Violation") for alert in alerts})
        return f"🚨 Safety Alert — {events} | {location}"

    def build_html(
        self,
        camera: str,
        location: str,
        alerts: list[dict],
        image_url: Optional[str] = None,
    ) -> str:
        alert_time = self._format_time(alerts[0].get("time") if alerts else None)
        violation_rows = "".join(
            f"""
            <tr>
                <td style="padding:12px 16px;border-bottom:1px solid #f0f0f0;">
                    <span style="color:#dc2626;font-weight:600;">{alert.get("event", "Violation")}</span>
                </td>
                <td style="padding:12px 16px;border-bottom:1px solid #f0f0f0;text-align:center;">
                    <span style="background:#fef2f2;color:#dc2626;padding:4px 10px;border-radius:20px;font-weight:600;">
                        {alert.get("confidence", 0):.1f}%
                    </span>
                </td>
                <td style="padding:12px 16px;border-bottom:1px solid #f0f0f0;text-align:center;">
                    <span style="background:#eff6ff;color:#2563eb;padding:4px 10px;border-radius:20px;font-size:12px;">
                        {alert.get("status", "New")}
                    </span>
                </td>
            </tr>"""
            for alert in alerts
        )

        image_block = ""
        if image_url:
            image_block = f"""
            <div style="margin-top:24px;text-align:center;">
                <p style="color:#6b7280;font-size:13px;margin:0 0 12px;">Violation Evidence</p>
                <a href="{image_url}" target="_blank">
                    <img src="{image_url}" alt="Violation screenshot"
                         style="max-width:100%;border-radius:8px;border:2px solid #fecaca;"/>
                </a>
                <p style="margin-top:8px;">
                    <a href="{image_url}" style="color:#2563eb;font-size:13px;">View full image</a>
                </p>
            </div>"""

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,Helvetica,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,0.07);">
        <tr>
          <td style="background:linear-gradient(135deg,#dc2626,#b91c1c);padding:28px 32px;">
            <h1 style="margin:0;color:#ffffff;font-size:22px;">⚠️ PPE Safety Violation Detected</h1>
            <p style="margin:8px 0 0;color:#fecaca;font-size:14px;">Immediate attention required</p>
          </td>
        </tr>
        <tr>
          <td style="padding:28px 32px;">
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="background:#f9fafb;border-radius:8px;margin-bottom:24px;">
              <tr>
                <td style="padding:16px 20px;">
                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td style="padding:6px 0;color:#6b7280;font-size:13px;width:100px;">📍 Location</td>
                      <td style="padding:6px 0;color:#111827;font-weight:600;">{location}</td>
                    </tr>
                    <tr>
                      <td style="padding:6px 0;color:#6b7280;font-size:13px;">📷 Camera</td>
                      <td style="padding:6px 0;color:#111827;font-weight:600;">{camera}</td>
                    </tr>
                    <tr>
                      <td style="padding:6px 0;color:#6b7280;font-size:13px;">🕐 Time</td>
                      <td style="padding:6px 0;color:#111827;font-weight:600;">{alert_time}</td>
                    </tr>
                    <tr>
                      <td style="padding:6px 0;color:#6b7280;font-size:13px;">🔢 Violations</td>
                      <td style="padding:6px 0;color:#dc2626;font-weight:700;">{len(alerts)} detected</td>
                    </tr>
                  </table>
                </td>
              </tr>
            </table>

            <h2 style="margin:0 0 12px;color:#111827;font-size:16px;">Detected Violations</h2>
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
              <tr style="background:#f9fafb;">
                <th style="padding:10px 16px;text-align:left;color:#6b7280;font-size:12px;text-transform:uppercase;">
                  Violation
                </th>
                <th style="padding:10px 16px;text-align:center;color:#6b7280;font-size:12px;text-transform:uppercase;">
                  Confidence
                </th>
                <th style="padding:10px 16px;text-align:center;color:#6b7280;font-size:12px;text-transform:uppercase;">
                  Status
                </th>
              </tr>
              {violation_rows}
            </table>
            {image_block}
          </td>
        </tr>
        <tr>
          <td style="background:#f9fafb;padding:16px 32px;border-top:1px solid #e5e7eb;">
            <p style="margin:0;color:#9ca3af;font-size:12px;text-align:center;">
              AI Safety Detection System — Automated Alert
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    def build_plain_text(
        self,
        camera: str,
        location: str,
        alerts: list[dict],
        image_url: Optional[str] = None,
    ) -> str:
        alert_time = self._format_time(alerts[0].get("time") if alerts else None)
        lines = [
            "PPE SAFETY VIOLATION DETECTED",
            "=" * 40,
            f"Location : {location}",
            f"Camera   : {camera}",
            f"Time     : {alert_time}",
            f"Count    : {len(alerts)} violation(s)",
            "",
            "Violations:",
        ]
        for i, alert in enumerate(alerts, start=1):
            lines.append(
                f"  {i}. {alert.get('event', 'Violation')} "
                f"({alert.get('confidence', 0):.1f}% confidence) — {alert.get('status', 'New')}"
            )
        if image_url:
            lines.extend(["", f"Evidence: {image_url}"])
        return "\n".join(lines)

    def send_detection_alert(
        self,
        camera: str,
        location: str,
        alerts: list[dict],
        image_url: Optional[str] = None,
    ) -> bool:
        if not self.enabled or not alerts:
            return False

        subject = self.build_subject(camera, location, alerts)
        html_body = self.build_html(camera, location, alerts, image_url)
        plain_body = self.build_plain_text(camera, location, alerts, image_url)

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.smtp_from
        message["To"] = ", ".join(self.alert_to)
        message.attach(MIMEText(plain_body, "plain", "utf-8"))
        message.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_from, self.alert_to, message.as_string())
            logger.info("Email alert sent to %s", ", ".join(self.alert_to))
            return True
        except smtplib.SMTPAuthenticationError as exc:
            logger.error(
                "Gmail login failed. Use your full email as SMTP_USER and a 16-char "
                "App Password (not your normal Gmail password). "
                "Create one at: https://myaccount.google.com/apppasswords — %s",
                exc,
            )
            return False
        except Exception as exc:
            logger.error("Failed to send email alert: %s", exc)
            return False

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
