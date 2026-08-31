"""
Discover UniFi Protect cameras on each NVR and enable RTSP.

Run this ON THE SITE PC (the AnyDesk machine that can reach 10.10.30.x):

    python unifi_discover.py

It logs into each NVR, lists cameras, enables RTSP if needed, and writes
unifi_cameras.json for nvr_config.py to load.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(__file__).resolve().parent / "unifi_cameras.json"
SSL_CTX = ssl._create_unverified_context()

USERNAME = os.environ.get("NVR_USERNAME", "admin")
PASSWORD = os.environ.get("NVR_PASSWORD", "")
# 0 = high, 1 = medium, 2 = low  (use low for 57 cameras)
RTSP_QUALITY = int(os.environ.get("UNIFI_RTSP_QUALITY", "2"))

NVR_HOSTS = [
    {
        "id": "nvr1",
        "name": os.environ.get("NVR1_NAME", "UniFi NVR 1"),
        "ip": os.environ.get("NVR1_IP", "10.10.30.2"),
        "expected_count": int(os.environ.get("NVR1_CAMERA_COUNT", "19")),
    },
    {
        "id": "nvr2",
        "name": os.environ.get("NVR2_NAME", "UniFi NVR 2"),
        "ip": os.environ.get("NVR2_IP", "10.10.30.3"),
        "expected_count": int(os.environ.get("NVR2_CAMERA_COUNT", "20")),
    },
    {
        "id": "nvr3",
        "name": os.environ.get("NVR3_NAME", "UniFi NVR 3"),
        "ip": os.environ.get("NVR3_IP", "10.10.30.4"),
        "expected_count": int(os.environ.get("NVR3_CAMERA_COUNT", "18")),
    },
]


def _cookie_header(cookies):
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def _parse_set_cookie(headers):
    cookies = {}
    csrf = None
    raw = headers.get_all("Set-Cookie") if hasattr(headers, "get_all") else []
    if not raw:
        single = headers.get("Set-Cookie")
        raw = [single] if single else []

    for item in raw:
        if not item:
            continue
        first = item.split(";", 1)[0]
        if "=" not in first:
            continue
        name, value = first.split("=", 1)
        cookies[name.strip()] = value.strip()
        if name.strip().upper() == "CSRF_TOKEN":
            csrf = value.strip()
    return cookies, csrf


def _request(url, method="GET", body=None, cookies=None, csrf=None, timeout=20):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if cookies:
        headers["Cookie"] = _cookie_header(cookies)
    if csrf:
        headers["X-CSRF-Token"] = csrf

    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            raw = resp.read()
            new_cookies, new_csrf = _parse_set_cookie(resp.headers)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            return resp.status, payload, new_cookies, new_csrf or csrf, None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            payload = {"error": raw.decode("utf-8", errors="replace")}
        return exc.code, payload, {}, csrf, str(exc)
    except Exception as exc:
        return None, {}, {}, csrf, str(exc)


def login(ip, username, password):
    """Login to UniFi OS / Protect. Returns (cookies, csrf) or raises."""
    url = f"https://{ip}/api/auth/login"
    status, payload, cookies, csrf, err = _request(
        url,
        method="POST",
        body={"username": username, "password": password, "rememberMe": True},
        timeout=15,
    )
    if status != 200 or not cookies.get("TOKEN"):
        raise RuntimeError(
            f"Login failed on {ip} (HTTP {status}): {err or payload}"
        )

    csrf = (
        csrf
        or payload.get("csrfToken")
        or payload.get("csrf_token")
        or cookies.get("TOKEN")
    )
    return cookies, csrf


def fetch_cameras(ip, cookies, csrf):
    """Return Protect camera list from bootstrap, with cameras endpoint fallback."""
    status, payload, _, csrf, err = _request(
        f"https://{ip}/proxy/protect/api/bootstrap",
        cookies=cookies,
        csrf=csrf,
        timeout=30,
    )
    if status == 200 and isinstance(payload, dict) and "cameras" in payload:
        return payload["cameras"], csrf

    status, payload, _, csrf, err = _request(
        f"https://{ip}/proxy/protect/api/cameras",
        cookies=cookies,
        csrf=csrf,
        timeout=30,
    )
    if status == 200 and isinstance(payload, list):
        return payload, csrf

    raise RuntimeError(f"Could not list cameras on {ip} (HTTP {status}): {err or payload}")


def _pick_channel(camera, quality):
    channels = camera.get("channels") or []
    if not channels:
        return None

    for ch in channels:
        if ch.get("id") == quality:
            return ch

    named = {0: "High", 1: "Medium", 2: "Low"}
    want = named.get(quality, "Low").lower()
    for ch in channels:
        if str(ch.get("name", "")).lower() == want:
            return ch

    # Prefer any enabled RTSP channel, then last (usually low)
    enabled = [ch for ch in channels if ch.get("isRtspEnabled") and ch.get("rtspAlias")]
    if enabled:
        return enabled[-1]
    return channels[min(quality, len(channels) - 1)]


def enable_rtsp(ip, camera, cookies, csrf, quality):
    """Enable RTSP on the chosen channel if Protect has not issued a token yet."""
    camera_id = camera.get("id")
    channel = _pick_channel(camera, quality)
    if channel is None:
        return camera, csrf

    if channel.get("isRtspEnabled") and channel.get("rtspAlias"):
        return camera, csrf

    channel_id = channel.get("id", quality)
    body = {
        "channels": [
            {
                "id": channel_id,
                "isRtspEnabled": True,
            }
        ]
    }
    status, payload, _, csrf, err = _request(
        f"https://{ip}/proxy/protect/api/cameras/{camera_id}",
        method="PATCH",
        body=body,
        cookies=cookies,
        csrf=csrf,
        timeout=20,
    )
    if status == 200 and isinstance(payload, dict):
        return payload, csrf

    print(f"    Could not enable RTSP for {camera.get('name')} (HTTP {status}): {err}")
    return camera, csrf


def camera_record(nvr, camera, quality):
    channel = _pick_channel(camera, quality) or {}
    token = (channel.get("rtspAlias") or "").strip()
    host = camera.get("host") or camera.get("connectionHost") or ""
    name = camera.get("name") or camera.get("displayName") or "Camera"

    return {
        "name": name,
        "location": f"{nvr['name']} — {name}",
        "rtsp_token": token,
        "camera_ip": host,
        "protect_id": camera.get("id"),
        "online": bool(camera.get("state") == "CONNECTED" or camera.get("isConnected")),
        "rtsp_enabled": bool(channel.get("isRtspEnabled")),
    }


def discover_nvr(nvr, username, password, quality, enable=True):
    print(f"Connecting to {nvr['name']} ({nvr['ip']})...")
    cookies, csrf = login(nvr["ip"], username, password)
    cameras, csrf = fetch_cameras(nvr["ip"], cookies, csrf)
    print(f"  Found {len(cameras)} camera(s) in Protect (expected {nvr['expected_count']})")

    records = []
    for cam in cameras:
        if enable:
            cam, csrf = enable_rtsp(nvr["ip"], cam, cookies, csrf, quality)
        rec = camera_record(nvr, cam, quality)
        token_state = rec["rtsp_token"] or "(no RTSP token yet)"
        print(f"    {rec['name']}: {token_state}")
        records.append(rec)

    return {
        "id": nvr["id"],
        "name": nvr["name"],
        "ip": nvr["ip"],
        "brand": "unifi",
        "rtsp_port": 7447,
        "cameras": records,
    }


def load_cache(path=CACHE_PATH):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("nvrs"):
        return None
    return data


def save_cache(nvrs, path=CACHE_PATH):
    payload = {
        "username": USERNAME,
        "rtsp_quality": RTSP_QUALITY,
        "nvrs": nvrs,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def discover_all(enable_rtsp_streams=True):
    if not PASSWORD:
        raise RuntimeError("NVR_PASSWORD is not set in .env")

    nvrs = []
    errors = []
    for nvr in NVR_HOSTS:
        try:
            nvrs.append(
                discover_nvr(
                    nvr,
                    USERNAME,
                    PASSWORD,
                    RTSP_QUALITY,
                    enable=enable_rtsp_streams,
                )
            )
        except Exception as exc:
            print(f"  FAILED {nvr['ip']}: {exc}")
            errors.append({"nvr": nvr["id"], "ip": nvr["ip"], "error": str(exc)})
            nvrs.append(
                {
                    "id": nvr["id"],
                    "name": nvr["name"],
                    "ip": nvr["ip"],
                    "brand": "unifi",
                    "rtsp_port": 7447,
                    "cameras": [],
                    "error": str(exc),
                }
            )

    path = save_cache(nvrs)
    total = sum(len(nvr.get("cameras") or []) for nvr in nvrs)
    print(f"\nSaved {total} camera(s) to {path}")
    if errors:
        print(f"{len(errors)} NVR(s) failed — run this script on the site PC that can reach 10.10.30.x")
    return nvrs, errors


def main():
    try:
        nvrs, errors = discover_all(enable_rtsp_streams=True)
    except Exception as exc:
        print(f"Discovery failed: {exc}", file=sys.stderr)
        return 1

    missing_tokens = [
        (nvr["id"], cam["name"])
        for nvr in nvrs
        for cam in nvr.get("cameras") or []
        if not cam.get("rtsp_token")
    ]
    if missing_tokens:
        print("\nCameras without an RTSP token:")
        for nvr_id, name in missing_tokens:
            print(f"  {nvr_id}: {name}")
        print("Enable RTSP in Protect (Device → Settings → Advanced → RTSP) and re-run.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
