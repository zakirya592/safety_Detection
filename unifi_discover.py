"""
Discover UniFi Protect cameras on each NVR.

Run this ON THE SITE PC:

    python unifi_discover.py

Login works even when RTSP cannot be enabled (HTTP 403). In that case
detection uses Protect snapshot URLs instead of RTSP tokens.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(__file__).resolve().parent / "unifi_cameras.json"
SSL_CTX = ssl._create_unverified_context()

USERNAME = os.environ.get("NVR_USERNAME", "admin")
PASSWORD = os.environ.get("NVR_PASSWORD", "")
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

_sessions = {}
_sessions_lock = threading.Lock()


def _cookie_header(cookies):
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def _header_csrf(headers):
    if headers is None:
        return None
    return (
        headers.get("X-CSRF-Token")
        or headers.get("x-csrf-token")
        or headers.get("X-Csrf-Token")
    )


def _parse_set_cookie(headers):
    cookies = {}
    csrf = _header_csrf(headers)
    raw = headers.get_all("Set-Cookie") if headers and hasattr(headers, "get_all") else []
    if not raw:
        single = headers.get("Set-Cookie") if headers else None
        raw = [single] if single else []

    for item in raw:
        if not item:
            continue
        first = item.split(";", 1)[0]
        if "=" not in first:
            continue
        name, value = first.split("=", 1)
        cookies[name.strip()] = value.strip()
        if name.strip().upper() in {"CSRF_TOKEN", "X-CSRF-TOKEN"}:
            csrf = csrf or value.strip()
    return cookies, csrf


def _request(url, method="GET", body=None, cookies=None, csrf=None, timeout=20, json_body=True):
    headers = {
        "Accept": "application/json, image/jpeg, */*",
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    if cookies:
        headers["Cookie"] = _cookie_header(cookies)
        token = cookies.get("TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    if csrf:
        headers["X-CSRF-Token"] = csrf

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8") if json_body else body

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            raw = resp.read()
            new_cookies, new_csrf = _parse_set_cookie(resp.headers)
            if cookies is not None and new_cookies:
                cookies.update(new_cookies)
            csrf = new_csrf or csrf
            content_type = (resp.headers.get("Content-Type") or "").lower()
            payload = {}
            if raw and "json" in content_type:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {}
            elif raw and "json" not in content_type and method != "GET":
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = {}
            return resp.status, payload, cookies or new_cookies, csrf, None, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        err_csrf = _header_csrf(exc.headers)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"error": raw.decode("utf-8", errors="replace") if raw else ""}
        return exc.code, payload, cookies or {}, err_csrf or csrf, str(exc), raw
    except Exception as exc:
        return None, {}, cookies or {}, csrf, str(exc), b""


def login(ip, username, password):
    url = f"https://{ip}/api/auth/login"
    status, payload, cookies, csrf, err, _ = _request(
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
        or cookies.get("csrf_token")
        or cookies.get("TOKEN")
    )
    return cookies, csrf


def get_session(ip):
    with _sessions_lock:
        session = _sessions.get(ip)
        if session:
            return session
        cookies, csrf = login(ip, USERNAME, PASSWORD)
        session = {"cookies": cookies, "csrf": csrf}
        _sessions[ip] = session
        return session


def refresh_session(ip):
    with _sessions_lock:
        cookies, csrf = login(ip, USERNAME, PASSWORD)
        _sessions[ip] = {"cookies": cookies, "csrf": csrf}
        return _sessions[ip]


def fetch_snapshot_jpeg(ip, camera_id):
    """JPEG bytes from Protect. Re-logins once if the session expired."""
    if not camera_id:
        return None
    url = f"https://{ip}/proxy/protect/api/cameras/{camera_id}/snapshot?force=true"
    for attempt in range(2):
        session = get_session(ip)
        status, _, _, csrf, err, raw = _request(
            url,
            cookies=session["cookies"],
            csrf=session["csrf"],
            timeout=10,
            json_body=False,
        )
        session["csrf"] = csrf or session["csrf"]
        if status == 200 and raw and raw[:2] == b"\xff\xd8":
            return raw
        if status in (401, 403) and attempt == 0:
            refresh_session(ip)
            continue
        if attempt == 0:
            time.sleep(0.2)
    return None


def fetch_bootstrap(ip, cookies, csrf):
    status, payload, cookies, csrf, err, _ = _request(
        f"https://{ip}/proxy/protect/api/bootstrap",
        cookies=cookies,
        csrf=csrf,
        timeout=30,
    )
    if status == 200 and isinstance(payload, dict) and "cameras" in payload:
        return payload, cookies, csrf

    status, payload, cookies, csrf, err, _ = _request(
        f"https://{ip}/proxy/protect/api/cameras",
        cookies=cookies,
        csrf=csrf,
        timeout=30,
    )
    if status == 200 and isinstance(payload, list):
        return {"cameras": payload, "nvr": {}}, cookies, csrf

    raise RuntimeError(f"Could not list cameras on {ip} (HTTP {status}): {err or payload}")


def _norm_mac(value):
    return (value or "").replace(":", "").replace("-", "").lower()


def filter_cameras_for_nvr(cameras, nvr_ip, nvr_mac=None):
    """Keep cameras adopted to this NVR, not the full Protect site list."""
    this_mac = _norm_mac(nvr_mac)
    matched = []
    adopted = []
    connected = []

    for cam in cameras:
        state = (cam.get("state") or "").upper()
        is_connected = state == "CONNECTED" or bool(cam.get("isConnected"))
        is_adopted = cam.get("isAdopted") is not False
        if cam.get("isAdoptedByOther"):
            continue
        if is_connected:
            connected.append(cam)
        if is_adopted:
            adopted.append(cam)

        cam_nvr = _norm_mac(cam.get("nvrMac"))
        conn = str(cam.get("connectionHost") or cam.get("host") or "")
        if this_mac and cam_nvr and cam_nvr == this_mac:
            matched.append(cam)
        elif nvr_ip and nvr_ip in conn:
            matched.append(cam)

    if matched:
        return matched, "adopted to this NVR"
    if connected:
        return connected, "connected cameras (could not match NVR MAC)"
    if adopted:
        return adopted, "adopted cameras"
    return cameras, "all Protect cameras (no filter matched)"


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

    enabled = [ch for ch in channels if ch.get("isRtspEnabled") and ch.get("rtspAlias")]
    if enabled:
        return enabled[-1]
    return channels[min(quality, len(channels) - 1)]


def enable_rtsp(ip, camera, cookies, csrf, quality):
    """Try to enable RTSP. Returns (camera, csrf, ok)."""
    camera_id = camera.get("id")
    channel = _pick_channel(camera, quality)
    if channel is None:
        return camera, csrf, False

    if channel.get("isRtspEnabled") and channel.get("rtspAlias"):
        return camera, csrf, True

    channels_body = []
    for ch in camera.get("channels") or [channel]:
        item = {"id": ch.get("id"), "isRtspEnabled": True}
        channels_body.append(item)

    body = {"channels": channels_body or [{"id": channel.get("id", quality), "isRtspEnabled": True}]}

    for method in ("PATCH", "PUT"):
        status, payload, cookies, csrf, err, _ = _request(
            f"https://{ip}/proxy/protect/api/cameras/{camera_id}",
            method=method,
            body=body,
            cookies=cookies,
            csrf=csrf,
            timeout=20,
        )
        if status == 200 and isinstance(payload, dict):
            return payload, csrf, True
        if status == 403:
            return camera, csrf, False

    print(f"    Could not enable RTSP for {camera.get('name')} (HTTP {status}): {err}")
    return camera, csrf, False


def camera_record(nvr, camera, quality):
    channel = _pick_channel(camera, quality) or {}
    token = (channel.get("rtspAlias") or "").strip()
    host = camera.get("host") or camera.get("connectionHost") or ""
    name = camera.get("name") or camera.get("displayName") or "Camera"
    protect_id = camera.get("id")
    online = bool((camera.get("state") or "").upper() == "CONNECTED" or camera.get("isConnected"))

    return {
        "name": name,
        "location": f"{nvr['name']} — {name}",
        "rtsp_token": token,
        "camera_ip": host,
        "protect_id": protect_id,
        "mac": camera.get("mac"),
        "owner_nvr_mac": camera.get("nvrMac"),
        "online": online,
        "rtsp_enabled": bool(channel.get("isRtspEnabled")),
        "stream": "rtsp" if token else "snapshot",
    }


def discover_nvr(nvr, username, password, quality, enable=True):
    print(f"Connecting to {nvr['name']} ({nvr['ip']})...")
    cookies, csrf = login(nvr["ip"], username, password)
    with _sessions_lock:
        _sessions[nvr["ip"]] = {"cookies": cookies, "csrf": csrf}

    bootstrap, cookies, csrf = fetch_bootstrap(nvr["ip"], cookies, csrf)
    all_cameras = bootstrap.get("cameras") or []
    nvr_info = bootstrap.get("nvr") or {}
    nvr_mac = nvr_info.get("mac")
    cameras, filter_reason = filter_cameras_for_nvr(all_cameras, nvr["ip"], nvr_mac)

    print(
        f"  Protect listed {len(all_cameras)} camera(s); "
        f"using {len(cameras)} ({filter_reason}). Expected about {nvr['expected_count']}."
    )

    records = []
    rtsp_blocked = False
    for cam in cameras:
        if enable and not rtsp_blocked:
            cam, csrf, ok = enable_rtsp(nvr["ip"], cam, cookies, csrf, quality)
            channel = _pick_channel(cam, quality) or {}
            if not ok and not channel.get("rtspAlias"):
                print(
                    "  RTSP cannot be enabled with this account (HTTP 403). "
                    "Live detection will use Protect snapshots instead."
                )
                rtsp_blocked = True
        rec = camera_record(nvr, cam, quality)
        mode = rec["rtsp_token"] or rec["stream"]
        state = "online" if rec["online"] else "offline"
        print(f"    {rec['name']} [{state}]: {mode}")
        records.append(rec)

    return {
        "id": nvr["id"],
        "name": nvr["name"],
        "ip": nvr["ip"],
        "brand": "unifi",
        "rtsp_port": 7447,
        "nvr_mac": nvr_mac,
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


def _assign_cameras_to_owning_nvr(nvrs):
    """If every Protect console lists the whole site, keep each camera on one NVR."""
    unique = {}
    for nvr in nvrs:
        for cam in nvr.get("cameras") or []:
            key = (cam.get("mac") or "").lower() or cam.get("protect_id") or f"{nvr['id']}:{cam.get('name')}"
            if key not in unique:
                unique[key] = {**cam, "_source_nvr": nvr["id"]}

    buckets = {nvr["id"]: [] for nvr in nvrs}
    nvr_macs = {nvr["id"]: _norm_mac(nvr.get("nvr_mac")) for nvr in nvrs}

    for cam in unique.values():
        owner = _norm_mac(cam.get("owner_nvr_mac"))
        placed = False
        if owner:
            for nvr_id, mac in nvr_macs.items():
                if mac and mac == owner:
                    buckets[nvr_id].append(cam)
                    placed = True
                    break
        if not placed:
            buckets[cam["_source_nvr"]].append(cam)

    for nvr in nvrs:
        cleaned = []
        for cam in buckets[nvr["id"]]:
            cam.pop("_source_nvr", None)
            cleaned.append(cam)
        nvr["cameras"] = cleaned
        print(f"  {nvr['id']} after split: {len(cleaned)} camera(s)")
    return nvrs


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

    nvrs = _assign_cameras_to_owning_nvr(nvrs)
    path = save_cache(nvrs)
    total = sum(len(nvr.get("cameras") or []) for nvr in nvrs)
    rtsp_ok = sum(
        1
        for nvr in nvrs
        for cam in nvr.get("cameras") or []
        if cam.get("rtsp_token")
    )
    print(f"\nSaved {total} camera(s) to {path}")
    print(f"  RTSP tokens: {rtsp_ok}")
    print(f"  Snapshot fallback: {total - rtsp_ok}")
    if errors:
        print(f"{len(errors)} NVR(s) failed — run this script on the site PC that can reach 10.10.30.x")
    return nvrs, errors


def main():
    try:
        nvrs, errors = discover_all(enable_rtsp_streams=True)
    except Exception as exc:
        print(f"Discovery failed: {exc}", file=sys.stderr)
        return 1

    print("\nWhat this means:")
    print("  Login to Protect succeeded.")
    print("  HTTP 403 on RTSP means this user can view cameras but cannot turn RTSP on.")
    print("  Detection can still run using Protect snapshots (no RTSP token needed).")
    print("\nNext:")
    print("  python live_detection_api.py")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
