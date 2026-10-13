"""A small HTTP/JSON client for the runner's API (keep-alive, no dependencies)."""

from __future__ import annotations

import http.client
import json
import math
import threading
from urllib.parse import urlparse


class RunnerError(RuntimeError):
    pass


_SHARED: dict[tuple, "RunnerClient"] = {}
_SHARED_LOCK = threading.Lock()


def shared_client(url: str, token: str | None) -> "RunnerClient":
    """One client per runner, shared by the robot and the teleoperator (one
    cached `info`, requests serialized)."""
    with _SHARED_LOCK:
        c = _SHARED.get((url, token))
        if c is None:
            c = _SHARED[(url, token)] = RunnerClient(url, token)
        return c


class RunnerClient:
    """`GET /v1/info|state`, `POST /v1/<path>`."""

    def __init__(self, url: str = "http://127.0.0.1:8080", token: str | None = None, timeout_s: float = 1.0):
        u = urlparse(url)
        if u.scheme != "http" or not u.hostname:
            raise ValueError(f"expected http://host:port, got {url!r}")
        self.host, self.port = u.hostname, u.port or 80
        self.token = token
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self.info: dict | None = None

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        # A new connection per request ("Connection: close"): on a kept-alive
        # one, the runner's server (tiny_http, no TCP_NODELAY) writes the
        # response in two segments and the second waited for the client's
        # delayed ACK, ~40 ms per request (0.2 ms this way, on localhost).
        headers = {"Content-Type": "application/json", "Connection": "close"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None if body is None else json.dumps(body)
        with self._lock:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
            try:
                conn.request(method, path, body=data, headers=headers)
                resp = conn.getresponse()
                text = resp.read().decode()
            finally:
                conn.close()
        try:
            out = json.loads(text) if text else {}
        except json.JSONDecodeError as e:
            raise RunnerError(f"{method} {path}: not JSON: {text[:200]}") from e
        if resp.status >= 400:
            raise RunnerError(f"{method} {path}: HTTP {resp.status}: {out}")
        return out

    def get(self, what: str) -> dict:
        return self._request("GET", f"/v1/{what}")

    def post(self, path: str, body: dict) -> dict:
        return self._request("POST", f"/v1/{path}", body)

    def close(self) -> None:
        pass

    # ── features ─────────────────────────────────────────────────────────

    def load_info(self) -> dict:
        self.info = self.get("info")
        return self.info

    def joints(self) -> list[dict]:
        if self.info is None:
            self.load_info()
        return self.info["joints"]

    def feature_names(self) -> list[str]:
        """`<joint>.pos` for revolute joints, `gripper.pos` for the gripper."""
        g = (self.info or self.load_info()).get("gripper")
        return ["gripper.pos" if j["name"] == g else f"{j['name']}.pos" for j in self.joints()]

    def to_features(self, q: list[float]) -> dict[str, float]:
        """Runner units (rad, m) -> features (deg, mm)."""
        out = {}
        for name, j, x in zip(self.feature_names(), self.joints(), q):
            out[name] = x * 1000.0 if j["kind"] == "prismatic" else math.degrees(x)
        return out

    def from_features(self, action: dict[str, float]) -> dict[str, float]:
        """Features (deg, mm) -> `{joint: value}` in runner units (rad, m)."""
        out = {}
        for name, j in zip(self.feature_names(), self.joints()):
            if name in action:
                x = float(action[name])
                out[j["name"]] = x / 1000.0 if j["kind"] == "prismatic" else math.radians(x)
        return out
