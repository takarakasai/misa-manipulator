from __future__ import annotations

import logging
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .client import shared_client
from .config import MisaB601Config

logger = logging.getLogger(__name__)


class MisaB601(Robot):
    """Observation from, and joint targets to, misa-manipulator's runner.

    While the runner teleoperates (``misa_leader``), ``send_action`` does not
    stream: the runner already drives the arm toward the leader, and the
    action is returned for recording as it is.
    """

    config_class = MisaB601Config
    name = "misa_b601"

    def __init__(self, config: MisaB601Config):
        super().__init__(config)
        self.config = config
        self.client = shared_client(config.url, config.token)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False

    # ── features ────────────────────────────────────────────────────────

    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(self.client.feature_names(), float)

    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft(), **self._cameras_ft()}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft()

    # ── lifecycle ───────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        info = self.client.load_info()
        logger.info("%s: runner %s, joints %s", self, info.get("robot"), self.client.feature_names())
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True  # the runner owns calibration (its profile's [hardware])

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self.client.close()
        self._connected = False

    # ── I/O ─────────────────────────────────────────────────────────────

    def get_observation(self) -> dict:
        state = self.client.get("state")
        obs = self.client.to_features(state["q"])
        for key, cam in self.cameras.items():
            obs[key] = cam.read_latest()
        return obs

    def send_action(self, action: dict) -> dict:
        motion = self.client.get("state").get("motion") or {}
        if motion.get("teleop"):
            return action
        q = self.client.from_features(action)
        self.client.post("stream/joint", {"q": q, "timeout": self.config.stream_timeout_s})
        return action
