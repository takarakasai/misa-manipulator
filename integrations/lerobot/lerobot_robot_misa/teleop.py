from __future__ import annotations

import logging
from functools import cached_property

from lerobot.teleoperators.teleoperator import Teleoperator

from .client import RunnerError, shared_client
from .config import MisaLeaderConfig

logger = logging.getLogger(__name__)


class MisaLeader(Teleoperator):
    """The runner's leader: connecting switches the runner to teleop, and the
    action is the leader's pose mapped onto the arm (the runner's
    ``state.leader``)."""

    config_class = MisaLeaderConfig
    name = "misa_leader"

    def __init__(self, config: MisaLeaderConfig):
        super().__init__(config)
        self.config = config
        self.client = shared_client(config.url, config.token)
        self._connected = False

    @cached_property
    def action_features(self) -> dict:
        return dict.fromkeys(self.client.feature_names(), float)

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        info = self.client.load_info()
        if not info.get("teleop"):
            raise RunnerError("the runner has no leader: start it with --leader <profile>")
        self.client.post("mode", {"mode": "teleop", "control": self.config.control})
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict:
        leader = self.client.get("state").get("leader")
        if leader is None:
            raise RunnerError("no fresh leader sample from the runner")
        return self.client.to_features(leader)

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        try:
            self.client.post("mode", {"mode": "hold"})
        except Exception as e:  # the runner may already be gone
            logger.warning("%s: could not switch the runner to hold: %s", self, e)
        self.client.close()
        self._connected = False
