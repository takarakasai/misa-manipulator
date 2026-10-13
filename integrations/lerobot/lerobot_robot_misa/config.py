from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@RobotConfig.register_subclass("misa_b601")
@dataclass
class MisaB601Config(RobotConfig):
    """A B601 driven by `manip run --source api` (see the package docstring)."""

    # The runner's API (`--api-bind`); a token is required off loopback.
    url: str = "http://127.0.0.1:8080"
    token: str | None = None
    # The runner stops a joint target stream this long after the last target [s].
    stream_timeout_s: float = 0.2
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@TeleoperatorConfig.register_subclass("misa_leader")
@dataclass
class MisaLeaderConfig(TeleoperatorConfig):
    """The runner's leader (`manip run --source api --leader ...`): the runner
    teleoperates the arm itself; this reads what it drives toward."""

    url: str = "http://127.0.0.1:8080"
    token: str | None = None
    # Teleop control in the runner: "joint", "osc" or "mpc".
    control: str = "joint"
