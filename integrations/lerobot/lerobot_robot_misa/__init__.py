"""LeRobot devices that go through misa-manipulator's runner instead of the motors.

LeRobot's own ``rebot_b601_follower`` drives the motors directly and has none of
the runner's guards (force cap, overspeed brake, workspace box, self-collision,
gravity feedforward, catching up with a far target). These devices talk to
``manip run --source api [--leader ...]`` over HTTP instead:

- ``misa_b601`` (robot): observation = the runner's measured joints; action =
  joint targets streamed to ``POST /v1/stream/joint`` (ignored while the runner
  itself teleoperates, see below).
- ``misa_leader`` (teleoperator): puts the runner in teleop (it reads the leader
  and drives the arm with all guards) and returns the leader's pose mapped onto
  the arm, i.e. the action to record.

Features: ``<joint>.pos`` in degrees for revolute joints and ``gripper.pos`` in
millimetres of finger travel (one finger, 0 = closed).

Importing this package (LeRobot does it for installed ``lerobot_robot_*``
distributions) registers both.
"""

from .client import RunnerClient
from .config import MisaB601Config, MisaLeaderConfig
from .robot import MisaB601
from .teleop import MisaLeader

__all__ = ["RunnerClient", "MisaB601Config", "MisaLeaderConfig", "MisaB601", "MisaLeader"]
