#!/usr/bin/env bash
# Regenerate models/ from the upstream URDF.
#
# Upstream: Seeed-Projects/reBot-DevArm (CERN-OHL-W-2.0). The import is pinned
# to a commit, so models/ does not change on its own when upstream edits the URDF.
# To update, bump REV, check the differences (mass, joint limits, gravity torque)
# with manip-inspect, and only then commit.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=https://github.com/Seeed-Projects/reBot-DevArm.git
REV=73af17ae54f8a2209dd856a0fea0611cf07d2107
SRC=ref/reBot-DevArm

if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none "$REPO" "$SRC"
fi
git -C "$SRC" fetch -q origin "$REV" 2>/dev/null || true
git -C "$SRC" checkout -q "$REV"

cargo build --release -q -p manip-tools
D="$SRC/Rebot_Arm_description"

# DM: finger travel is set to the reBotArm_control_py value (0.0285 m).
# DevArm's 0.05 m is from an older revision. Which one matches the real hardware
# is unconfirmed (see README).
rm -rf models/rebot_b601_dm
./target/release/manip-import "$D/DM/urdf/ReBot_Arm_DM.urdf" models/rebot_b601_dm \
  --name rebot_b601_dm --visual-tris 3000 --finger-travel 0.0285

# RS: the URDF models the two fingers as independent prismatic joints, but the real gripper has one motor.
rm -rf models/rebot_b601_rs
./target/release/manip-import "$D/RS/urdf/ReBot_Arm_RS.urdf" models/rebot_b601_rs \
  --name rebot_b601_rs --visual-tris 3000 --mimic gripper_joint2=gripper_joint1

cp "$SRC/LICENSE" models/LICENSE-reBot-DevArm
