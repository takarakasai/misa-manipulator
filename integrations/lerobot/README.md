# lerobot_robot_misa

LeRobot から B601 を **b601-runner 経由で**動かすためのロボット／テレオペレータ。

LeRobot 標準の `rebot_b601_follower` はモータを直接動かし、runner の安全装置
（力の上限、速度超過のブレーキ、作業空間の箱、自己干渉、重力補償、遠い目標への
ゆっくりした寄り）がどれも効かない。こちらは `manip run --source api` の HTTP API を使う。

| 種類 | `type` | 中身 |
|---|---|---|
| ロボット | `misa_b601` | 観測 = runner の実測関節（`GET /v1/state`）。行動 = 関節目標を `POST /v1/stream/joint` へ（runner が teleop 中は送らずにそのまま返す） |
| テレオペレータ | `misa_leader` | 接続で runner を teleop に切り替え（runner がリーダーを読んで安全装置付きで腕を動かす）、行動 = リーダーを腕に写した値（`state.leader`） |

特徴量: 回転関節は `joint1.pos` … `joint6.pos` [deg]、グリッパは `gripper.pos`
[mm、片側の指の変位、0 = 閉]。

## 入れ方

```bash
source ~/work/dp/lerobot-env.sh
uv pip install -e ~/work/dp/misa-manipulator/integrations/lerobot
```

LeRobot は `lerobot_robot_*` という名前の配布物を自動で import する。

## 録画（リーダーで操作）

```bash
# 端末 1: runner（実機）。リーダーも runner が読む
manip --rt-priority 80 run --robot robots/rebot_b601_dm.toml --plant can --source api --leader leaders/stararm102.toml

# 端末 2: LeRobot
lerobot-record \
  --robot.type=misa_b601 --robot.id=b601dm $REBOT_CAMERAS \
  --teleop.type=misa_leader --teleop.id=stararm102 \
  --dataset.repo_id=<user>/<name> --dataset.single_task="..." --dataset.fps=30 ...
```

`--teleop.control=mpc` で runner の teleop を Mpc にできる（既定 `joint`）。
LeRobot が終わるとテレオペレータが runner を Hold に戻す。

## 学習したポリシーの実行

runner は `--leader` なしでよい。`lerobot-record --policy.path=...`（または rollout）を
`--robot.type=misa_b601` で。行動は 30 Hz の関節目標として runner に流れ、
`stream_timeout_s`（既定 0.2 s）以上途切れると runner が腕を止める。

## メモ

- 1 要求ごとに接続し直す（`Connection: close`）。runner の HTTP サーバ（tiny_http）は
  TCP_NODELAY が無く、keep-alive だと遅延 ACK で 1 要求 ~40 ms 待った（今は ~0.3 ms）。
- 較正は runner のプロファイル（`[hardware]` の `zero` / `sign` / `ratio`）が持つ。
  LeRobot の calibrate は何もしない。
- シムで確かめるには runner を `--plant rigid` で（リーダーは読むだけなので本物でよい）。
