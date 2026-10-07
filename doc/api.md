# 運動指令 API（HTTP/JSON）

`manip run --source api` で制御ループと一緒に HTTP サーバが立ち上がり、
姿勢への移動・経由点・速度・加速度・力・インピーダンス・グリッパ・モード切替を
JSON で受け付ける。`manip cmd` はその CLI クライアント。

```bash
# サーバ（実機）。--leader を付けると teleop モードに切り替えられる
manip --rt-priority 80 run --robot robots/rebot_b601_dm.toml --plant can --source api

# クライアント（別端末）
manip cmd move/pose name=ready --wait
manip cmd move/tcp position=0,0,-0.05 relative=true --wait
manip cmd gripper width=0.06
manip cmd state
```

- **最初に `move/pose name=ready`。** 畳んだ姿勢（肩・肘が可動域の上限）から TCP 指令
  （OSC）に入ると QP が退化する（handover §4-4）。
- シムで試すには `--plant rigid`（または `sim`）。手順を JSON にまとめて
  `--source script --script file.json --fast` で流すと、同じ指令を時刻どおりに再生できる
  （§スクリプト）。
- 起動直後は保持（Hold）。最初の指令でそれに合ったモードへ移る。

## 接続と認証

| 項目 | 既定 | |
|---|---|---|
| 待ち受け | `--api-bind 127.0.0.1:8080` | ループバック以外（`0.0.0.0:8080` など）は**トークン必須**（無いと起動しない） |
| トークン | `--api-token-file <file>` か環境変数 `MANIP_API_TOKEN` | リクエストに `Authorization: Bearer <token>` |
| クライアント | `manip cmd --url http://<host>:8080 --token <t>` | `MANIP_API_TOKEN` も読む |

```bash
curl -s -X POST http://127.0.0.1:8080/v1/move/tcp \
     -H 'Content-Type: application/json' \
     -d '{"position": [0, 0, 0.05], "relative": true}'
```

## 単位と座標系

- 長さ m、角度 rad（`"deg": true` で度。関節は回転関節だけ変換）、力 N、モーメント N·m。
- 関節は**独立自由度**の順（`GET /v1/info` の `joints`）。B601 は `joint1..joint6` と
  `finger_left`（グリッパ、片側の指の変位 [m]、もう片方は鏡映）。
- 関節値は配列（全関節、またはグリッパを除いた 6 個）か `{"joint2": -1.2}`
  （書いていない関節は今の参照のまま）。
- TCP の `frame`: `base`（既定、台座の座標）か `tool`（TCP 自身。移動は指令を受けた
  時点の TCP、速度・力は動く TCP に付いていく）。`tool` は `relative: true` と組で使う。
- 姿勢は `quat: [x, y, z, w]`、`rpy: [roll, pitch, yaw]`（`R = Rz(yaw)·Ry(pitch)·Rx(roll)`）、
  `axis_angle: [x, y, z]` のどれか 1 つ。省略すると今の姿勢を保つ。
- 6 成分の並びは `[回転 x, y, z, 並進 x, y, z]`（状態の `twist`、`wrench_estimate` も同じ）。

## 指令の流れと状態

POST した指令にはすべて `id` が付き、状態は
`queued → active → done / aborted / rejected`（`message` に理由や到達誤差）。

- **運動**（move / waypoints / velocity / accel）は 1 つずつ実行する。既定では新しい運動が
  **今の運動を置き換える**（`aborted: replaced by #n`）。`"queue": true` なら後ろに並ぶ。
- 運動は**前の運動が残した参照から始まる**（途中で置き換えても飛ばない）。
- `?wait=true`（`manip cmd --wait`）で終わるまで待つ。`&timeout=秒`（既定 60）。
  `manip cmd --wait` は `aborted` / `rejected` なら終了コード 1。
- 力・インピーダンス・関節トルク・グリッパは**設定**で、運動と並行して効き続ける（解除するまで）。
- 制御器が Hold に落ちた（OSC が解けない、MPC が止まった）ら実行中の運動は `aborted` に
  なり、次の指令まで何も再要求しない。

## エンドポイント

すべて `/v1/` の下。GET は `state`、`info`、`motions/<id>`。他は POST（本文 JSON）。

### 移動

| パス | 本文 | 内容 |
|---|---|---|
| `move/joint` | `q`、`relative`、`deg`、`duration`、`speed` | 関節空間で移動（5 次多項式、静止で終わる） |
| `move/pose` | `name`、`duration`、`speed` | プロファイルの `[pose.*]`（`ready`、`rest`）へ。グリッパはそのまま |
| `move/tcp` | `position`、`quat`/`rpy`/`axis_angle`、`frame`、`relative`、`deg`、`duration`、`speed`、`via` | TCP を直線（姿勢は回転軸まわり）で移動。`via: "mpc"` なら MPC が経路を決める（到達 2 mm / 0.02 rad で done、15 s で aborted） |
| `waypoints/joint` | `points`（関節値の配列）、`times`、`relative`、`deg`、`speed` | 経由点を止まらずに通る（向きが変わる軸だけ一旦止まる） |
| `waypoints/tcp` | `points`（`{position, rpy…}` の配列）、`times`、`frame`、`relative`、`deg`、`speed` | 同、TCP。`relative` は 1 つ前の点から |

- `speed`: 制限（関節: プロファイルの `v_max` / `a_max`、TCP: `[osc] lin_v_max` など）に
  対する割合、既定 0.5。
- `duration` / `times`（開始からの秒、点ごと）が制限を越えるなら**延ばす**
  （`message: slowed down to keep the limits`）。
- 可動域の外の関節値は可動域に収める（`message: clamped`）。TCP の目標は IK で検査しない
  （届かなければ届くところで止まり、done の `message` に残りの誤差が出る）。

### 速度・加速度（ストリーミング）

| パス | 本文 | 内容 |
|---|---|---|
| `velocity/joint` | `v`、`deg`、`timeout` | 関節速度 |
| `velocity/tcp` | `linear`、`angular`、`frame`、`deg`、`timeout` | TCP の速度 |
| `accel/joint` | `a`、`deg`、`timeout` | 関節加速度（速度は制限で頭打ち） |
| `accel/tcp` | `linear`、`angular`、`frame`、`deg`、`timeout` | TCP の加速度 |

- **`timeout`（既定 0.2 s）以内に次の指令が来なければ減速して止まる**（done:
  `no command within the timeout`）。送り続けるなら 50 Hz 程度で。
- 同じ種類の指令が続けば同じ運動が続く（前の id は `done: continued by #n`）。
- 関節は可動域の手前（0.02 rad）で止まる。TCP は参照が実測から 2 cm / 0.1 rad 以上先へ
  行かない（壁や可動域で腕が止まっても参照だけが進まない）。作業領域の箱と自己干渉は
  制御器が守る。

### 力・インピーダンス・関節トルク（設定）

| パス | 本文 | 内容 |
|---|---|---|
| `force` | `force`、`moment`、`frame`、`free`、`ramp`、`max_speed`、`max_angular_speed`／`clear` | TCP が周りに及ぼす力・モーメント。OSC へ切り替わる |
| `impedance` | `stiffness: {linear, angular}`、`damping: {linear, angular}`、`frame`／`clear` | TCP のばね [N/m, N·m/rad]（各 3 成分か 1 つの数）。`damping` 省略で臨界減衰。OSC へ切り替わる |
| `joint/torque` | `torque`（関節値）、`stiffness_scale`、`ramp`／`clear` | 関節トルクを足す [N·m]。`stiffness_scale` は追従の硬さの倍率（0 でトルクのみ）。Joint へ切り替わる |

- **力センサは無い。** 力はモデルから開ループで出す（`τ = Jᵀ·w`）。押し付けた力の誤差は
  関節の静止摩擦ぶん（B601-DM の肩・肘で約 1.75 N·m、TCP で数 N）。`state` の
  `wrench_estimate`（報告トルク − 重力）も静止時の目安でしかない。
- `force` の `free`（既定: 0 でない成分の軸）は位置を保たない軸。空中では
  `max_speed`（既定 0.05 m/s）を越えると力を弱める。力を解除すると**その場で止まる**
  （参照が free 軸に沿って腕に付いていく）。`ramp`（既定 0.3 s）で立ち上げ・解除。
- 作業領域の箱の壁は力でも越えない（空中で押すと壁の手前で止まる）。その代わり壁の
  数 cm 以内では力が弱まるので、**押し付ける面は箱の壁より数 cm 内側**に置く。
- `impedance` は TCP 運動（移動・速度・保持）すべてに効く。B601 の手首は軽く、硬い回転
  ばねは 500 Hz では保てないので、上限（`[osc] compliance_kp_max` など）を越える分は
  回転から先に弱める（実機未確認の値）。
- `joint/torque` は可動域の 0.1 rad 手前で、その向きのトルクを止める。

### グリッパとモード

| パス | 本文 | 内容 |
|---|---|---|
| `gripper` | `position` / `width` / `open` / `close` のどれか、`speed`、`max_force` | 指の変位 [m]（`width` は指の間隔 = 2 × position）。`max_force` [N] で握る力を抑える。物に当たって止まっても done（`message: blocked … short`） |
| `mode` | `mode`: `hold` / `gravity` / `park` / `teleop`、`control`: `joint` / `osc` / `mpc` | `gravity` は手で動かせる。`park` は畳んで（通電したまま）待つ。`teleop` はリーダー（`--leader` で起動したとき） |
| `stop` | — | 今の運動を減速して止め、待ちの指令を捨て、力・関節トルクを解除する |
| `shutdown` | — | 畳んで脱力し、プロセスを終える（Ctrl-C と同じ） |

### 状態

`GET /v1/state`:

```json
{
  "t": 12.3, "mode": "osc",
  "q": [...], "v": [...], "tau_measured": [...],
  "tcp": {"position": [x, y, z], "quat": [x, y, z, w], "rpy": [r, p, y], "twist": [6]},
  "gripper": {"position": 0.03, "width": 0.06},
  "wrench_estimate": [6],
  "motion": {"active": {"id": 7, "kind": "tcp_trajectory"}, "queued": 0, "teleop": false,
             "force": null, "impedance": null, "joint_torque": null, "gripper_goal": 0.03}
}
```

`GET /v1/info` は関節名・可動域・速度上限・トルク上限・名前付き姿勢・グリッパ・
teleop / MPC の有無。`GET /v1/motions/<id>` は指令 1 つの状態。

## `manip cmd`

```
manip cmd [--url URL] [--token T] [--wait] [--timeout S] <path> [key=value ...]
```

- 値は JSON として読む（`3`、`true`、`[1,2]`、`{"joint2":-1.2}`）。`1,2,3` は数の配列、
  それ以外は文字列。`a.b=1` で入れ子（`stiffness.linear=300`）。
- 例:

```bash
manip cmd move/joint 'q={"joint1":30}' deg=true relative=true --wait
manip cmd move/tcp position=0.03,0,0 rpy=0,0,15 deg=true frame=tool relative=true
manip cmd waypoints/tcp 'points=[{"position":[0,0.05,0]},{"position":[0,0,0.05]}]' relative=true
manip cmd velocity/tcp linear=0,0.05,0 timeout=0.5
manip cmd force force=0,0,-5 max_speed=0.03
manip cmd impedance stiffness.linear=300 stiffness.angular=5
manip cmd gripper close=true max_force=20 --wait
manip cmd mode mode=teleop control=osc
manip cmd stop
```

## スクリプト

`--source script --script file.json`: API と同じ本文を時刻どおりに渡す。終わって何も
動いていなければ畳んで終了。`--fast`（シム）で実時間より速く回せる。

```json
[
  {"t": 0.0, "path": "move/pose", "body": {"name": "ready"}},
  {"t": 0.0, "path": "move/tcp", "body": {"position": [0, 0, -0.05], "relative": true, "queue": true}},
  {"t": 0.5, "path": "gripper", "body": {"width": 0.06}},
  {"t": 9.0, "path": "force", "body": {"force": [0, 0, -3]}},
  {"t": 12.0, "path": "force", "body": {"clear": true}}
]
```

## 記録と再生

API で受けた指令は、モード要求と目標（生成した参照）として**記録される入力**になる
（`--log`、log format 3）。ネットワークから何がいつ届いたかに関係なく、
`manip replay` で指令がビット単位で再現する。
