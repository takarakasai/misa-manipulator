# misa-manipulator

[misarta](https://github.com/takarakasai/misarta)（剛体力学）と
[misa-wbc](https://github.com/takarakasai/misa-wbc)（階層 QP）を使った
**モデルベースのマニピュレータ制御**。ロボット 1 台は「モデル（`.misa`）+
プロファイル（TOML）」で定義し、フォロワーもリーダーもファイルの差し替えで
組み替えられる。

| 役割 | 初期ターゲット | 通信 |
|---|---|---|
| フォロワー | Seeed reBot Arm **B601-DM**（DAMIAO DM4340P ×3 + DM4310 ×4） | CAN 1 Mbps（SocketCAN） |
| フォロワー | Seeed reBot Arm **B601-RS**（RobStride RS-06 ×3 + RS-00 ×4） | CAN 1 Mbps（SocketCAN） |
| リーダー | Star Arm 102 LD（= reBot Arm 102、FashionStar UART サーボ ×7） | UART 1 Mbps（CH340） |

**実機では未検証**（2026-09-27 時点。この PC にロボットが繋がっていない）。
シム（MuJoCo / 剛体）と単体テストで確かめた範囲は
[`doc/handover.md`](doc/handover.md) の表を見ること。

## できること

| モード | 中身 |
|---|---|
| `hold` | その場で保持（保持ゲイン + 重力フィードフォワード） |
| `gravity` | 重力補償だけ（手で動かせる） |
| `joint` | 関節空間の追従。モータの MIT の PD + モデルのフィードフォワード（重力 or 逆動力学） |
| `osc` | TCP の追従（操作空間制御）。misa-wbc の階層 QP: トルク上限・可動域 CBF > TCP 加速度 > 姿勢 |
| Park | 終了時（`--duration` 経過・Ctrl-C）に休止姿勢へ畳んでから脱力 |

目標の出どころ（`--source`）: リーダーアーム / 合成の正弦波 / TCP の円 / なし。

## 構成

```
crates/
├── manip-model/         misarta を腕向けに包む（独立自由度・TCP・M/h/g/J/J̇v）
├── manip-control/       制御則（I/O なし）: 重力補償・関節インピーダンス・OSC・参照整形
├── manip-leader/        リーダー（Star Arm 102）と合成の目標源。別スレッドで読む
├── manip-plant-can/     実機: DAMIAO / RobStride の MIT を CAN で（バスごとに自由走行スレッド）
├── manip-plant-mujoco/  シム: articara の MuJoCo（ワークスペース外、`--features sim`）
├── manip-runner/        アプリ `manip`（プロファイル → 組み立て → 状態機械 → ループ）
└── manip-tools/         manip-import（URDF → .misa、メッシュ間引き）/ manip-inspect
models/    取り込んだモデル（手で編集しない。scripts/import-models.sh で作り直す）
robots/    フォロワーのプロファイル（ゲイン・可動域・モータ割り当て・テレオペの写像）
leaders/   リーダーのプロファイル（サーボ ID・符号・範囲）
```

Plant（実機・MuJoCo・剛体）は misa-runner の `misa-core` の `Plant` をそのまま使う。
**制御則はモデルの関節座標しか知らない。** モータの符号・ゼロ点・減速比
（グリッパはラック&ピニオンの m/rad）は `manip-plant-can` が変換する。

## ビルド

```sh
cargo build --release                 # 実機 + 剛体シム（MuJoCo 不要）
cargo test --release --workspace

# MuJoCo シム（MuJoCo 3.8 が要る）
export MUJOCO_DYNAMIC_LINK_DIR=$HOME/.mujoco/mujoco-3.8.0/lib
export LD_LIBRARY_PATH=$MUJOCO_DYNAMIC_LINK_DIR
cargo build --release -p manip-runner --features sim
```

依存は GitHub の git 依存（`git clone && cargo build` だけで立つ）。兄弟
チェックアウト（misa-actuator など）を併行して直すときは
`./scripts/dev-siblings.sh`（misa-runner と同じ作法）。

## 使い方

```sh
# シム: 合成の目標を関節インピーダンスで追う（--fast は実時間を待たない）
manip run --robot robots/rebot_b601_dm.toml --plant sim --source sine --mode joint --duration 20
# シム: TCP で円を描く（OSC）。畳んだ姿勢は可動域の端なので ready へ運んでから
manip run --robot robots/rebot_b601_dm.toml --plant sim --source circle --mode osc --start-pose ready
# シムは既定で実機の「悪さ」（遅延・ジッタ・量子化・摩擦）を掛ける。
# --ideal で外す、--delay-ticks / --jitter で上書き。掃引は scripts/sweep_effects.py
# 実機用 Plant（バススレッド・座標変換）を仮想の腕で回す（実時間）
manip run --robot robots/rebot_b601_dm.toml --plant virtual-can --source sine --start-pose ready --duration 10
# MuJoCo が無い環境: --plant rigid（接触なしの剛体積分、摩擦と悪さは同じく掛かる）
manip run --robot robots/rebot_b601_rs.toml --plant rigid --source sine --duration 10 --fast

# 実機の立ち上げ（scan / monitor / sign は通電しない）。--plant virtual-can でリハーサル可
manip hw --robot robots/rebot_b601_dm.toml scan
manip hw --robot robots/rebot_b601_dm.toml sign
manip hw --robot robots/rebot_b601_dm.toml jog --joint joint2 --delta 5
# 摩擦の同定（シムでも実行可: --plant sim / rigid）。--write でプロファイルへ
manip hw --robot robots/rebot_b601_dm.toml friction

# リーダーの値を見る / ゼロ姿勢のオフセットを読む（サーボには書かない）
manip leader --leader leaders/stararm102.toml
manip leader --leader leaders/stararm102.toml --zero

# テレオペ（シムのフォロワー）→ 実機のフォロワー
manip run --robot robots/rebot_b601_dm.toml --plant sim --source leader --leader leaders/stararm102.toml
manip run --robot robots/rebot_b601_dm.toml --plant can --source leader --leader leaders/stararm102.toml

# 記録（CSV: 関節ごとの q, v, qref, vref, τ と TCP）
manip run ... --record logs/run.csv

# 実行ログ（バイナリ）と再生。記録した観測・モード要求・目標をいまのコードに
# 通し直し、指令が 1 bit でも変われば報告して非 0 で終わる（改修の回帰確認用）
manip run ... --log logs/run.mrec
manip replay logs/run.mrec                 # 記録時のプロファイルで
manip replay logs/run.mrec --robot <別のプロファイル>
```

Ctrl-C 1 回目で休止姿勢へ畳んでから脱力、2 回目で即脱力（**腕は落ちる**）。

## ロボットを足すには

1. モデル: `manip-import <urdf> models/<name> --name <name>`（必要なら `--mimic`）。
   `manip-inspect models/<name>/<name>.misa --tcp <link>` で自由度・可動域・
   重力トルクの最大値を確かめる。
2. フォロワー: `robots/<name>.toml` を既存のものから写し、`[[joint]]`（モデルの
   独立自由度をすべて）・`[[hardware.bus]]`・`[[teleop]]` を書く。
3. リーダー: 出力は**中立空間**（LeRobot の action と同じ約束の関節名と角度）。
   フォロワー側の `[[teleop]]` が中立空間 → 自分の関節（`q = offset + scale·x`）を持つので、
   リーダーとフォロワーは独立に差し替えられる。
