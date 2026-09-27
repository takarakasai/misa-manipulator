# handover — misa-manipulator

2026-09-27 に PC（ロボット未接続）で作った内容の引き継ぎ。**コードが語れない
こと**（何を確かめたか・なぜそうしたか・次に何をするか）をここに置く。

---

## 1. 確かめた範囲 / 確かめていない範囲

| 項目 | 状態 | 根拠 |
|---|---|---|
| モデル（独立自由度・mimic・TCP の J / J̇v） | ✅ | manip-model のテスト（数値微分と照合、重力 = ∇U） |
| misarta と MuJoCo の重力項の一致 | ✅ | `compare_gravity`: TCP の鎖で最大差 3.5e-6 N·m（DM / RS） |
| 重力補償の保持（MuJoCo） | ✅ | kp = 0 で ready から 2 s、最大ずれ 0.07 rad（起動の過渡、その後静止） |
| 関節インピーダンスの追従（MuJoCo, DM, 正弦波） | ✅ | 肩 rms 8.4 mrad（重力 FF）→ 3.0 mrad（逆動力学 FF） |
| OSC の TCP 追従（MuJoCo, 半径 5 cm の円） | ✅ | DM 1.3 mm rms / RS 1.5 mm rms、1 周期の計算 平均 0.14 ms |
| OSC の可動域 CBF・特異点 | ✅ | 届かない目標で手首特異姿勢に入っても止まる（剛体テスト） |
| DM と RS の符号の整合 | ✅ | 同じリーダー入力で両機の TCP が 22–29 mm 以内（テスト、符号 1 本の反転で落ちる） |
| Park → 脱力、Ctrl-C | ✅ シム | 剛体・MuJoCo とも Done まで |
| FashionStar ドライバ | ⚠️ 単体テストのみ | misa-actuator `144d190`（56 テスト、SDK の pack 出力と照合） |
| **CAN の実機（DAMIAO / RobStride）** | ❌ **未** | DM4340P / RS-06 / RS-00 は misa-actuator でも実機未確認 |
| **Star Arm 102 の実機** | ❌ **未** | 符号（`sign`）とグリッパの倍率は上流の値のまま |
| 実時間性（500 Hz） | ⚠️ | 計算は中央値 50–140 µs。ただしこの PC は他の負荷（load 60）で数 ms の外れ値が出た |

---

## 2. 設計の判断

### 出力は MIT（硬さはモータ、モデルはフィードフォワード）

DAMIAO も RobStride も MIT（`τ = kp(q*−q) + kd(v*−v) + τff`）をネイティブに
持ち、モータの中で kHz で回る。こちらの周期（500 Hz）と CAN の往復遅れで同じ
硬さをトルクで閉じると位相が回る。**位置の硬さはモータ、モデル（重力・慣性）は
`τff`** という分担にした。OSC だけは硬さもトルクで作る（`kp = 0`）。

### 独立自由度で話す

B601 のグリッパは URDF では 2 本の prismatic（片方 mimic）だがモータは 1 個。
manip-model が `G`（mimic 射影）で独立座標に落とし、以降はすべて 7 自由度。
**RS の URDF は 2 本の指が独立**（0–0.05 / 0–0.0715 m）だったので、取り込みで
mimic（1:1）を足した（`--mimic gripper_joint2=gripper_joint1`）。

### OSC は TCP の鎖の上だけを解く

グリッパの反映慣性（ラック&ピニオンで数十 kg 相当）と手首（0.003 kg·m²）を
同じ `M` に入れると条件数が 10⁴ を超え、Clarabel が level 0 で
NumericalFailure を返した。TCP を動かさない自由度は QP から外し、関節
インピーダンスで追う（`ArmModel::tcp_chain`）。

### QP は ActiveSet + AccelSpace

`osc_bench`（剛体の円）: Clarabel 中央値 0.46 ms、ActiveSet 0.05–0.09 ms、
追従は同じ。固定ベースで接触が無いので AccelSpace（τ を消去、変数 6 個）。
プロファイルの `[osc] backend / formulation` で戻せる。

### 特異点の減衰は「止める」向き

misa-wbc の `cartesian_acceleration_damped` は `‖q̈‖²` を罰するので、**動いて
いる軸を止めない**（等速で流れ続ける）。手首の特異姿勢（J5 = ±90°）で J4 と J6 が
逆向きに回り続けた（TCP は動かないので level 1 は気づかない）。
`λ²‖q̈ + K·v‖²` を σ_min(J) に応じて level 1 に混ぜている。

### 脱力しない

どの失敗も保持に落とす（OSC が解けない → Hold、リーダー断 → 整形器が減速して
止まる、制御ループ停止 → バススレッドが硬さを落として重力 FF だけ残す）。
脱力するのは Park で畳み終えたときと、2 回目の Ctrl-C だけ。**腕は落ちる。**

### 起動時の可動域チェックで通電を拒む

実測の姿勢が可動域から範囲の 10 % 以上外れていたら通電しない。ゼロ点か符号の
取り違えがいちばんありそうで、そのまま通電すると整形器が「可動域の中」へ運ぼう
とする。

---

## 3. 実機の「悪さ」を入れたシムでの評価（2026-09-27）

シムは既定で `[sim.effects]` を掛ける（`--ideal` で外す）。中身は
`crates/manip-runner/src/effects.rs`:

- **指令の遅延**（既定 1 tick = 2 ms。7 台を 1 Mbps の CAN で巡回するバス
  スレッドの 1 周の目安、**推定**）と**ジッタ**（既定 10 % で 1 tick 追加）
- **フィードバックの量子化**: `[hardware]` の型番からプロトコル表どおりに
  （DAMIAO の速度 12 bit = DM4310 で 0.015 rad/s）
- **クーロン摩擦**（`[[joint]] sim_friction`、肩肘 0.3 N·m・手首 0.08 N·m、**推定**）
  は物理の刻みごとに MuJoCo / 剛体 Plant の中で掛ける

`scripts/sweep_effects.py` の結果（B601-DM、MuJoCo。RS もほぼ同じ）:

| 条件 | 関節追従（正弦波）rms | OSC（半径 5 cm の円）TCP rms |
|---|---|---|
| 摩擦 0・理想 | 7.1 mrad | 1.31 mm |
| 摩擦あり・理想 | 11.0 mrad | 5.01 mm |
| 摩擦あり + 遅延 1 tick + ジッタ + 量子化（既定） | 12.3 mrad | 5.01 mm |
| 同・遅延 4 tick（8 ms） | 16.2 mrad | 5.03 mm |

- **精度を落としているのは遅延ではなく摩擦。** OSC は積分を持たないので、
  補償されないクーロン摩擦がそのまま TCP の誤差になる（1.3 → 5.0 mm）。
  関節追従の誤差は遅延 1 tick あたり約 1.25 mrad 増えるが、発振はしない。
- **遅延への余裕（OSC）**: 既定のゲインは 8 ms でも安定。ゲイン ×8
  （kp_lin 3200）でも 8 ms で安定（誤差 0.73 mm）。**×16 は 4 ms まで安定、
  8 ms で発散**（86 mm）。実機のバス周期を測ってから上げること。
- **量子化**は指令トルクの周期ごとの変化（チャタリング）をゲインに比例して
  増やす（既定ゲインで 0.002 → 0.06 N·m/tick、×16 で 0.75）。主因は 12 bit の
  速度に kd が掛かる分。

→ 次の手: 摩擦の補償（同定したクーロン摩擦のフィードフォワード、または OSC の
姿勢・TCP に積分を足す）。値は実機で misa-sysid により同定してから入れる。

---

## 4. 踏んだ罠

1. **質量 0 のリンクに MuJoCo が質量を作る。** MJCF に `<inertial>` が無いと
   ジオメトリの体積 × 水の密度になる。B601-DM の指（上流の URDF で質量 0）が
   シムでだけ 0.18 kg 増え、肩の重力トルクが 15 % ずれた。manip-import が
   動くリンクの質量 0 を 1 g に置き換える。
2. **ベンダの衝突メッシュは畳んだ姿勢・閉じたグリッパでめり込んでいる。**
   閉じた指が接触力で 0.03 m 開き、保持の PD が数千 N で押し合った。シムの
   自己干渉は既定で切ってある（`[sim] self_collision`）。
3. **通電前の読み出しの間にシムを進めると自由落下する。** 実機はストッパに
   乗っているがシムは宙に浮いている。MuJoCo Plant は `arm()` まで時間を進めない。
4. **畳んだ姿勢（ゼロ）は肩・肘がちょうど可動域の上限。** OSC をそこから始めると
   CBF が退化して QP が解けない。`--start-pose ready` で離してから入る。
5. **misa-core の SafetyGate がトルク・位置の変化率制限を丸め誤差で誤報告していた。**
   `from + (want − from)` が浮動小数点で `want` に戻らず、τ ≈ 0 の軸で「制限した」と
   出た（値は不変）。misa-runner `0706607` で修正済み。
6. **URDF の速度上限は実機より 1 桁大きい**（50 / 200 rad/s）。OSC の速度 CBF が
   効かないので、プロファイルの `v_max` で必ず上書きする（`load_arm` が当てる）。

---

## 5. 実機の立ち上げ手順（案、未実施）

**電源はすぐ切れるようにしておく。** DAMIAO は disable 後も次のフレームで再通電
しうる（misa-actuator の handover §2）。

1. CAN: `sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up`。
   `damiao-cli scan` / `robstride-cli scan` で ID 1–7 が見えること（DM の Master ID は
   ID + 0x10、RS の host は 0xFD）。
2. ゼロ点: Seeed の手順（LeRobot `lerobot-calibrate`、畳んだ姿勢・グリッパ閉）で
   済ませてあれば、`[hardware]` の `zero = 0`。**`manip` はゼロを書かない。**
3. 読むだけ: 腕を手で支えて `manip run --plant can --mode gravity --duration 5`。
   起動時に初期姿勢が表示される。畳んだ姿勢で全軸 ≈ 0 か、可動域チェックで
   止められないかを見る。
4. 重力補償: そのまま手を離して落ちないこと（落ちるなら `gravity_scale`・
   `armature`・モデルの質量を疑う。符号が逆なら加速して落ちる）。
5. 保持 → 関節の追従: `--mode hold`、次に `--source sine --mode joint`
   （`[[sine]]` の振幅を小さくしてから）。
6. リーダー: `manip leader --leader leaders/stararm102.toml` で 1 軸ずつ動かし、
   中立空間の符号を確かめる（**upstream の符号が正しいか未確認**）。
   シムのフォロワーで `--source leader` → 実機。
7. グリッパの `ratio`（m/rad）を実測で直す（開閉の端で指の変位を測る）。

---

## 6. 未確定・次にやること

- **グリッパの換算**（DM 0.00605 / RS 0.0106 m/rad）は可動幅 ÷ LeRobot の
  モータ可動域から出した仮の値。BOM のピニオン（8 mm）とは合わない。
- **反映ロータ慣性（armature）と速度上限**は目安。misa-sysid で同定する。
- **Star Arm 102 の URDF**（LD のもの）が無いので、リーダーの FK による
  TCP 対 TCP のテレオペ（形の違うリーダー → フォロワー）は未実装。いまは
  関節対関節で、`--mode osc` のときはフォロワーの FK で TCP 目標を作る。
- **可視化**: articara の Zenoh フィードは四脚 12 関節専用。腕用のフレーム型が要る。
- 実時間: 実機では `chrt -f 80` と CPU の分離を検討（この PC は他のシムで
  load 60 のとき数 ms の外れ値が出た）。
- DAMIAO 専用の USB-CAN ブリッジ（HDSC 2e88:4603、B601-DM 同梱）は使わない
  （SocketCAN の決定、2026-09-27）。
