# models

| ディレクトリ | 機体 | 由来 |
|---|---|---|
| `rebot_b601_dm/` | Seeed reBot Arm B601-DM（DAMIAO） | reBot-DevArm `Rebot_Arm_description/DM` |
| `rebot_b601_rs/` | Seeed reBot Arm B601-RS（RobStride） | reBot-DevArm `Rebot_Arm_description/RS` |

上流は [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm)
（コミット `73af17a`、ライセンスは CERN-OHL-W-2.0 → [`LICENSE-reBot-DevArm`](LICENSE-reBot-DevArm)）。
`scripts/import-models.sh` で作り直せる。**手で編集しないこと**（作り直すと消える）。
機体ごとの値（ポーズ・ゲイン・モータ割り当て）は `robots/*.toml` に書く。

## 取り込みで変えたもの

質量・慣性・関節の幾何は一切変えていない。変えたのは次だけ。

- **メッシュを間引いた**（表示 3000 面 / 衝突 800 面、116 MB → 8 MB）。
- **DM のフィンガー可動幅は 0.05 m（DevArm の値）。** reBotArm_control_py の同じ
  URDF は 0.0285 m だが、実機（2026-10-03）は指の間隔が 0 → 約 100 mm 開く。
- **RS の `gripper_joint2` を `gripper_joint1` の mimic（1:1）にした。**
  上流の URDF は 2 本の指を独立な prismatic（0–0.05 m と 0–0.0715 m）で
  書いているが、実機のグリッパは RS-00 が 1 個でラック&ピニオンを回す。
  可動幅は 0–0.05 m に揃えた。**実測は未。**

## 未確認のこと

- **モータ角 → 指の変位の換算**は DM で実測済み（0.00844 m/rad、`robots/rebot_b601_dm.toml`）。
  BOM のピニオン（ピッチ半径 8 mm）と測定の範囲で一致。RS は未測定。
- DM の指リンクは質量 0（上流のまま）。
- RS の `effort` はグリッパが 500 N（上流のまま、明らかに仮の値）。
