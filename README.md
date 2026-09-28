# Excavator Digging RL — Isaac Lab

Train the **MathScavator9000** excavator (từ `autonomous-excavator-main`) thực hiện
**digging task** trong Isaac Sim, dùng PPO với hàng nghìn env song song.

Thuật toán / cấu trúc task port từ
[hanzunye/vortexRL](https://github.com/hanzunye/vortexRL) (Han & Stein, KIT —
*Applying Reinforcement Learning to Digital Twin of Excavator to Dig Automatically*),
vốn chạy trên Vortex Studio với 1 environment.

---

## 1. Model của bạn: những gì đã kiểm tra và đã sửa

Đọc trực tiếp từ `MathScavator9000_flat.SLDASM_physics.usd` + các file STL:

| Hạng mục | Giá trị trong asset | Xử lý |
|---|---|---|
| Kinematic chain | 4 revolute: `base_chassis_joint` (swing, Z), `chassis_boom_joint`, `boom_stick_joint`, `stick_bucket_joint` | giữ nguyên; digging dùng 3 joint sau (như vortexRL) |
| Joint limits | boom −32°…+71°, stick −46.9°…+62°, bucket −130°…0°, swing ±180° | giữ nguyên, khai báo lại trong `excavator_cfg.py` |
| **Joint drives** | position drive, **stiffness 5.4e7 – 2.9e8**, damping 2.2e4 – 1.1e5, **maxForce = 3.4e38 (vô hạn)**, **maxJointVelocity vô hạn** | **thay bằng velocity drive**: stiffness = 0, effort/velocity limit thật |
| Effort/velocity trong URDF | `effort="0" velocity="0"` (exporter không điền) | điền theo lực đào máy 36 t |
| Mesh URI | `package://MathScavator9000_flat.SLDASM/meshes/...` | `convert_urdf.py` đổi sang absolute path |
| Khối lượng | base 139.4 t (root cố định), chassis 17.4 t, boom 7.2 t, stick 3.3 t, bucket 5.1 t → **33.1 t phần chuyển động** | giữ nguyên — đúng class 36 t |
| Collision | convexHull cho mọi link | giữ nguyên (nhanh, và bucket không thể "hốt" particle bằng hình học ⇒ dùng soil model giải tích) |
| Ground | `base_link` mesh chạm đáy ở z = −1.295 | spawn root ở z = +1.295 ⇒ hết lún đất |

**Đây chính là lý do `chassis_boom_joint` trước đây "không phản hồi"**: drive stiffness
2.6e8 với lực không giới hạn là một servo vị trí cứng tuyệt đối — mọi lệnh
velocity/torque đều bị nó ghì lại.

Hình học đo từ STL (dùng cho soil model và FK):

```
boom 6.24 m · stick 2.98 m · bucket pivot→cutting edge 1.835 m
bucket width 1.148 m · tầm với tip tối đa 10.79 m
tip offset (bucket frame) = (0, −1.835, −0.04)
```

Tầm với 10.8 m + khối lượng 33 t ⇒ máy này **cùng phân khúc 36 t với model Vortex
Studio trong vortexRL**, nên toàn bộ task/reward port sang gần như 1-1.

Tự kiểm tra lại bất kỳ USD nào (không cần Isaac Sim):

```bash
pip install usd-core
python tools/inspect_usd.py /đường/dẫn/..._physics.usd
```

---

## 2. Cài trên máy Ubuntu

Yêu cầu: Ubuntu 20.04/22.04/24.04, NVIDIA driver ≥ 535, GPU RTX ≥ 8 GB VRAM.

```bash
# 1) Isaac Sim + Isaac Lab (pip route, đơn giản nhất)
conda create -n isaaclab python=3.11 -y && conda activate isaaclab
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
./isaaclab.sh --install          # cài luôn rsl_rl, skrl, rl_games
cd ..

# 2) package này
unzip excavator_isaaclab.zip && cd excavator_isaaclab
pip install -e .                 # optional, chỉ để import từ nơi khác

# test chạy được không cần Isaac Sim (chỉ cần torch)
python tests/test_soil.py        # soil model: lực, bóc đất, spill, bảo toàn thể tích
python tests/test_reward.py      # reward + phase machine (Isaac Lab được stub)
```

> Package viết cho **Isaac Lab 2.x** (Isaac Sim 4.5 / 5.0). Các chỗ API đổi giữa
> 2.0 → 2.2 đều đã có fallback (`quat_apply`, `is_global` của external wrench,
> `UrdfConverterCfg`, `body_link_*_w` vs `body_com_*_w`).

### Copy asset sang Ubuntu

Cần 2 thứ từ máy Windows:

```
MathScavator9000_flat/urdf/MathScavator9000_flat.SLDASM.urdf
MathScavator9000_flat/meshes/*.STL
```

(hoặc copy nguyên thư mục `MathScavator9000_flat.SLDASM/` đã convert sẵn — nhưng
nên convert lại để lấy drive/limit đã sửa.)

---

## 3. Chạy

```bash
cd ~/IsaacLab                    # dùng isaaclab.sh để có đúng python env

# a) URDF -> USD, kèm sửa mesh path + effort/velocity limit + velocity drive
./isaaclab.sh -p ~/excavator_isaaclab/scripts/convert_urdf.py \
    --input  ~/assets/MathScavator9000_flat/urdf/MathScavator9000_flat.SLDASM.urdf \
    --output ~/excavator_assets/excavator.usd

export EXCAVATOR_USD=~/excavator_assets/excavator.usd

# b) kiểm tra trước khi train: joint/drive/mass thật + 1 chu kỳ đào scripted
./isaaclab.sh -p ~/excavator_isaaclab/scripts/check_model.py

# c) train
./isaaclab.sh -p ~/excavator_isaaclab/scripts/train.py --num_envs 4096 --headless

# d) xem policy chạy
./isaaclab.sh -p ~/excavator_isaaclab/scripts/play.py \
    --checkpoint ~/IsaacLab/logs/excavator/model_final.pt --num_envs 4
```

**Bước (b) là bước quan trọng nhất.** Nó in ra depth / lực cản đất / độ đầy gàu
theo thời gian. Nếu cả ba đều bằng 0 thì tip offset hoặc bán kính soil bed sai —
sửa trước, đừng train.

Theo dõi: `tensorboard --logdir ~/IsaacLab/logs/excavator`

---

## 4. Thiết kế task

| | vortexRL (Vortex Studio) | Bản port này (Isaac Lab) |
|---|---|---|
| Action | 3 × vận tốc xy-lanh, low-pass | 3 × vận tốc khớp (boom/stick/bucket), low-pass `α=0.25` |
| State | boom/stick/bucket position + cờ `back` | 16-dim: q, q̇, action, tip xyz, depth, fill ratio, cờ phase, sai số độ cao |
| Đất | particle/mesh hybrid solver | soil model giải tích (mục 5) |
| Reward | phân đoạn (đào → nâng), toàn giá trị âm | `dense` (mặc định) hoặc `vortex` (port nguyên bản) |
| Kết thúc | gàu đủ đất & cao > 4.2 m | như trên (`target_fill=0.6`, `lift_height=4.2 m`) |
| Thuật toán | REINFORCE / DDPG / PPO / TRPO, 1 env | PPO (rsl_rl), 4096 env |

`reward_mode="vortex"` port đúng `Reward/RewardDDPG.py`. Lưu ý: bản gốc dùng chuỗi
`if` (không phải `elif`) nên nhánh `M < M_old` bị nhánh `else` ghi đè — ở đây
implement theo **ý định** của tác giả, có ghi chú trong code.

Dùng `--reward_mode vortex` nếu bạn muốn so sánh trực tiếp với kết quả trong paper;
dùng mặc định `dense` nếu muốn PPO hội tụ nhanh với nhiều env song song.

---

## 5. Soil model

Isaac Sim không có đất biến dạng chạy nổi ở quy mô nghìn env, nên `excavator_rl/soil.py`
dùng mô hình giải tích — cách hầu hết paper excavation-RL quy mô lớn vẫn làm:

1. **Địa hình** — height profile 1-D `h(r)` theo bán kính từ trục swing (cánh tay
   chỉ đào trong mặt phẳng đó khi swing bị khóa).
2. **Lực cản** — Fundamental Earthmoving Equation (Reece):
   `F = w·(ρ·g·d²·N_γ + c·d·N_c + q·d·N_q) + viscous`, ngược chiều vận tốc tip,
   đặt tại lưỡi cắt ⇒ sinh đúng moment phản lực lên stick và boom.
3. **Bóc đất** — swept-min carving: ô đất lưỡi cắt quét qua bị hạ xuống bằng cao độ
   lưỡi, thể tích chênh lệch vào gàu (× hiệu suất, chặn ở dung tích gàu).
   Chính xác về hình học và ổn định vô điều kiện.
4. **Rơi vãi** — gàu nghiêng quá `spill_angle` thì đất chảy ra theo hằng số thời gian,
   trả lại nền và **trượt theo góc nghỉ** (không tạo cột đất kim).
5. **Tải trọng** — khối lượng đất trong gàu tác dụng thành lực xuống tại COM gàu.

Toàn bộ vectorised bằng torch, không import Isaac Lab ⇒ test được trên laptop:

```bash
python tests/test_soil.py     # 20 assertion: lực theo độ sâu, bảo toàn thể tích,
                              # spill, độ độc lập giữa env, không NaN...
```

Tham số mặc định (`SoilCfg`): ρ = 1800 kg/m³, c = 2–20 kPa (random mỗi episode),
dung tích gàu 1.5 m³, bed r = 3–11 m.

---

## 6. Các nút vặn hay dùng

Trong `excavator_rl/digging_env_cfg.py`:

| Tham số | Ý nghĩa |
|---|---|
| `target_fill` | tỉ lệ đầy gàu để chuyển sang pha nâng (0.6) |
| `lift_height` | cao độ pivot gàu để hoàn thành (4.2 m) |
| `action_lowpass` | hằng số lọc lệnh, nhỏ hơn = "mềm" hơn |
| `reward_mode` | `dense` \| `vortex` |
| `control_swing` | thêm khớp quay toa vào action (`--control_swing`) |
| `soil.cohesion_range` | độ cứng đất — mở rộng để domain randomization |
| `w_fill / w_lift / w_success / w_spill` | trọng số reward |

Trong `excavator_rl/excavator_cfg.py`: `EFFORT_LIMITS`, `VELOCITY_LIMITS`,
`DEFAULT_JOINT_POS`, `BUCKET_TIP_OFFSET`.

---

## 7. Giới hạn đã biết

- **Không có thủy lực thật.** Lệnh là vận tốc khớp, không phải lưu lượng van /
  áp suất xy-lanh. CAD không có cơ cấu bốn khâu của xy-lanh nên không thể map trực
  tiếp. Muốn fidelity thủy lực: thêm một lớp actuator tính lực xy-lanh → moment khớp,
  hoặc co-sim với AMESim model bạn đang có.
- **Soil 1-D.** Bật `control_swing` thì nền đất vẫn là profile theo bán kính,
  không phụ thuộc góc quay toa — chấp nhận được cho học chuyển động, không đúng cho
  bài toán múc nhiều điểm.
- **Không có track/di chuyển.** URDF không mô hình xích, root cố định. "Không người lái"
  ở đây = tự động hóa chu trình đào, chưa gồm tự hành.
- **Sim-to-real.** Khối lượng/quán tính lấy từ CAD đặc (SolidWorks), không phải kết cấu
  rỗng thật — nên dùng domain randomization nếu định transfer.

---

## 8. Cấu trúc

```
excavator_isaaclab/
├── excavator_rl/
│   ├── excavator_cfg.py      # khai báo articulation: joint, drive, effort, tip offset
│   ├── soil.py               # soil model giải tích (torch thuần)
│   ├── digging_env_cfg.py    # config task
│   ├── digging_env.py        # DirectRLEnv: obs / action / reward / reset
│   └── agents/rsl_rl_ppo_cfg.py
├── scripts/
│   ├── convert_urdf.py       # sửa URDF + convert sang USD
│   ├── check_model.py        # kiểm tra model + chu kỳ đào scripted
│   ├── train.py              # PPO
│   └── play.py               # chạy checkpoint + thống kê
├── tools/inspect_usd.py      # đọc USD không cần Isaac Sim
└── tests/test_soil.py
```

Dùng được cả với script chính thức của Isaac Lab (task đã register vào gym):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Excavator-Digging-v0
```

---

## Credits

- Excavator CAD/URDF: [mathworks-robotics/autonomous-excavator](https://github.com/mathworks-robotics/autonomous-excavator)
- Task, reward, RL setup: [hanzunye/vortexRL](https://github.com/hanzunye/vortexRL) — Yunze Han, Alexander Stein (KIT)
