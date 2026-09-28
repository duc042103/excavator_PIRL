# Excavator Digging RL — Isaac Lab

Huấn luyện máy xúc **MathScavator9000** (~36 t, từ
[mathworks-robotics/autonomous-excavator](https://github.com/mathworks-robotics/autonomous-excavator))
thực hiện **chu trình đào + nâng gàu** trong Isaac Sim, với hàng nghìn môi trường song song.

Task và reward port từ [hanzunye/vortexRL](https://github.com/hanzunye/vortexRL)
(Han & Stein, KIT — *Applying Reinforcement Learning to Digital Twin of Excavator to Dig
Automatically*), vốn chạy trên Vortex Studio với 1 môi trường.

Thuật toán RL có 2 nhóm:

- **Có sẵn trong Isaac Lab**: PPO của 4 thư viện `rsl_rl`, `skrl`, `rl_games`,
  `Stable-Baselines3` (mục 6.1–6.2).
- **4 thuật toán của vortexRL** — REINFORCE, PPO, TRPO, DDPG — viết lại bằng PyTorch để chạy
  hàng nghìn môi trường song song trên GPU (mục 6.4, thư mục `excavator_rl/algorithms/`).

---

## Mục lục

0. [Tóm tắt lệnh](#0-tóm-tắt-lệnh)
1. [Yêu cầu máy](#1-yêu-cầu-máy)
2. [Cài Isaac Sim + Isaac Lab](#2-cài-isaac-sim--isaac-lab)
3. [Cài project này](#3-cài-project-này)
4. [Chuẩn bị model máy xúc (URDF → USD)](#4-chuẩn-bị-model-máy-xúc-urdf--usd)
5. [Kiểm tra model trước khi train](#5-kiểm-tra-model-trước-khi-train)
6. [Train bằng các thuật toán RL có sẵn](#6-train-bằng-các-thuật-toán-rl-có-sẵn) · [Thuật toán vortexRL](#64-4-thuật-toán-của-vortexrl-reinforce--ppo--trpo--ddpg)
7. [Xem policy đã train](#7-xem-policy-đã-train)
8. [Lỗi thường gặp](#8-lỗi-thường-gặp)
9. [Thiết kế task](#9-thiết-kế-task) · [Soil model](#10-soil-model) · [Model máy xúc](#11-model-máy-xúc-đã-kiểm-tra-và-sửa) · [Tham số](#12-các-tham-số-hay-chỉnh) · [Giới hạn](#13-giới-hạn-đã-biết) · [Cấu trúc](#14-cấu-trúc-thư-mục)

---

## 0. Tóm tắt lệnh

Khi đã cài xong (mục 1–4), toàn bộ quy trình là:

```bash
conda activate isaaclab
cd ~/excavator_PIRL

scripts/setup_assets.sh                                              # URDF -> assets/usd/excavator.usd (1 lần)
~/IsaacLab/isaaclab.sh -p scripts/check_model.py --headless          # kiểm tra model + soil
~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless  # train PPO (rsl_rl)
~/IsaacLab/isaaclab.sh -p scripts/play.py                             # xem checkpoint mới nhất
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ddpg --headless # thuật toán vortexRL (mục 6.4)
tensorboard --logdir logs                                             # theo dõi
```

---

## 1. Yêu cầu máy

| | Tối thiểu | Khuyến nghị |
|---|---|---|
| OS | Ubuntu **22.04** (cần GLIBC ≥ 2.35) | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA RTX, 8 GB VRAM | RTX ≥ 16 GB (4096 env) |
| Driver | theo [yêu cầu của Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) | bản driver mới nhất trong trang đó |
| RAM / ổ cứng | 32 GB / 50 GB trống | 64 GB / SSD |
| Mạng | cần Internet lần chạy đầu (tải extension + asset ground plane của NVIDIA) | |

Kiểm tra nhanh:

```bash
nvidia-smi                   # thấy GPU + driver version
ldd --version | head -n1     # GLIBC >= 2.35
```

> GPU không phải RTX (GTX 10xx, …) **không chạy được** Isaac Sim.

---

## 2. Cài Isaac Sim + Isaac Lab

Project được viết và kiểm tra theo **Isaac Sim 5.1 + Isaac Lab v2.3.2** (rsl-rl-lib 3.0.1).
Isaac Lab 2.2 (Isaac Sim 5.0) vẫn chạy được với `scripts/train.py` / `scripts/play.py`.

```bash
# 2.1  Miniconda (bỏ qua nếu đã có conda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash

# 2.2  Môi trường Python 3.11 (Isaac Sim 5.x bắt buộc 3.11)
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
pip install --upgrade pip

# 2.3  PyTorch CUDA 12.8 + Isaac Sim 5.1 (pip, ~15 GB)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 2.4  Isaac Lab v2.3.2 + cả 4 thư viện RL (rsl_rl, skrl, rl_games, sb3)
sudo apt update && sudo apt install -y cmake build-essential git
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout v2.3.2
./isaaclab.sh --install
```

Kiểm tra Isaac Lab chạy được (lần đầu mất vài phút để biên dịch shader; phải đồng ý EULA —
có thể đặt sẵn `export OMNI_KIT_ACCEPT_EULA=YES`):

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# hoặc chạy thử một task mẫu:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-Direct-v0 --headless --max_iterations 5
```

> Mọi lệnh bên dưới đều giả định đã `conda activate isaaclab` và Isaac Lab nằm ở `~/IsaacLab`.
> Nếu để chỗ khác: `export ISAACLAB_PATH=/đường/dẫn/IsaacLab`.

---

## 3. Cài project này

```bash
git clone https://github.com/duc042103/excavator_PIRL.git ~/excavator_PIRL
cd ~/excavator_PIRL
pip install -e .          # tuỳ chọn: cho phép `import excavator_rl` từ bất kỳ đâu

# test nhanh, KHÔNG cần Isaac Sim (chỉ cần torch):
python tests/test_soil.py      # soil model: lực, bóc đất, rơi vãi, bảo toàn thể tích
python tests/test_reward.py    # reward, phase machine, wrench đất lên gàu
python tests/test_algorithms.py  # 4 thuật toán vortexRL học được một bài toán mẫu (~1 phút)
```

Cả ba phải in `all ... tests passed`.

---

## 4. Chuẩn bị model máy xúc (URDF → USD)

Repo **không** chứa sẵn file USD (nó được sinh riêng trên từng máy). Cần URDF + mesh STL gốc:

```
MathScavator9000_flat/
├── urdf/MathScavator9000_flat.SLDASM.urdf
└── meshes/*.STL
```

**Cách A — khuyến nghị: đưa model vào repo một lần** (từ máy Windows đang có sẵn thư mục này):

```powershell
# Windows PowerShell, trong thư mục repo
Copy-Item -Recurse <đường dẫn>\MathScavator9000_flat assets\MathScavator9000_flat
git add assets/MathScavator9000_flat
git commit -m "Add excavator URDF and meshes"
git push
```

Sau đó mọi máy Ubuntu chỉ cần `git pull` rồi chạy `scripts/setup_assets.sh`.

**Cách B — chỉ ra đường dẫn:** copy thư mục đó sang máy Ubuntu (USB, `scp`, …) rồi:

```bash
scripts/setup_assets.sh ~/Downloads/MathScavator9000_flat
```

**Cách C — tự tải:** chạy `scripts/setup_assets.sh` không có tham số và không có
`assets/MathScavator9000_flat/`; script sẽ `git clone` repo MathWorks và tìm URDF trong đó.
(Nếu repo gốc đổi cấu trúc, dùng cách A/B.)

`setup_assets.sh` làm 3 việc: sửa URDF (đường dẫn mesh `package://`, effort/velocity limit
bằng 0, thiếu damping) → chạy Isaac Lab URDF converter với **velocity drive** →
ghi ra `assets/usd/excavator.usd`, là đường dẫn mặc định của task. Muốn để chỗ khác:
`export EXCAVATOR_USD=/đường/dẫn/excavator.usd` (hoặc `--usd` trên từng script).

Chỉ sửa URDF, không cần Isaac Sim: `python scripts/convert_urdf.py --input ... --output ... --fix-only`.

---

## 5. Kiểm tra model trước khi train

```bash
cd ~/excavator_PIRL
~/IsaacLab/isaaclab.sh -p scripts/check_model.py            # có cửa sổ 3D
~/IsaacLab/isaaclab.sh -p scripts/check_model.py --headless # chỉ in số liệu (máy qua SSH)
```

Script in ra những gì PhysX thật sự load (thứ tự khớp, limit, **stiffness phải = 0**, khối
lượng link), vị trí lưỡi gàu, rồi chạy một chu trình đào viết tay 14 s và in theo thời gian:
độ sâu, lực cản đất, độ đầy gàu.

**Đây là bước quan trọng nhất.** Nếu depth / lực / fill đều = 0 suốt chu trình thì tip offset
hoặc bán kính soil bed sai — sửa trước (`BUCKET_TIP_OFFSET` trong
`excavator_rl/excavator_params.py`), đừng train.

---

## 6. Train bằng các thuật toán RL có sẵn

Task đăng ký 2 ID gym: `Excavator-Digging-v0` (train) và `Excavator-Digging-Play-v0`
(4 env, có hiển thị). Mỗi ID mang sẵn cấu hình PPO cho cả 4 thư viện RL của Isaac Lab
(`excavator_rl/agents/`), cùng mạng `[256, 128, 64]` ELU và cùng siêu tham số, để so sánh
công bằng.

Luôn chạy từ thư mục repo (log ghi vào `./logs/`):

```bash
cd ~/excavator_PIRL
```

### 6.1 rsl_rl — PPO (khuyến nghị bắt đầu từ đây)

```bash
~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--num_envs N` | số môi trường song song (giảm nếu thiếu VRAM: 2048, 1024…) |
| `--max_iterations N` | số vòng PPO (mặc định 3000; thử nhanh: 50) |
| `--reward_mode vortex` | reward nguyên bản của vortexRL (mặc định `dense`) |
| `--control_swing` | thêm khớp quay toa vào action (4 action) |
| `--resume <file.pt>` | train tiếp từ checkpoint |
| `--usd <file.usd>` | dùng USD khác |
| `--seed N`, `--run_name tên` | seed, hậu tố tên thư mục run |

Bỏ `--headless` và dùng `--num_envs 16` để xem trực tiếp.
Kết quả: `logs/rsl_rl/excavator_digging/<ngày>_<giờ>/model_*.pt` + TensorBoard.

### 6.2 Các thư viện khác — qua script chính thức của Isaac Lab

`scripts/run_rl.py` đăng ký task máy xúc rồi chạy **nguyên** script
`IsaacLab/scripts/reinforcement_learning/<thư viện>/{train,play}.py`, nên mọi tuỳ chọn và
hydra override của script gốc đều dùng được:

```bash
# PPO của skrl
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py skrl train --num_envs 4096 --headless

# PPO của rl_games
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rl_games train --num_envs 4096 --headless

# PPO của Stable-Baselines3 (chạy trên CPU/numpy nên chậm — dùng ít env)
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py sb3 train --num_envs 1024 --headless

# rsl_rl bằng script chính thức (tương đương mục 6.1)
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rsl_rl train --num_envs 4096 --headless
```

| Thư viện | Thuật toán | Cấu hình | Log |
|---|---|---|---|
| rsl_rl | PPO | `agents/rsl_rl_ppo_cfg.py` | `logs/rsl_rl/excavator_digging/` |
| skrl | PPO | `agents/skrl_ppo_cfg.yaml` | `logs/skrl/excavator_digging/` |
| rl_games | PPO (a2c_continuous) | `agents/rl_games_ppo_cfg.yaml` | `logs/rl_games/excavator_digging/` |
| sb3 | PPO | `agents/sb3_ppo_cfg.yaml` | `logs/sb3/Excavator-Digging-v0/` |

Ví dụ override: `... run_rl.py skrl train --headless agent.agent.learning_rate=1e-4`.

> **rl_games**: `minibatch_size` (mặc định 32768) phải chia hết `num_envs × 32`. Với số env khác
> 4096, thêm override, ví dụ 1024 env: `agent.params.config.minibatch_size=8192`.

### 6.3 Theo dõi

```bash
tensorboard --logdir ~/excavator_PIRL/logs      # mở http://localhost:6006
```

Chỉ số đáng xem — rsl_rl: `Train/mean_reward`, `Train/mean_episode_length` (giảm = đào xong
nhanh hơn); thuật toán vortexRL (mục 6.4): `Episode/return`, `Episode/success_rate`,
`Episode/length`. Máy chủ qua SSH: `ssh -L 6006:localhost:6006 user@máy`.

### 6.4 4 thuật toán của vortexRL: REINFORCE · PPO · TRPO · DDPG

[vortexRL](https://github.com/hanzunye/vortexRL) so sánh 4 thuật toán (TensorFlow, 1 môi trường
Vortex). Ở đây chúng được viết lại bằng PyTorch trong `excavator_rl/algorithms/`, giữ kiến
trúc và siêu tham số đặc trưng của bản gốc, nhưng chạy song song hàng nghìn môi trường:

| `--algo` | Loại | Action | Giữ từ vortexRL | Số env mặc định |
|---|---|---|---|---|
| `reinforce` | on-policy, Monte-Carlo, không critic | rời rạc 27 = {−1,0,+1}³ | softmax policy, loss −G·log π, baseline cho return | 512 |
| `ppo` | on-policy, actor-critic | rời rạc 27 (mặc định) hoặc liên tục | clip 0.2, λ 0.97, lr 3e-4 / 1e-3, dừng sớm khi KL > 1.5×0.01 | 4096 |
| `trpo` | on-policy, trust region | liên tục (Gauss) | log-var độc lập trạng thái, value net riêng, λ 0.98 | 1024 |
| `ddpg` | off-policy, actor-critic | liên tục | actor/critic 400-300, nhiễu OU, τ 5e-3, lr 5e-4, replay buffer | 256 |

```bash
cd ~/excavator_PIRL
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ppo       --headless
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo trpo      --headless
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ddpg      --headless
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo reinforce --headless

# xem kết quả (checkpoint mới nhất của thuật toán đó)
~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --algo ppo
```

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--max_iterations N` | số vòng lặp (mặc định 1500; thử nhanh: 20) |
| `--num_envs N` | mặc định theo bảng trên |
| `--cfg key=value ...` | đổi siêu tham số, tên trường xem ở `excavator_rl/algorithms/<algo>.py` |
| `--reward_mode vortex` | dùng đúng reward phân đoạn của vortexRL |
| `--resume <file.pt>` | train tiếp (DDPG: replay buffer được nạp lại từ đầu) |
| `--seed`, `--run_name`, `--usd` | như mục 6.1 |

Ví dụ:

```bash
# PPO liên tục (như PPO_agentcontinuous.py) thay vì 27 action rời rạc
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ppo --headless --cfg action_type=continuous
# TRPO với vùng tin cậy nhỏ như vortexRL (kl_targ 0.003)
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo trpo --headless --cfg max_kl=0.003
# so sánh đúng như paper: reward vortex cho cả 4 thuật toán
for a in reinforce ppo trpo ddpg; do
  ~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo $a --headless --reward_mode vortex --run_name vortex
done
```

Kết quả: `logs/<algo>/excavator_digging/<ngày>_<giờ>/` gồm `model_<vòng>.pt`, `model_final.pt`,
`config.json` và log TensorBoard. So sánh cả 4 trên cùng biểu đồ: `tensorboard --logdir logs`.

Khác biệt so với bản gốc (có ghi rõ trong docstring từng file):
- Cập nhật theo lô `horizon × num_envs` bước thay vì theo từng episode của 1 môi trường.
- Time-out được xử lý đúng: PPO/TRPO bootstrap bằng critic; DDPG không lưu transition bị cắt
  vì reset (quan sát kế tiếp đã là trạng thái mới).
- REINFORCE chỉ dùng các bước có return Monte-Carlo gần đầy đủ (episode kết thúc trong lượt
  rollout, hoặc còn ≥ 3/(1−γ) bước phía sau).
- TRPO của vortexRL thực chất là biến thể phạt KL (code của P. Coady); bản này cài TRPO đúng
  theo paper: gradient tự nhiên (conjugate gradient) + line search giới hạn KL.
- γ = 0.99 ở 30 Hz cho cả 4 (vortexRL dùng γ = 0.9 ở 2 Hz ≈ 0.993 ở 30 Hz).

---

## 7. Xem policy đã train

```bash
# rsl_rl: tự lấy checkpoint mới nhất, in độ đầy gàu / lượng đất / success mỗi episode
~/IsaacLab/isaaclab.sh -p scripts/play.py
~/IsaacLab/isaaclab.sh -p scripts/play.py --checkpoint logs/rsl_rl/excavator_digging/<run>/model_2999.pt
~/IsaacLab/isaaclab.sh -p scripts/play.py --export     # xuất policy.pt (TorchScript) + policy.onnx

# các thư viện khác (script play chính thức, tự tìm checkpoint mới nhất)
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py skrl play --num_envs 4
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rl_games play --num_envs 4
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py sb3 play --num_envs 4

# thuật toán vortexRL (mục 6.4)
~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --algo ddpg
~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --checkpoint logs/trpo/excavator_digging/<run>/model_1500.pt
```

---

## 8. Lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách sửa |
|---|---|
| `/usr/bin/env: 'bash\r'` hoặc `$'\r': command not found` | file bị đổi sang CRLF trên Windows. Repo đã có `.gitattributes` ép LF; với bản clone cũ: `git rm --cached -r . && git reset --hard` (hoặc `sed -i 's/\r$//' scripts/*.sh`) |
| `excavator USD not found at ...` | chưa chạy `scripts/setup_assets.sh`, hoặc USD ở chỗ khác → `export EXCAVATOR_USD=...` |
| `Isaac Lab not found at ~/IsaacLab` | `export ISAACLAB_PATH=/đường/dẫn/IsaacLab` |
| `ModuleNotFoundError: isaaclab` / `rsl_rl` | chạy bằng `~/IsaacLab/isaaclab.sh -p`, không dùng `python` trực tiếp; kiểm tra `conda activate isaaclab` |
| Lần đầu chạy treo vài phút | Isaac Sim đang biên dịch shader / tải extension — bình thường |
| Hỏi EULA rồi thoát | `export OMNI_KIT_ACCEPT_EULA=YES` |
| `CUDA out of memory` | giảm `--num_envs` (2048, 1024, 512) |
| Không có màn hình (SSH) | luôn thêm `--headless`; muốn xem thì dùng `--livestream 2` + ứng dụng Isaac Sim WebRTC Streaming Client |
| Không tải được ground plane / asset | cần Internet tới server asset của NVIDIA ở lần chạy đầu |
| rl_games báo lỗi batch/minibatch | xem ghi chú rl_games ở mục 6.2 |
| Tay máy xúc rơi / không nhúc nhích | chạy `check_model.py`: stiffness phải = 0, effort limit khác 0. Nếu dùng USD cũ convert thủ công thì convert lại bằng `setup_assets.sh` |

---

## 9. Thiết kế task

| | vortexRL (Vortex Studio) | Bản port này (Isaac Lab) |
|---|---|---|
| Action | 3 × vận tốc xy-lanh, low-pass | 3 × vận tốc khớp (boom/stick/bucket), low-pass `α=0.25` |
| State | boom/stick/bucket position + cờ `back` | 16 chiều: q, q̇, action, tip xyz, depth, fill ratio, cờ phase, sai số độ cao |
| Đất | particle/mesh hybrid solver | soil model giải tích (mục 10) |
| Reward | phân đoạn (đào → nâng), toàn giá trị âm | `dense` (mặc định) hoặc `vortex` (port nguyên bản) |
| Kết thúc | gàu đủ đất & cao > 4.2 m | như trên (`target_fill=0.6`, `lift_height=4.2 m`) |
| Thuật toán | REINFORCE / DDPG / PPO / TRPO, 1 env | REINFORCE / DDPG / PPO / TRPO (mục 6.4) + PPO của rsl_rl / skrl / rl_games / sb3, hàng nghìn env |

`reward_mode="vortex"` port đúng `Reward/RewardDDPG.py`. Bản gốc dùng chuỗi `if` (không phải
`elif`) nên nhánh `M < M_old` bị nhánh `else` ghi đè — ở đây implement theo **ý định** của tác
giả, có ghi chú trong code. Dùng `--reward_mode vortex` để so sánh với paper; dùng mặc định
`dense` để PPO hội tụ nhanh với nhiều env song song.

---

## 10. Soil model

Isaac Sim không có đất biến dạng chạy nổi ở quy mô nghìn env, nên `excavator_rl/soil.py` dùng
mô hình giải tích — cách hầu hết paper excavation-RL quy mô lớn vẫn làm:

1. **Địa hình** — height profile 1-D `h(r)` theo bán kính từ trục swing.
2. **Lực cản** — Fundamental Earthmoving Equation (Reece):
   `F = w·(ρ·g·d²·N_γ + c·d·N_c + q·d·N_q) + viscous`, ngược chiều vận tốc lưỡi gàu, đặt tại
   lưỡi cắt ⇒ sinh đúng moment phản lực lên stick và boom.
3. **Bóc đất** — swept-min carving: ô đất lưỡi cắt quét qua bị hạ xuống bằng cao độ lưỡi,
   thể tích chênh lệch vào gàu (× hiệu suất, chặn ở dung tích gàu).
4. **Rơi vãi** — gàu nghiêng quá `spill_angle` thì đất chảy ra, trả lại nền và trượt theo góc nghỉ.
5. **Tải trọng** — khối lượng đất trong gàu tác dụng thành lực xuống tại COM gàu.

Tham số mặc định (`SoilCfg`): ρ = 1800 kg/m³, c = 2–20 kPa (random mỗi episode), dung tích gàu
1.5 m³, bed r = 3–11 m. Toàn bộ vectorised bằng torch, không import Isaac Lab.

---

## 11. Model máy xúc: đã kiểm tra và sửa

Đọc trực tiếp từ `MathScavator9000_flat.SLDASM_physics.usd` + các file STL:

| Hạng mục | Giá trị trong asset gốc | Xử lý |
|---|---|---|
| Kinematic chain | 4 revolute: `base_chassis_joint` (swing, Z), `chassis_boom_joint`, `boom_stick_joint`, `stick_bucket_joint` | giữ nguyên; digging dùng 3 khớp sau |
| Joint limits | boom −32°…+71°, stick −46.9°…+62°, bucket −130°…0°, swing ±180° | giữ nguyên |
| **Joint drives** | position drive, stiffness 5.4e7 – 2.9e8, maxForce vô hạn, maxJointVelocity vô hạn | **velocity drive**: stiffness = 0, effort/velocity limit thật |
| Effort/velocity trong URDF | `effort="0" velocity="0"` | điền theo lực đào máy 36 t |
| Mesh URI | `package://…` | `convert_urdf.py` đổi sang đường dẫn tuyệt đối |
| Khối lượng phần chuyển động | 33.1 t (chassis 17.4, boom 7.2, stick 3.3, bucket 5.1) | giữ nguyên — đúng class 36 t |
| Ground | `base_link` chạm đáy ở z = −1.295 | spawn root ở z = +1.295 |

Drive stiffness ~1e8 với lực không giới hạn là servo vị trí cứng tuyệt đối — đó là lý do trước
đây `chassis_boom_joint` "không phản hồi" lệnh vận tốc/moment.

Hình học đo từ STL: boom 6.24 m · stick 2.98 m · bucket pivot→lưỡi cắt 1.835 m · rộng gàu
1.148 m · tầm với tối đa 10.79 m · tip offset (bucket frame) = (0, −1.835, −0.04).

Tự kiểm tra một file USD bất kỳ (không cần Isaac Sim): `pip install usd-core && python tools/inspect_usd.py file.usd`.

---

## 12. Các tham số hay chỉnh

`excavator_rl/digging_env_cfg.py`:

| Tham số | Ý nghĩa |
|---|---|
| `target_fill` | tỉ lệ đầy gàu để chuyển sang pha nâng (0.6) |
| `lift_height` | cao độ pivot gàu để hoàn thành (4.2 m) |
| `action_lowpass` | hệ số lọc lệnh, nhỏ hơn = "mềm" hơn |
| `reward_mode` | `dense` \| `vortex` |
| `control_swing` | thêm khớp quay toa vào action |
| `soil.cohesion_range` | độ cứng đất — mở rộng để domain randomization |
| `w_fill / w_lift / w_success / w_spill` | trọng số reward |

`excavator_rl/excavator_params.py`: `EFFORT_LIMITS`, `VELOCITY_LIMITS`,
`VELOCITY_TRACKING_GAIN`, `DEFAULT_JOINT_POS`, `BUCKET_TIP_OFFSET`.

Siêu tham số PPO: các file trong `excavator_rl/agents/`.

---

## 13. Giới hạn đã biết

- **Không có thủy lực thật.** Lệnh là vận tốc khớp, không phải lưu lượng van / áp suất xy-lanh
  (CAD không có cơ cấu xy-lanh).
- **Soil 1-D.** Bật `control_swing` thì nền đất vẫn là profile theo bán kính, không phụ thuộc
  góc quay toa.
- **Không có xích/di chuyển.** Root cố định; "tự động" ở đây là tự động hóa chu trình đào.
- **Sim-to-real.** Khối lượng/quán tính lấy từ CAD đặc (SolidWorks) — cần domain randomization
  nếu định chuyển sang máy thật.
- **Thuật toán vortexRL chưa được tinh chỉnh trên máy xúc thật trong Isaac Sim.** Chúng đã
  được kiểm tra học được trên bài toán mẫu (`tests/test_algorithms.py`); siêu tham số trên
  task đào có thể cần chỉnh (`--cfg`).

---

## 14. Cấu trúc thư mục

```
excavator_PIRL/
├── excavator_rl/
│   ├── __init__.py            # đăng ký gym ID + entry point cho 4 thư viện RL
│   ├── excavator_params.py    # hằng số: tên khớp, limit, effort, hình học (python thuần)
│   ├── excavator_cfg.py       # ArticulationCfg: drive, actuator
│   ├── soil.py                # soil model giải tích (torch thuần)
│   ├── digging_env_cfg.py     # config task
│   ├── digging_env.py         # DirectRLEnv: obs / action / reward / reset
│   ├── agents/                # PPO: rsl_rl (.py), skrl / rl_games / sb3 (.yaml)
│   └── algorithms/            # vortexRL: reinforce.py, ppo.py, trpo.py, ddpg.py (PyTorch thuần)
├── scripts/
│   ├── setup_assets.sh        # tìm URDF -> sửa -> convert USD
│   ├── convert_urdf.py        # sửa URDF + convert sang USD
│   ├── check_model.py         # kiểm tra model + chu kỳ đào viết tay
│   ├── train.py / play.py     # PPO rsl_rl
│   ├── run_rl.py              # chạy script train/play chính thức của Isaac Lab
│   └── train_algo.py / play_algo.py  # 4 thuật toán vortexRL
├── assets/                    # đặt MathScavator9000_flat/ ở đây; usd/ được sinh ra
├── tools/inspect_usd.py       # đọc USD không cần Isaac Sim
└── tests/                     # test_soil / test_reward / test_algorithms (không cần Isaac Sim)
```

---

## Credits

- Excavator CAD/URDF: [mathworks-robotics/autonomous-excavator](https://github.com/mathworks-robotics/autonomous-excavator)
- Task, reward, RL setup, 4 thuật toán REINFORCE / DDPG / PPO / TRPO: [hanzunye/vortexRL](https://github.com/hanzunye/vortexRL) — Yunze Han, Alexander Stein (KIT)
- Simulator & RL framework: [Isaac Sim](https://developer.nvidia.com/isaac/sim), [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
