# Excavator Digging RL — Isaac Lab

Train the **MathScavator9000** excavator (~36 t, from
[mathworks-robotics/autonomous-excavator](https://github.com/mathworks-robotics/autonomous-excavator))
to perform a **dig-and-lift cycle** in Isaac Sim, with thousands of parallel environments.

The task and reward are ported from [hanzunye/vortexRL](https://github.com/hanzunye/vortexRL)
(Han & Stein, KIT — *Applying Reinforcement Learning to Digital Twin of Excavator to Dig
Automatically*), which ran on Vortex Studio with a single environment.

Two families of RL algorithms are included:

- **Built into Isaac Lab**: PPO from `rsl_rl`, `skrl`, `rl_games` and `Stable-Baselines3`
  (sections 6.1–6.2).
- **The four vortexRL algorithms** — REINFORCE, PPO, TRPO, DDPG — re-implemented in PyTorch
  for thousands of parallel GPU environments (section 6.4, `excavator_rl/algorithms/`).

---

## Contents

0. [Command summary](#0-command-summary)
1. [Requirements](#1-requirements)
2. [Install Isaac Sim + Isaac Lab](#2-install-isaac-sim--isaac-lab)
3. [Install this project](#3-install-this-project)
4. [Excavator model (URDF → USD)](#4-excavator-model-urdf--usd)
5. [Check the model before training](#5-check-the-model-before-training)
6. [Training](#6-training) · [vortexRL algorithms](#64-the-four-vortexrl-algorithms-reinforce--ppo--trpo--ddpg)
7. [Running a trained policy](#7-running-a-trained-policy)
8. [Testing without a GPU (PyBullet)](#8-testing-without-a-gpu-pybullet)
9. [Troubleshooting](#9-troubleshooting)
10. [Task design](#10-task-design) · [Soil model](#11-soil-model) · [Excavator model](#12-excavator-model-checked-and-fixed) · [Parameters](#13-common-parameters) · [Limitations](#14-known-limitations) · [Layout](#15-repository-layout)

---

## 0. Command summary

Once everything is installed (sections 1–4):

```bash
conda activate isaaclab
cd ~/excavator_PIRL

scripts/setup_assets.sh                                                # URDF -> assets/usd/excavator.usd (once)
~/IsaacLab/isaaclab.sh -p scripts/check_model.py --headless            # check model + soil + scripted dig
~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless  # train PPO (rsl_rl)
~/IsaacLab/isaaclab.sh -p scripts/play.py                              # run the newest checkpoint
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ddpg --headless # vortexRL algorithms (6.4)
tensorboard --logdir logs                                              # monitor
```

---

## 1. Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu **22.04** (GLIBC ≥ 2.35) | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA RTX, 8 GB VRAM | RTX with ≥ 16 GB (4096 envs) |
| Driver | see the [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) | latest driver listed there |
| RAM / disk | 32 GB / 50 GB free | 64 GB / SSD |
| Network | Internet on the first run (extensions and NVIDIA's ground-plane asset are downloaded) | |

Quick check:

```bash
nvidia-smi                   # shows the GPU and driver version
ldd --version | head -n1     # GLIBC >= 2.35
```

> Isaac Sim does **not** run on non-RTX GPUs (GTX 10xx, …). Without a suitable GPU you can
> still run the tests and the PyBullet smoke test (section 8).

---

## 2. Install Isaac Sim + Isaac Lab

The project targets **Isaac Sim 5.1 + Isaac Lab v2.3.2** (rsl-rl-lib 3.0.1).
Isaac Lab 2.2 (Isaac Sim 5.0) also works with `scripts/train.py` / `scripts/play.py`.

```bash
# 2.1  Miniconda (skip if conda is already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash

# 2.2  Python 3.11 environment (required by Isaac Sim 5.x)
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
pip install --upgrade pip

# 2.3  PyTorch for CUDA 12.8 + Isaac Sim 5.1 (pip, ~15 GB)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 2.4  Isaac Lab v2.3.2 + all four RL libraries (rsl_rl, skrl, rl_games, sb3)
sudo apt update && sudo apt install -y cmake build-essential git
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout v2.3.2
./isaaclab.sh --install
```

Check that Isaac Lab runs (the first start compiles shaders for a few minutes; the EULA must be
accepted — `export OMNI_KIT_ACCEPT_EULA=YES` does it non-interactively):

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# or train a sample task for a few iterations:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-Direct-v0 --headless --max_iterations 5
```

> All commands below assume `conda activate isaaclab` and Isaac Lab in `~/IsaacLab`.
> If it lives elsewhere: `export ISAACLAB_PATH=/path/to/IsaacLab`.

---

## 3. Install this project

```bash
git clone https://github.com/duc042103/excavator_PIRL.git ~/excavator_PIRL
cd ~/excavator_PIRL
pip install -e .          # optional: makes `import excavator_rl` work from anywhere

# fast tests, NO Isaac Sim needed (torch only):
python tests/test_soil.py        # soil model: forces, cutting, spilling, volume conservation
python tests/test_reward.py      # rewards, phase machine, soil wrench on the bucket
python tests/test_algorithms.py  # the 4 vortexRL algorithms learn a toy task (~1 min)
```

All three must print `all ... tests passed`.

---

## 4. Excavator model (URDF → USD)

The URDF and STL meshes are included in the repository
(`assets/MathScavator9000_flat/`, 12 MB, BSD licence from MathWorks — see its `LICENSE.txt`).
The USD is generated on each machine:

```bash
cd ~/excavator_PIRL
scripts/setup_assets.sh
```

`setup_assets.sh` repairs the URDF (`package://` mesh paths, `effort="0" velocity="0"` limits,
missing damping), runs the Isaac Lab URDF converter with **velocity drives**, and writes
`assets/usd/excavator.usd` — the task's default USD path. To put it elsewhere:
`export EXCAVATOR_USD=/path/to/excavator.usd` (or pass `--usd` to each script).

Other sources: `scripts/setup_assets.sh /path/to/model.urdf` or `/path/to/folder`. Without
`assets/MathScavator9000_flat/` the script clones the MathWorks repository and searches it.

Repair only, without Isaac Sim: `python scripts/convert_urdf.py --input ... --output ... --fix-only`.

---

## 5. Check the model before training

```bash
cd ~/excavator_PIRL
~/IsaacLab/isaaclab.sh -p scripts/check_model.py            # with a 3D viewport
~/IsaacLab/isaaclab.sh -p scripts/check_model.py --headless # numbers only (e.g. over SSH)
```

The script prints what PhysX actually loaded (joint order, limits, **stiffness must be 0**,
link masses) and where the bucket teeth are, then runs a **scripted IK dig cycle**
(`excavator_rl/scripted.py`): bite into the soil, drag the bucket towards the cab at 0.55 m
depth, curl it and lift it. It prints depth, soil resistance and bucket fill over time and must
end with `SUCCESS`.

**This is the most important step.** The same cycle succeeds on the PyBullet twin (section 8);
if it fails in Isaac Sim, the articulation behaves differently (drive gains, limits, joint
signs) — fix that before training.

---

## 6. Training

The task registers two gym ids: `Excavator-Digging-v0` (training) and
`Excavator-Digging-Play-v0` (4 envs, visualised). Each id carries a PPO config for all four
Isaac Lab RL libraries (`excavator_rl/agents/`), with the same `[256, 128, 64]` ELU networks and
hyper-parameters for a fair comparison.

Always run from the repository folder (logs go to `./logs/`):

```bash
cd ~/excavator_PIRL
```

### 6.1 rsl_rl — PPO (recommended starting point)

```bash
~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

| Option | Meaning |
|---|---|
| `--num_envs N` | parallel environments (lower it if VRAM runs out: 2048, 1024, …) |
| `--max_iterations N` | PPO iterations (default 3000; quick test: 50) |
| `--reward_mode vortex` | vortexRL's original reward (default `dense`) |
| `--control_swing` | add the slew joint to the actions (4 actions) |
| `--resume <file.pt>` | continue from a checkpoint |
| `--usd <file.usd>` | use another USD |
| `--seed N`, `--run_name name` | seed, suffix of the run folder |

Drop `--headless` and use `--num_envs 16` to watch it.
Output: `logs/rsl_rl/excavator_digging/<date>_<time>/model_*.pt` + TensorBoard.

### 6.2 Other libraries — through Isaac Lab's own scripts

`scripts/run_rl.py` registers the excavator task and then runs
`IsaacLab/scripts/reinforcement_learning/<library>/{train,play}.py` unchanged, so every option
and hydra override of those scripts works:

```bash
# skrl PPO
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py skrl train --num_envs 4096 --headless

# rl_games PPO
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rl_games train --num_envs 4096 --headless

# Stable-Baselines3 PPO (steps through numpy on the CPU, so slower — use fewer envs)
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py sb3 train --num_envs 1024 --headless

# rsl_rl through the official script (equivalent to 6.1)
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rsl_rl train --num_envs 4096 --headless
```

| Library | Algorithm | Config | Logs |
|---|---|---|---|
| rsl_rl | PPO | `agents/rsl_rl_ppo_cfg.py` | `logs/rsl_rl/excavator_digging/` |
| skrl | PPO | `agents/skrl_ppo_cfg.yaml` | `logs/skrl/excavator_digging/` |
| rl_games | PPO (a2c_continuous) | `agents/rl_games_ppo_cfg.yaml` | `logs/rl_games/excavator_digging/` |
| sb3 | PPO | `agents/sb3_ppo_cfg.yaml` | `logs/sb3/Excavator-Digging-v0/` |

Override example: `... run_rl.py skrl train --headless agent.agent.learning_rate=1e-4`.

> **rl_games**: `minibatch_size` (default 32768) must divide `num_envs × 32`. For other env
> counts add an override, e.g. for 1024 envs: `agent.params.config.minibatch_size=8192`.

### 6.3 Monitoring

```bash
tensorboard --logdir ~/excavator_PIRL/logs      # open http://localhost:6006
```

Useful curves — rsl_rl: `Train/mean_reward`, `Train/mean_episode_length` (shorter = the cycle
is completed faster); vortexRL algorithms (6.4): `Episode/return`, `Episode/success_rate`,
`Episode/length`. On a remote machine: `ssh -L 6006:localhost:6006 user@host`.

### 6.4 The four vortexRL algorithms: REINFORCE · PPO · TRPO · DDPG

[vortexRL](https://github.com/hanzunye/vortexRL) compares four algorithms (TensorFlow, one
Vortex environment). They are re-implemented in PyTorch in `excavator_rl/algorithms/`, keeping
the characteristic architecture and hyper-parameters of the originals but running thousands of
environments in parallel:

| `--algo` | Type | Actions | Kept from vortexRL | Default envs |
|---|---|---|---|---|
| `reinforce` | on-policy, Monte-Carlo, no critic | discrete, 27 = {−1,0,+1}³ | softmax policy, loss −G·log π, return baseline | 512 |
| `ppo` | on-policy, actor-critic | discrete 27 (default) or continuous | clip 0.2, λ 0.97, lr 3e-4 / 1e-3, early stop at KL > 1.5×0.01 | 4096 |
| `trpo` | on-policy, trust region | continuous (Gaussian) | state-independent log-variance, separate value net, λ 0.98 | 1024 |
| `ddpg` | off-policy, actor-critic | continuous | 400-300 actor/critic, OU noise, τ 5e-3, lr 5e-4, replay buffer | 256 |

```bash
cd ~/excavator_PIRL
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ppo       --headless
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo trpo      --headless
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ddpg      --headless
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo reinforce --headless

# run the result (newest checkpoint of that algorithm)
~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --algo ppo
```

| Option | Meaning |
|---|---|
| `--max_iterations N` | iterations (default 1500; quick test: 20) |
| `--num_envs N` | default: see the table above |
| `--cfg key=value ...` | hyper-parameters; field names in `excavator_rl/algorithms/<algo>.py` |
| `--reward_mode vortex` | vortexRL's segmented reward |
| `--resume <file.pt>` | continue training (DDPG refills its replay buffer) |
| `--seed`, `--run_name`, `--usd` | as in 6.1 |

Examples:

```bash
# continuous PPO (as PPO_agentcontinuous.py) instead of 27 discrete actions
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ppo --headless --cfg action_type=continuous
# TRPO with vortexRL's small trust region (kl_targ 0.003)
~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo trpo --headless --cfg max_kl=0.003
# the paper's comparison: vortex reward for all four algorithms
for a in reinforce ppo trpo ddpg; do
  ~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo $a --headless --reward_mode vortex --run_name vortex
done
```

Output: `logs/<algo>/excavator_digging/<date>_<time>/` with `model_<iteration>.pt`,
`model_final.pt`, `config.json` and TensorBoard logs. Compare all four in one plot:
`tensorboard --logdir logs`.

Differences from the originals (documented in each file's docstring):
- Updates use batches of `horizon × num_envs` steps instead of single episodes of one env.
- Time-outs are handled correctly: PPO/TRPO bootstrap them with the critic; DDPG does not
  store transitions cut by a reset (their next observation is already the new episode).
- REINFORCE only uses steps whose Monte-Carlo return is (nearly) complete: the episode ended
  inside the rollout, or at least 3/(1−γ) steps follow.
- vortexRL's TRPO is actually a KL-penalty method (P. Coady's code); this one implements the
  paper's update: natural gradient (conjugate gradient) + KL-bounded line search.
- γ = 0.99 at 30 Hz for all four (vortexRL used γ = 0.9 at 2 Hz ≈ 0.993 at 30 Hz).

---

## 7. Running a trained policy

```bash
# rsl_rl: newest checkpoint; prints bucket fill / soil moved / success per episode
~/IsaacLab/isaaclab.sh -p scripts/play.py
~/IsaacLab/isaaclab.sh -p scripts/play.py --checkpoint logs/rsl_rl/excavator_digging/<run>/model_2999.pt
~/IsaacLab/isaaclab.sh -p scripts/play.py --export     # export policy.pt (TorchScript) + policy.onnx

# other libraries (official play scripts, newest checkpoint)
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py skrl play --num_envs 4
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rl_games play --num_envs 4
~/IsaacLab/isaaclab.sh -p scripts/run_rl.py sb3 play --num_envs 4

# vortexRL algorithms (6.4)
~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --algo ddpg
~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --checkpoint logs/trpo/excavator_digging/<run>/model_1500.pt
```

---

## 8. Testing without a GPU (PyBullet)

`tools/pybullet_twin.py` runs the **real task code** — `DiggingEnv` with its observations, soil
model, rewards, phases and resets, and the real `DiggingEnvCfg` — on PyBullet (CPU) instead of
Isaac Sim. Only the Isaac Lab layer underneath is replaced by a small adapter that loads the same
URDF with the same actuator limits. Use it on any machine to catch geometry, kinematics, reward
or soil mistakes and to try the algorithms end to end:

```bash
pip install pybullet torch gymnasium
python tools/pybullet_twin.py check                                  # model, pose, drives, scripted IK dig
python tools/pybullet_twin.py train --algo ppo --num_envs 16 --iterations 50
```

It is not a replacement for Isaac Sim: PyBullet's velocity motors are stiff constraints capped
at the effort limit (Isaac Lab uses a damped drive), and it runs ~700 control steps/s instead of
hundreds of thousands.

<!-- TWIN-RESULTS -->

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/usr/bin/env: 'bash\r'` or `$'\r': command not found` | files converted to CRLF on Windows. The repo's `.gitattributes` forces LF; for an old clone: `git rm --cached -r . && git reset --hard` (or `sed -i 's/\r$//' scripts/*.sh`) |
| `excavator USD not found at ...` | run `scripts/setup_assets.sh`, or point to the USD with `export EXCAVATOR_USD=...` |
| `Isaac Lab not found at ~/IsaacLab` | `export ISAACLAB_PATH=/path/to/IsaacLab` |
| `ModuleNotFoundError: isaaclab` / `rsl_rl` | run through `~/IsaacLab/isaaclab.sh -p`, not plain `python`; check `conda activate isaaclab` |
| First start hangs for minutes | Isaac Sim compiles shaders / downloads extensions — normal |
| Asks for the EULA and exits | `export OMNI_KIT_ACCEPT_EULA=YES` |
| `CUDA out of memory` | lower `--num_envs` (2048, 1024, 512) |
| No display (SSH) | always add `--headless`; to watch, use `--livestream 2` + the Isaac Sim WebRTC Streaming Client |
| Ground plane / assets fail to load | the first run needs Internet access to NVIDIA's asset server |
| rl_games batch / minibatch error | see the rl_games note in 6.2 |
| Arm falls or does not move | run `check_model.py`: stiffness must be 0 and effort limits non-zero. Re-convert old hand-made USDs with `setup_assets.sh` |
| `check_model.py` scripted dig does not reach `SUCCESS` | compare with `python tools/pybullet_twin.py check`; differing joint limits / drive gains / joint signs point to the USD conversion |

---

## 10. Task design

| | vortexRL (Vortex Studio) | This port (Isaac Lab) |
|---|---|---|
| Actions | 3 × cylinder velocity, low-pass | 3 × joint velocity (boom/stick/bucket), low-pass `α=0.25` |
| State | boom/stick/bucket position + `back` flag | 16-dim: q, q̇, action, tooth xyz, depth, fill ratio, phase flag, height error |
| Soil | particle/mesh hybrid solver | analytic soil model (section 11) |
| Reward | segmented (dig → lift), all negative | `dense` (default) or `vortex` (faithful port) |
| Termination | bucket loaded & higher than 4.2 m | same (`target_fill=0.6`, `lift_height=4.2 m`) |
| Algorithms | REINFORCE / DDPG / PPO / TRPO, 1 env | REINFORCE / DDPG / PPO / TRPO (6.4) + PPO of rsl_rl / skrl / rl_games / sb3, thousands of envs |

`reward_mode="vortex"` ports `Reward/RewardDDPG.py`. The original is a chain of `if`s (not
`elif`), so its `M < M_old` branch is overwritten by the trailing `else`; the **intended**
semantics are implemented here, with a note in the code. Use `--reward_mode vortex` to compare
with the paper; the default `dense` reward converges faster with many parallel envs.

---

## 11. Soil model

Isaac Sim has no deformable soil that scales to thousands of environments, so
`excavator_rl/soil.py` uses an analytic model — as most large-scale excavation-RL work does:

1. **Terrain** — a 1-D height profile `h(r)` along the radius from the swing axis.
2. **Resistance** — Fundamental Earthmoving Equation (Reece):
   `F = w·(ρ·g·d²·N_γ + c·d·N_c + q·d·N_q) + viscous`, opposing the velocity of the teeth and
   applied there, so it produces the right reaction torques on stick and boom. The cut depth `d`
   is measured against the **undisturbed** soil around the blade, not the trench already cut.
3. **Cutting** — swept-min carving: cells the teeth swept through since the previous step are
   lowered to the teeth height; that volume goes into the bucket (× efficiency, up to capacity).
4. **Spilling** — once the teeth are out of the soil and the bucket opening tilts more than
   `spill_angle`, the payload drains out, returns to the bed and slumps to the angle of repose.
   While cutting, soil is pressed into the bucket (which drags with its opening facing the cab)
   and does not spill.
5. **Payload** — the soil mass in the bucket acts as a downward force at the bucket COM.

Defaults (`SoilCfg`): ρ = 1800 kg/m³, c = 2–20 kPa (randomised per episode), bucket capacity
1.5 m³, bed r = 3–11 m. Fully vectorised torch, no Isaac Lab import.

---

## 12. Excavator model: checked and fixed

Read from the MathWorks URDF / `..._physics.usd` and the STL meshes:

| Item | Value in the source asset | Handling |
|---|---|---|
| Kinematic chain | 4 revolute joints: `base_chassis_joint` (swing, Z), `chassis_boom_joint`, `boom_stick_joint`, `stick_bucket_joint` | kept; digging uses the last three |
| Joint limits | boom −32°…+71°, stick −46.9°…+62°, bucket −130°…0°, swing ±180° | kept |
| **Joint drives** | position drives, stiffness 5.4e7 – 2.9e8, unlimited force and velocity | **velocity drives**: stiffness 0, real effort / velocity limits |
| Effort/velocity in the URDF | `effort="0" velocity="0"` | filled in for a 36 t machine |
| Mesh URIs | `package://…` | made absolute by `convert_urdf.py` |
| Moving mass | 33.1 t (chassis 17.4, boom 7.2, stick 3.3, bucket 5.1) | kept — right for the 36 t class |
| Ground | `base_link` bottom at z = −1.295 | root spawned at z = +1.295 |

A drive with stiffness ~1e8 and unlimited force is an infinitely stiff position servo — which
is why `chassis_boom_joint` did not respond to velocity / torque commands before.

Geometry from the STL meshes: boom 6.24 m · stick 2.98 m · bucket pivot → tooth tips 2.18 m ·
bucket width 1.148 m · max tooth reach 11.1 m · tooth tips (bucket frame) = (0, −1.813, 1.207) ·
bucket opening direction (bucket frame) = (0, 0.554, 0.832).

Dig-entry pose (`DEFAULT_JOINT_POS`): teeth on the soil surface 7.5 m out, pointing 75° below
horizontal, bucket opening facing the cab.

Inspect any USD without Isaac Sim: `pip install usd-core && python tools/inspect_usd.py file.usd`.

---

## 13. Common parameters

`excavator_rl/digging_env_cfg.py`:

| Parameter | Meaning |
|---|---|
| `target_fill` | bucket fill ratio that starts the lift phase (0.6) |
| `lift_height` | bucket pivot height that completes the task (4.2 m) |
| `action_lowpass` | command filter coefficient; smaller = smoother |
| `reward_mode` | `dense` \| `vortex` |
| `control_swing` | add the slew joint to the actions |
| `soil.cohesion_range` | soil stiffness — widen it for domain randomisation |
| `w_fill / w_lift / w_success / w_spill` | reward weights |

`excavator_rl/excavator_params.py`: `EFFORT_LIMITS`, `VELOCITY_LIMITS`,
`VELOCITY_TRACKING_GAIN`, `DEFAULT_JOINT_POS`, `BUCKET_TIP_OFFSET`, `BUCKET_OPEN_DIR`.

PPO hyper-parameters: `excavator_rl/agents/`; vortexRL algorithms: `excavator_rl/algorithms/`.

---

## 14. Known limitations

- **No real hydraulics.** Commands are joint velocities, not valve flow / cylinder pressure
  (the CAD has no cylinder linkage).
- **1-D soil.** With `control_swing` the bed is still a radial profile, independent of the
  slew angle.
- **No tracks / travel.** The root is fixed; "autonomous" means automating the dig cycle.
- **Sim-to-real.** Masses and inertias come from solid CAD (SolidWorks) — use domain
  randomisation before transferring to a real machine.
- **Not yet run in Isaac Sim by the authors of these changes.** Everything was validated with
  the unit tests and the PyBullet twin (section 8); hyper-parameters may need tuning in Isaac Sim.

---

## 15. Repository layout

```
excavator_PIRL/
├── excavator_rl/
│   ├── __init__.py            # gym ids + agent config entry points for the 4 RL libraries
│   ├── excavator_params.py    # constants: joints, limits, efforts, geometry (pure python)
│   ├── excavator_cfg.py       # ArticulationCfg: drives, actuators
│   ├── soil.py                # analytic soil model (pure torch)
│   ├── scripted.py            # arm kinematics + scripted IK dig cycle
│   ├── digging_env_cfg.py     # task config
│   ├── digging_env.py         # DirectRLEnv: obs / action / reward / reset
│   ├── agents/                # PPO: rsl_rl (.py), skrl / rl_games / sb3 (.yaml)
│   └── algorithms/            # vortexRL: reinforce.py, ppo.py, trpo.py, ddpg.py (pure PyTorch)
├── scripts/
│   ├── setup_assets.sh        # find URDF -> repair -> convert to USD
│   ├── convert_urdf.py        # repair URDF + convert to USD
│   ├── check_model.py         # model check + scripted IK dig cycle (Isaac Sim)
│   ├── train.py / play.py     # rsl_rl PPO
│   ├── run_rl.py              # Isaac Lab's official train/play scripts
│   └── train_algo.py / play_algo.py  # the 4 vortexRL algorithms
├── assets/
│   └── MathScavator9000_flat/ # URDF + STL meshes (MathWorks, BSD); usd/ is generated
├── tools/
│   ├── pybullet_twin.py       # run the task on PyBullet, no GPU needed
│   └── inspect_usd.py         # read a USD without Isaac Sim
└── tests/                     # test_soil / test_reward / test_algorithms (no Isaac Sim)
```

---

## Credits

- Excavator CAD/URDF: [mathworks-robotics/autonomous-excavator](https://github.com/mathworks-robotics/autonomous-excavator) (BSD licence, `assets/MathScavator9000_flat/LICENSE.txt`)
- Task, reward, RL setup and the REINFORCE / DDPG / PPO / TRPO algorithms: [hanzunye/vortexRL](https://github.com/hanzunye/vortexRL) — Yunze Han, Alexander Stein (KIT)
- Simulator & RL framework: [Isaac Sim](https://developer.nvidia.com/isaac/sim), [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
