#!/usr/bin/env python3
"""Run Isaac Lab's own train / play scripts on the excavator task.

Isaac Lab ships ready-made PPO training scripts for four RL libraries
(rsl_rl, skrl, rl_games, Stable-Baselines3) under
``IsaacLab/scripts/reinforcement_learning/<library>/``.  They only know the
tasks registered by ``isaaclab_tasks``; this wrapper registers the excavator
task first and then hands over to the official script unchanged, so every
command-line option (and hydra override) of that script keeps working.

    cd ~/excavator_PIRL
    ~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rsl_rl   train --headless --num_envs 4096
    ~/IsaacLab/isaaclab.sh -p scripts/run_rl.py skrl     train --headless --num_envs 4096
    ~/IsaacLab/isaaclab.sh -p scripts/run_rl.py rl_games train --headless --num_envs 4096
    ~/IsaacLab/isaaclab.sh -p scripts/run_rl.py sb3      train --headless --num_envs 1024

    ~/IsaacLab/isaaclab.sh -p scripts/run_rl.py skrl play --num_envs 4

``--task`` defaults to Excavator-Digging-v0 for ``train`` and
Excavator-Digging-Play-v0 for ``play``.  Logs go to ``logs/<library>/`` under
the current directory.

The Isaac Lab checkout is found through ``$ISAACLAB_PATH`` (exported by
``isaaclab.sh``), falling back to ``~/IsaacLab``.
"""

from __future__ import annotations

import os
import runpy
import sys

LIBRARIES = ("rsl_rl", "skrl", "rl_games", "sb3")
MODES = ("train", "play")
DEFAULT_TASK = {"train": "Excavator-Digging-v0", "play": "Excavator-Digging-Play-v0"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def usage(msg: str = "") -> None:
    if msg:
        print(f"error: {msg}\n", file=sys.stderr)
    print(
        "usage: isaaclab.sh -p scripts/run_rl.py {" + ",".join(LIBRARIES) + "} {train,play} "
        "[options of the Isaac Lab script ...]",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        usage()
    lib, mode, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if lib not in LIBRARIES:
        usage(f"unknown library '{lib}'")
    if mode not in MODES:
        usage(f"unknown mode '{mode}'")

    isaaclab_path = os.environ.get("ISAACLAB_PATH") or os.path.expanduser("~/IsaacLab")
    script = os.path.join(isaaclab_path, "scripts", "reinforcement_learning", lib, f"{mode}.py")
    if not os.path.isfile(script):
        usage(
            f"{script} not found.\n"
            "Run this through ~/IsaacLab/isaaclab.sh -p (which exports ISAACLAB_PATH), "
            "or export ISAACLAB_PATH=/path/to/IsaacLab."
        )

    if not any(a == "--task" or a.startswith("--task=") for a in rest):
        rest = ["--task", DEFAULT_TASK[mode]] + rest

    # register the gym ids; excavator_rl/__init__.py only imports gymnasium, so
    # this is safe before the official script launches the simulator
    sys.path.insert(0, REPO_ROOT)
    import excavator_rl  # noqa: F401

    # the official scripts import helpers that sit next to them (cli_args.py)
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script] + rest
    print(f"[run_rl] {lib} {mode}: {script} {' '.join(rest)}")
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
