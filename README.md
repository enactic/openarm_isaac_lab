# OpenArm Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

This repository provides simulation and learning environments for the **OpenArm robotic platform**, built on **NVIDIA Isaac Sim** and **Isaac Lab**.
It enables research and development in **reinforcement learning (RL)**, **imitation learning (IL)**, **teleoperation**, and **sim-to-real transfer** for both **unimanual (single-arm)** and **bimanual (dual-arm)** robotic systems.

### What this repo offers
- **Isaac Sim models** for OpenArm robots.
- **Isaac Lab training environments** for RL tasks (reach, lift a cube, open a drawer).
- **Imitation learning**, **teleoperation interfaces**, and **sim-to-sim / sim-to-real transfer pipelines** are currently under development and will be available soon.

This repository has been tested with:
- **Ubuntu 22.04**
- **Isaac Sim v5.1.0**
- **Isaac Lab v2.3.0**
- **Python 3.11**

---

## Index
- [OpenArm Isaac Lab](#openarm-isaac-lab)
  - [Overview](#overview)
    - [What this repo offers](#what-this-repo-offers)
  - [Index](#index)
  - [Installation Guide](#installation-guide)
    - [(Option 1) Docker installation (linux only)](#option-1-docker-installation-linux-only)
    - [(Option 2) Local installation](#option-2-local-installation)
  - [Reinforcement Learning (RL)](#reinforcement-learning-rl)
    - [Training Model](#training-model)
    - [Replay Trained Model](#replay-trained-model)
    - [Analyze logs](#analyze-logs)
  - [Sim2sim](#sim2sim)
  - [Sim2Real Deployment using OpenArm](#sim2real-deployment-using-openarm)
  - [Related links](#related-links)
  - [License](#license)
  - [Code of Conduct](#code-of-conduct)

## Installation Guide

### (Option 1) Docker installation (linux only)

1. Pull the minimal Isaac Lab container
```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.0
```

1. Create container
```bash
xhost +
docker run --name isaac-lab --entrypoint bash -it --gpus all --rm -e "ACCEPT_EULA=Y" --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.3.0
```

1. Clone git at your HOME directory
```bash
cd /workspace
git clone git@github.com:enactic/openarm_isaac_lab.git
```

1. Install python package with
```bash
cd openarm_isaac_lab
python -m pip install -e source/openarm
```

1. As a test, you can visualize random movement of OpenArm with
```bash
python ./scripts/tools/list_envs.py
```

### (Option 2) Local installation

It is assumed that you have created a virtual environment named env_isaaclab using miniconda or anaconda and will be working within that environment.

1. Clone git at your HOME directory
```bash
cd ~
git clone git@github.com:enactic/openarm_isaac_lab.git
```

2. Activate your virtual env which contains Isaac Lab package
```bash
conda activate env_isaaclab
```

3. Install python package with
```bash
cd openarm_isaac_lab
python -m pip install -e source/openarm
```

4. As a test, you can visualize random movement of OpenArm with
```bash
python ./scripts/tools/list_envs.py
```

## Reinforcement Learning (RL)
You can run different tasks and policies.

Replace `<TASK_NAME>` and `<POLICY_NAME>` with one of the following available tasks:

| Task Description        | Task Name                      | Policy Name                         | Demo                                |
| ----------------------- | ------------------------------ | ----------------------------------- | ------------------------------------ |
| Reach target position   | `Isaac-Reach-OpenArm-v0`       | `rsl_rl`, `rl_games`, `skrl`        | <img src="video/openarm-reach-demo.gif" width="400"/>          |
| Lift a cube             | `Isaac-Lift-Cube-OpenArm-v0`   | `rsl_rl`, `rl_games`, `skrl` | <img src="video/openarm-lift-demo.gif" width="400"/>           |
| Open a cabinet's drawer | `Isaac-Open-Drawer-OpenArm-v0` | `rsl_rl`, `rl_games`, `skrl`        | <img src="video/openarm-drawer-demo.gif" width="400"/>         |
| Reach target position (Bimanual)  | `Isaac-Reach-OpenArm-Bi-v0`    | `rsl_rl`, `rl_games`, `skrl`        | <img src="video/openarm-bi-reach-demo.gif" width="400"/>          |

### Training Model

```bash
python ./scripts/reinforcement_learning/<POLICY_NAME>/train.py --task <TASK_NAME> --headless
```

### Replay Trained Model

```bash
python ./scripts/reinforcement_learning/<POLICY_NAME>/play.py --task <TASK_NAME> --num_envs 64
```

### Analyze logs

```bash
python -m tensorboard.main --logdir=logs
```

And open the google and go to `http://localhost:6006/`

## Sim2sim
<img src="video/sim2sim-reach-demo.gif" width="700"/>

Coming soon...

## Sim2Real Deployment using OpenArm

Coming soon...

## Related links

* Read the [documentation](https://docs.openarm.dev/)
* Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
* Contact us through <openarm@enactic.ai>

## License

[Apache License 2.0](LICENSE.txt)

Copyright 2025 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
