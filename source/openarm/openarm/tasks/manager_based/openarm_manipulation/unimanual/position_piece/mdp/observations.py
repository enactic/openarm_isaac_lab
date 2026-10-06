# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, quat_mul, sample_uniform, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b

def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The pose (position + quaternion) of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_quat_w  # (w, x, y, z)

    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object_pos_w,
        object_quat_w,
    )

    roll, pitch, yaw = euler_xyz_from_quat(object_quat_b)
    object_euler_b = torch.stack([roll, pitch, yaw], dim=-1)

    return torch.cat([object_pos_b, object_euler_b], dim=-1)

def reset_root_state_polar(env, env_ids, pose_range, asset_cfg):
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] += env.scene.env_origins[env_ids]

    # Default ranges when a key is missing from the dict
    default_pose_range = {k: (0.0, 0.0) for k in ["r", "psi", "z", "roll", "pitch", "yaw"]}

    # Merge user-provided ranges over the defaults
    pose_range = {**default_pose_range, **pose_range}

    # Build the (6, 2) tensor
    ranges = torch.tensor(
        [
            pose_range["r"], pose_range["psi"], pose_range["z"],
            pose_range["roll"], pose_range["pitch"], pose_range["yaw"],
        ],
        device=asset.device,
    )
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    pos_offset = torch.stack(
        [rand_samples[:, 0]*torch.cos(rand_samples[:, 1]), rand_samples[:, 0]*torch.sin(rand_samples[:, 1]), rand_samples[:, 2]],
        dim=1
        )
    positions = root_states[:, 0:3] + pos_offset
    
    orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)
    
    velocities = root_states[:, 7:13]

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)