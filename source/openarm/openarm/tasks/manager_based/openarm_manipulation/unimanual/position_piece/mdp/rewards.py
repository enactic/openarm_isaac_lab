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
import torch.nn.functional as F 
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def frame_height_from_table(
    env: ManagerBasedRLEnv,
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("hand_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    frame: FrameTransformer = env.scene[frame_cfg.name]
    height = frame.data.target_pos_w[..., 0, :][:, 2] - robot.data.root_pos_w[:, 2] 

    # Softplus of -(height - threshold): 
    # - large when height << threshold
    # - ~0 when height >> threshold
    # - equals softplus(0)=ln(2)≈0.693 at height == threshold
    #slope = 0.60 + 2.5 / threshold
    #print("MEOWWWWWWWWWWWWWWWWWWWWWW")
    #print(height)
    #print(F.softplus(-(height - threshold) * slope - 2.5))
    #return F.softplus(-(height - threshold) * slope - 2.5)
    return (threshold - height).clamp(min=0)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    #minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    #return (object.data.root_pos_w[:, 2] > minimal_height) * (
    return    1 - torch.tanh(distance / std)
    #)


def object_goal_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    #minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal orientation using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired orientation in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    angle_error = quat_error_magnitude(des_quat_w, object.data.root_quat_w)

    return (distance < 0.05) * (
        1 - torch.tanh(angle_error / std)
    )

def debug(env: ManagerBasedRLEnv):
    sensor_cfg = SceneEntityCfg("table_contact_l7")
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # 3. Compute the force magnitude and find the maximum over history
    force_magnitude = net_forces.norm(dim=-1)  # (num_envs, history_length, num_bodies)
    max_force = force_magnitude.max(dim=1)[0]   # (num_envs, num_bodies)
    #print(max_force)
    return torch.zeros(1, device = max_force.device)