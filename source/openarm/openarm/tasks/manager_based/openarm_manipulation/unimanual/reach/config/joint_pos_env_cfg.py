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


import math

from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm
from .. import mdp
from ..reach_env_cfg import (
    ReachEnvCfg,
)

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_unimanual import (
    OPEN_ARM_CFG,
)
from isaaclab.assets.articulation import ArticulationCfg

##
# Environment configuration
##


@configclass
class OpenArmReachEnvCfg(ReachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to OpenArm
        self.scene.robot = OPEN_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_joint1": 1.57,
                    "openarm_joint2": 0.0,
                    "openarm_joint3": -1.57,
                    "openarm_joint4": 1.57,
                    "openarm_joint5": 0.0,
                    "openarm_joint6": 0.0,
                    "openarm_joint7": 0.0,
                    "openarm_finger_joint.*": 0.0,
                },  # Close the gripper
            ),
        )

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["openarm_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["openarm_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["openarm_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_joint.*",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "openarm_hand"


@configclass
class OpenArmReachEnvCfg_PLAY(OpenArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
