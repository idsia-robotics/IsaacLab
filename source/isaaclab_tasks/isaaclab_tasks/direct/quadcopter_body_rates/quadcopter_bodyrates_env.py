# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG  # isort: skip


class QuadcopterBodyRatesEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter BodyRates environment."""

    def __init__(self, env: QuadcopterBodyRatesEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterBodyRatesEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0

    debug_vis = True

    ui_window_class_type = QuadcopterBodyRatesEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # robot parameters
    arm_length = 0.05           # [m]
    thrust_to_weight = 1.9      # [-]
    max_body_rate = 4.0         # [rad/s]
    max_motor_rads = 10_000 * (2 * math.pi / 60)          # [rad/s]
    motor_time_constant = 0.05  # [s]

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterBodyRatesEnv(DirectRLEnv):
    cfg: QuadcopterBodyRatesEnvCfg

    def __init__(self, cfg: QuadcopterBodyRatesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Drone state
        self._motor_rads = torch.zeros(self.num_envs, 4, device=self.device)

        # Total thrust and moment applied to the base of the quadcopter
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # Drone parameters
        L = self.cfg.arm_length / math.sqrt(2.0)

        self._propeller_pos_b = torch.tensor([
            [ L, -L, 0.0],  # M1 front-right
            [-L, -L, 0.0],  # M2 rear-right
            [-L, +L, 0.0],  # M3 rear-left
            [+L, +L, 0.0],  # M4 front-left
        ], device=self.device, dtype=torch.float32)
        
        self._propeller_directions = torch.tensor([
            +1.0,   # M1 CCW
            -1.0,   # M2 CW
            +1.0,   # M3 CCW
            -1.0    # M4 CW
        ], device=self.device, dtype=torch.float32)

        self._max_thrust = self.cfg.thrust_to_weight * self._robot_weight
        propeller_max_thrust = self._max_thrust / 4.0
        self._propeller_thrust_coeff = propeller_max_thrust / (self.cfg.max_motor_rads ** 2)
        self._propeller_torque_coeff = 0.02 * self._propeller_thrust_coeff  # FIXME: torque coefficient

        hover_propeller_thrust = self._robot_weight / 4.0
        self._hover_motor_rads = math.sqrt(hover_propeller_thrust / self._propeller_thrust_coeff)

        # Controller parameters
        self._rate_kp = torch.tensor(
            [0.02, 0.02, 0.002],
            device=self.device,
        )

        self._rate_kd = torch.tensor(
            [0.0, 0.0, 0.0],
            device=self.device,
        )

        self._prev_body_rate = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    ## Drone controller

    def rate_controller(self, body_rate_des: torch.Tensor) -> torch.Tensor:
        """
        Body-rate PID controller.

        Args:
            body_rate_des: (N,3) desired body angular velocity [rad/s]

        Returns:
            moment_des: (N,3) desired body moment [Nm]
        """
        # Measured body rates
        body_rate = self._robot.data.root_ang_vel_b  # (N,3)

        # Rate error
        rate_error = body_rate_des - body_rate

        # Proportional term
        moment_p = self._rate_kp * rate_error

        # Derivative term
        domega = (body_rate - self._prev_body_rate) / self.step_dt
        moment_d = -self._rate_kd * domega

        self._prev_body_rate = body_rate.clone()

        moment_des = moment_p + moment_d
        return moment_des

    def inverse_mix_propellers(
        self,
        thrust_des: torch.Tensor,   # (N,)
        moment_des: torch.Tensor,   # (N,3)
    ) -> torch.Tensor:
        """
        Map desired body wrench → per-propeller thrusts.

        Returns:
            motor_thrusts: (N,4)
        """
        L = self.cfg.arm_length / math.sqrt(2.0)
        kM = self._propeller_torque_coeff / self._propeller_thrust_coeff

        T  = thrust_des
        tx = moment_des[:, 0]
        ty = moment_des[:, 1]
        tz = moment_des[:, 2]

        f1 = 0.25 * (T - tx/L - ty/L + tz/kM)
        f2 = 0.25 * (T - tx/L + ty/L - tz/kM)
        f3 = 0.25 * (T + tx/L + ty/L + tz/kM)
        f4 = 0.25 * (T + tx/L - ty/L - tz/kM)

        motor_thrusts = torch.stack([f1, f2, f3, f4], dim=-1)

        return motor_thrusts

    def inverse_propeller_model(self, motor_thrusts: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the propeller model.

        Args:
            motor_thrusts: (N,4) desired thrust per propeller [N]

        Returns:
            motor_rads_des: (N,4) desired motor speeds [rad/s]
        """
        # Clamp negative thrusts to zero, critical to avoid NaNs
        motor_thrusts = motor_thrusts.clamp(min=0.0)

        motor_rads_des = torch.sqrt(motor_thrusts / self._propeller_thrust_coeff)
        motor_rads_des.clamp_(0.0, self.cfg.max_motor_rads)

        return motor_rads_des

    ## Drone dynamics

    def motor_model(self, motor_rads_des: torch.Tensor) -> torch.Tensor:
        """First-order motor model: d(omega)/dt = (omega_des - omega) / tau"""
        dt = self.step_dt
        tau = self.cfg.motor_time_constant
        alpha = dt / (tau + dt)

        self._motor_rads = (1.0 - alpha) * self._motor_rads + alpha * motor_rads_des
        self._motor_rads.clamp_(0.0, self.cfg.max_motor_rads)

        return self._motor_rads

    def propeller_model(self, motor_rads: torch.Tensor) -> torch.Tensor:
        """Simple quadratic propeller model: thrust = k * omega^2

        Args:
            motor_rads: (N,4) motor angular velocities in radians per second

        Returns:
            prop_forces: (N,4,3) motor thrusts in Newtons
            prop_moments: (N,4,3) motor reaction torques in Newton-meters
        """
        thrust = self._propeller_thrust_coeff * torch.square(motor_rads)
        torque = self._propeller_torque_coeff * torch.square(motor_rads)

        prop_forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        prop_forces[..., 2] = thrust  # Thrust (Z axis)

        prop_moments = torch.zeros(self.num_envs, 4, 3, device=self.device) 
        prop_moments[..., 2] = torque # Yaw torque (yaw)

        return prop_forces, prop_moments

    def mix_propellers(
        self,
        prop_forces: torch.Tensor,   # (N,4,3)
        prop_moments: torch.Tensor,  # (N,4,3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate propeller forces and moments into body wrench.

        Args:
            prop_forces:  (N,4,3) force vectors in body frame
            prop_moments: (N,4,3) reaction moments in body frame

        Returns:
            thrust: (N,3) net force at CoM (body frame)
            moment: (N,3) net moment about CoM (body frame)
        """
        # Net thrust
        thrust = prop_forces.sum(dim=1)  # (N,3)

        # Lever-arm moments: r_i × f_i
        r = self._propeller_pos_b.unsqueeze(0)               # (1,4,3)
        lever_moments = torch.cross(r, prop_forces, dim=-1)  # (N,4,3)

        d = self._propeller_directions[None, :, None]  # (1,4,1)
        prop_moments = d * prop_moments                # (N,4,3)

        # Net moment
        moment = lever_moments.sum(dim=1) + prop_moments.sum(dim=1)

        return thrust, moment

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        thrust_des = self._max_thrust * (self._actions[:, 0] + 1.0) / 2.0
        body_rate_des = self.cfg.max_body_rate * self._actions[:, 1:4]

        # Drone controller
        moment_des = self.rate_controller(body_rate_des)
        motor_thrusts_des = self.inverse_mix_propellers(thrust_des, moment_des)
        motor_rads_des = self.inverse_propeller_model(motor_thrusts_des)

        # Drone dynamics
        motor_rads = self.motor_model(motor_rads_des)
        prop_forces, prop_moments = self.propeller_model(motor_rads)
        thrust, moment = self.mix_propellers(prop_forces, prop_moments)
        
        self._thrust[:, 0, :] = thrust
        self._moment[:, 0, :] = moment

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        # self._robot.write_joint_velocity_to_sim(self._motor_rads) # FIXME 

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._motor_rads[env_ids] = self._hover_motor_rads
        # self._robot.write_joint_velocity_to_sim(self._motor_rads[env_ids], env_ids) # FIXME

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
