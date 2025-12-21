# Quadcopter Environment

This environment implements a reinforcement learning task for quadcopter control using Isaac Lab. The quadcopter learns to navigate to random 3D goal positions while minimizing energy expenditure and maintaining stable flight.

## Environment Overview

The environment trains a Crazyflie quadcopter to hover and navigate to target positions in 3D space using direct thrust and moment control.

## File Structure

- `quadcopter_env.py` - Main environment implementation

## Classes

### 1. QuadcopterEnvWindow
UI window manager for visualization and debugging (lines 29-46).

### 2. QuadcopterEnvCfg
Environment configuration class containing all hyperparameters.

### 3. QuadcopterEnv
Main RL environment implementing the quadcopter control task.

---

## Environment Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episode length | 10.0 s | Maximum episode duration |
| Simulation timestep | 100 Hz | Physics simulation rate |
| Control frequency | 50 Hz | Agent action frequency (2x decimation) |
| Action space | 4D | Thrust + roll/pitch/yaw moments |
| Observation space | 12D | Body velocities, gravity, goal position |
| Parallel environments | Up to 4096 | For efficient parallel training |
| Environment spacing | 2.5 m | Distance between parallel environments |

### Robot Configuration

- **Model**: Crazyflie quadcopter
- **Thrust-to-weight ratio**: 1.9 (can hover and accelerate)
- **Moment scale**: 0.01 (for attitude control)

### Reward Scales

| Component | Scale | Purpose |
|-----------|-------|---------|
| Linear velocity | -0.05 | Penalize fast movements |
| Angular velocity | -0.01 | Penalize rotation |
| Distance-to-goal | 15.0 | Primary objective |

---

## Observation Space (12D)

The agent observes the following in the robot's body frame:

1. **Root linear velocity** (3D) - Body-frame linear velocity
2. **Root angular velocity** (3D) - Body-frame angular velocity
3. **Projected gravity** (3D) - Gravity vector in body frame
4. **Desired position** (3D) - Goal position relative to robot

**Implementation**: `_get_observations()` (lines 158-172)

---

## Action Space (4D)

Actions are continuous values in [-1, 1]:

- **action[0]**: Vertical thrust
  - Mapped to [0, 2 × thrust_to_weight × robot_weight]
  - Value of 0 gives hovering thrust

- **action[1:4]**: Roll, pitch, yaw moments
  - Scaled by `moment_scale` (0.01)
  - Controls rotational dynamics

**Implementation**: `_pre_physics_step()` (lines 150-153)

---

## Reward Function

The reward function encourages the quadcopter to reach the goal while minimizing unnecessary motion:

```
reward = -0.05 * ||linear_vel||² - 0.01 * ||angular_vel||² + 15.0 * (1 - tanh(distance/0.8))
```

### Reward Components

1. **Linear velocity penalty**: Discourages fast, jerky movements
2. **Angular velocity penalty**: Encourages stable orientation
3. **Distance-to-goal reward**: Primary objective using tanh shaping for smooth gradients
   - Maximum reward when at goal position
   - Smoothly decreases with distance (tanh provides smooth gradients)

**Implementation**: `_get_rewards()` (lines 174-188)

---

## Termination Conditions

### Episode Termination

Episodes terminate when either condition is met:

1. **Died** (line 192):
   - Robot altitude < 0.1 m (crashed into ground)
   - Robot altitude > 2.0 m (flew too high)

2. **Timeout** (line 191):
   - Episode reaches maximum length (10 seconds)

**Implementation**: `_get_dones()` (lines 190-193)

---

## Reset Behavior

When environments are reset:

1. **Goal randomization** (lines 224-226):
   - XY position: Uniform in [-2, 2] m
   - Z position: Uniform in [0.5, 1.5] m
   - Centered around terrain origin

2. **Robot state** (lines 228-234):
   - Position: Default pose at terrain origin
   - Velocity: Zero
   - Joint states: Default configuration

3. **Episode staggering** (line 220):
   - Initial episodes have random lengths to avoid synchronous resets
   - Prevents training spikes when many environments reset simultaneously

**Implementation**: `_reset_idx()` (lines 195-234)

---

## Visualization

### Debug Visualization

When `debug_vis=True`:
- Small cuboid markers (0.05 m) visualize goal positions
- Markers update in real-time as goals change
- Can be toggled via UI window

**Implementation**:
- `_set_debug_vis_impl()` (lines 236-249)
- `_debug_vis_callback()` (lines 251-253)

---

## Training Task

The quadcopter learns to:

- **Navigate to random 3D goal positions** within a bounded volume
- **Minimize energy expenditure** through velocity penalties
- **Stay within altitude bounds** [0.1 m, 2.0 m]
- **Hover smoothly** at goal positions without excessive oscillation

## Design Patterns

1. **Direct control**: Applies thrust/torque directly to quadcopter body (no motor-level control)
2. **Parallel simulation**: Supports thousands of environments for sample-efficient training
3. **Body-frame observations**: All observations relative to robot frame for learning stability
4. **Shaped rewards**: Uses tanh for smooth, continuous reward gradients
5. **Episode logging**: Tracks cumulative rewards and termination statistics for monitoring

---

## Training Recommendations

### Suitable Algorithms

This continuous control task works well with:
- **PPO** (Proximal Policy Optimization) - Recommended for stability
- **SAC** (Soft Actor-Critic) - Good for sample efficiency
- **TD3** (Twin Delayed DDPG) - Alternative off-policy approach

### Key Considerations

- High parallel environment count (1024-4096) recommended for PPO
- Body-frame observations provide translation invariance
- Reward scaling balances goal-reaching vs. smooth control
- Episode staggering important for training stability

---

## Code Reference

Main implementation: `quadcopter_env.py`

Key methods:
- `_get_observations()` (lines 158-172) - Observation construction
- `_pre_physics_step()` (lines 150-153) - Action processing
- `_apply_action()` (lines 155-156) - Force/torque application
- `_get_rewards()` (lines 174-188) - Reward calculation
- `_get_dones()` (lines 190-193) - Termination logic
- `_reset_idx()` (lines 195-234) - Environment reset

---

## License

Copyright (c) 2022-2025, The Isaac Lab Project Developers
SPDX-License-Identifier: BSD-3-Clause
