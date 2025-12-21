# Quadcopter Motors Environment

This environment implements a reinforcement learning task for quadcopter control using Isaac Lab with **motor-level control**. The quadcopter learns to navigate to random 3D goal positions while minimizing energy expenditure and maintaining stable flight by commanding individual motor angular velocities.

## Environment Overview

The environment trains a Crazyflie quadcopter to hover and navigate to target positions in 3D space using **low-level motor control**. Unlike direct thrust/torque control, this environment simulates:
- Individual motor dynamics with first-order lag
- Quadratic propeller thrust and torque models
- Force and moment mixing from four rotors
- Realistic motor saturation and time constants

## Registration

- **Environment ID**: `IDSIA-Quadcopter-Motors-Direct-v0`

## File Structure

- `quadcopter_motors_env.py` - Main environment implementation

## Classes

### 1. QuadcopterMotorsEnvWindow
UI window manager for visualization and debugging ([quadcopter_motors_env.py:30-48](quadcopter_motors_env.py#L30-L48)).

### 2. QuadcopterMotorsEnvCfg
Environment configuration class containing all hyperparameters ([quadcopter_motors_env.py:50-107](quadcopter_motors_env.py#L50-L107)).

### 3. QuadcopterMotorsEnv
Main RL environment implementing the quadcopter control task with motor-level physics simulation ([quadcopter_motors_env.py:109-365](quadcopter_motors_env.py#L109-L365)).

---

## Environment Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episode length | 10.0 s | Maximum episode duration |
| Simulation timestep | 100 Hz | Physics simulation rate |
| Control frequency | 50 Hz | Agent action frequency (2x decimation) |
| Action space | 4D | Individual motor angular velocity commands |
| Observation space | 12D | Body velocities, gravity, goal position |
| Parallel environments | Up to 4096 | For efficient parallel training |
| Environment spacing | 2.5 m | Distance between parallel environments |

### Robot Configuration

- **Model**: Crazyflie quadcopter
- **Arm length**: 0.05 m (distance from center to motor)
- **Thrust-to-weight ratio**: 1.9 (can hover and accelerate)
- **Max motor speed**: 10,000 RPM (1047 rad/s)
- **Motor time constant**: 0.05 s (first-order lag)
- **Motor configuration**:
  - M1 (front-right): Counter-clockwise
  - M2 (rear-right): Clockwise
  - M3 (rear-left): Counter-clockwise
  - M4 (front-left): Clockwise

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

**Implementation**: `_get_observations()` ([quadcopter_motors_env.py:263-277](quadcopter_motors_env.py#L263-L277))

---

## Action Space (4D)

Actions are continuous values in [-1, 1] representing desired motor angular velocities:

- **action[0]**: Motor M1 (front-right, CCW)
- **action[1]**: Motor M2 (rear-right, CW)
- **action[2]**: Motor M3 (rear-left, CCW)
- **action[3]**: Motor M4 (front-left, CW)

### Action Processing Pipeline

1. **Action mapping** ([quadcopter_motors_env.py:248-250](quadcopter_motors_env.py#L248-L250)):
   - Actions clamped to [-1, 1]
   - Mapped to [0, max_motor_rads] = [0, 1047 rad/s]
   - Formula: `motor_rads_des = max_motor_rads * (action + 1) / 2`

2. **Motor dynamics** ([quadcopter_motors_env.py:185-194](quadcopter_motors_env.py#L185-L194)):
   - First-order lag: `d(ω)/dt = (ω_des - ω) / τ`
   - Time constant τ = 0.05 s
   - Simulates realistic motor response

3. **Propeller model** ([quadcopter_motors_env.py:196-215](quadcopter_motors_env.py#L196-L215)):
   - Thrust: `F = k_thrust * ω²`
   - Torque: `τ = k_torque * ω²` (with direction based on motor rotation)
   - Quadratic relationship between motor speed and forces

4. **Force mixing** ([quadcopter_motors_env.py:217-245](quadcopter_motors_env.py#L217-L245)):
   - Aggregates individual motor forces into net body thrust
   - Computes moments from lever arms: `M = r × F`
   - Adds propeller reaction torques for yaw control

**Implementation**: `_pre_physics_step()` ([quadcopter_motors_env.py:247-257](quadcopter_motors_env.py#L247-L257))

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

**Implementation**: `_get_rewards()` ([quadcopter_motors_env.py:279-293](quadcopter_motors_env.py#L279-L293))

---

## Termination Conditions

### Episode Termination

Episodes terminate when either condition is met:

1. **Died** ([quadcopter_motors_env.py:297](quadcopter_motors_env.py#L297)):
   - Robot altitude < 0.1 m (crashed into ground)
   - Robot altitude > 2.0 m (flew too high)

2. **Timeout** ([quadcopter_motors_env.py:296](quadcopter_motors_env.py#L296)):
   - Episode reaches maximum length (10 seconds)

**Implementation**: `_get_dones()` ([quadcopter_motors_env.py:295-298](quadcopter_motors_env.py#L295-L298))

---

## Reset Behavior

When environments are reset:

1. **Goal randomization** ([quadcopter_motors_env.py:330-332](quadcopter_motors_env.py#L330-L332)):
   - XY position: Uniform in [-2, 2] m
   - Z position: Uniform in [0.5, 1.5] m
   - Centered around terrain origin

2. **Robot state** ([quadcopter_motors_env.py:335-341](quadcopter_motors_env.py#L335-L341)):
   - Position: Default pose at terrain origin
   - Velocity: Zero
   - Joint states: Default configuration

3. **Motor initialization** ([quadcopter_motors_env.py:343-344](quadcopter_motors_env.py#L343-L344)):
   - Motor angular velocities set to hover speed
   - Computed as: `ω_hover = sqrt(hover_thrust / k_thrust)`
   - Ensures stable initialization

4. **Episode staggering** ([quadcopter_motors_env.py:323-325](quadcopter_motors_env.py#L323-L325)):
   - Initial episodes have random lengths to avoid synchronous resets
   - Prevents training spikes when many environments reset simultaneously

**Implementation**: `_reset_idx()` ([quadcopter_motors_env.py:300-344](quadcopter_motors_env.py#L300-L344))

---

## Visualization

### Debug Visualization

When `debug_vis=True`:
- Small cuboid markers (0.05 m) visualize goal positions
- Markers update in real-time as goals change
- Can be toggled via UI window

**Implementation**:
- `_set_debug_vis_impl()` ([quadcopter_motors_env.py:346-360](quadcopter_motors_env.py#L346-L360))
- `_debug_vis_callback()` ([quadcopter_motors_env.py:362-364](quadcopter_motors_env.py#L362-L364))

---

## Training Task

The quadcopter learns to:

- **Navigate to random 3D goal positions** within a bounded volume
- **Minimize energy expenditure** through velocity penalties
- **Stay within altitude bounds** [0.1 m, 2.0 m]
- **Hover smoothly** at goal positions without excessive oscillation

## Design Patterns

1. **Motor-level control**: Commands individual motor angular velocities (not direct thrust/torque)
2. **Physics-based modeling**: Includes motor dynamics, propeller models, and force mixing
3. **Parallel simulation**: Supports thousands of environments for sample-efficient training
4. **Body-frame observations**: All observations relative to robot frame for learning stability
5. **Shaped rewards**: Uses tanh for smooth, continuous reward gradients
6. **Episode logging**: Tracks cumulative rewards and termination statistics for monitoring

## Motor Physics Pipeline

The environment implements a realistic quadcopter physics pipeline:

```
Action [-1,1] → Motor Command [0, max_ω]
              ↓
Motor Dynamics (1st order lag, τ=0.05s)
              ↓
Motor State ω [rad/s]
              ↓
Propeller Model (F = k*ω², τ = k'*ω²)
              ↓
Individual Forces & Torques (4 motors)
              ↓
Force Mixing (sum forces, compute moments)
              ↓
Body Wrench (net thrust & moment)
              ↓
Applied to Rigid Body
```

This provides more realistic dynamics compared to direct thrust/torque control, requiring the agent to learn motor coordination and compensate for motor lag.

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
- Motor-level control increases task difficulty compared to direct thrust/torque
- Agent must learn to coordinate four motors and compensate for motor lag
- Consider curriculum learning: start with closer goals, gradually increase range

---

## Code Reference

Main implementation: [quadcopter_motors_env.py](quadcopter_motors_env.py)

### Key Methods

**Core RL Interface:**
- `_get_observations()` ([quadcopter_motors_env.py:263-277](quadcopter_motors_env.py#L263-L277)) - Observation construction
- `_pre_physics_step()` ([quadcopter_motors_env.py:247-257](quadcopter_motors_env.py#L247-L257)) - Action processing
- `_apply_action()` ([quadcopter_motors_env.py:259-261](quadcopter_motors_env.py#L259-L261)) - Force/torque application
- `_get_rewards()` ([quadcopter_motors_env.py:279-293](quadcopter_motors_env.py#L279-L293)) - Reward calculation
- `_get_dones()` ([quadcopter_motors_env.py:295-298](quadcopter_motors_env.py#L295-L298)) - Termination logic
- `_reset_idx()` ([quadcopter_motors_env.py:300-344](quadcopter_motors_env.py#L300-L344)) - Environment reset

**Physics Models:**
- `motor_model()` ([quadcopter_motors_env.py:185-194](quadcopter_motors_env.py#L185-L194)) - First-order motor dynamics
- `propeller_model()` ([quadcopter_motors_env.py:196-215](quadcopter_motors_env.py#L196-L215)) - Quadratic thrust/torque model
- `mix_propellers()` ([quadcopter_motors_env.py:217-245](quadcopter_motors_env.py#L217-L245)) - Force and moment aggregation

**Visualization:**
- `_set_debug_vis_impl()` ([quadcopter_motors_env.py:346-360](quadcopter_motors_env.py#L346-L360)) - Debug visualization setup
- `_debug_vis_callback()` ([quadcopter_motors_env.py:362-364](quadcopter_motors_env.py#L362-L364)) - Marker updates

---

## License

Copyright (c) 2022-2025, The Isaac Lab Project Developers
SPDX-License-Identifier: BSD-3-Clause
