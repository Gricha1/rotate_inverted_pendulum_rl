# rotate_inverted_pendulum_rl
This repository is dedicated to rotary inverted pendulum task with RL approuch. The task is swing-up the pendulum and stabilize it. The Proximal Policy Optimization(PPO) were used to solve the problem.
PPO implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py


# Installation
1. Check versions
Python 3.10.12
conda 4.13.0
2. Install conda virtual environment
```commandline
conda env create --file=environment.yml
```

# Environment
1. Action Space: 
    represents the numerical force applied to the cart
```commandline
Box(-3.0, 3.0, (1,), float32)
```
2. Observation Space: 
    position of the cart along the linear surface(m), 
    vertical angle of the pole on the cart(rad),
    linear velocity of the cart(m/s),
    angular velocity of the pole on the cart(rad/s).
```commandline
Box(-inf, inf, (4,), float64)
```
3. Reward function:
```commandline
reward = r_stable + r_swingup_1 + r_swingup_2 + r_collision + r_xchange
```
where 
```commandline
r_stable = 10 if abs(theta) < 0.2
```
```commandline
r_swingup_1 = 5 * (1 - abs(theta)) if abs(theta) < pi
```
```commandline
r_swingup_2 = 1 * (1 - abs(theta)) 
```
```commandline
r_collision = -200
```
```commandline
r_xchange = -abs(x)
```
4. Termination: env terminates 
```commandline
if abs(x) > 0.8 or steps >= 1000
```

# Train
to start train use
```commandline
python train_ppo.py
```


# Validate
to start validation use
```commandline
python validate_ppo.py
```

# Results



