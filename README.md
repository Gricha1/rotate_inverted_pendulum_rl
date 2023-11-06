# rotate_inverted_pendulum_rl
This repository is dedicated to rotary inverted pendulum task with RL approuch. The Proximal Policy Optimization(PPO) were used to solve the task.
PPO implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

# Installation
Python 3.10.12

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

# Train
to start train use
```commandline
python train_ppo.py
```

# Validate
to start validation use
```commandline
python validate.py
```




