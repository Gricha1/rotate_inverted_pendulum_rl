# rotate_inverted_pendulum_rl

# Installation


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

# Train
to start train use
```commandline
python train_ppo.py
```

# Validate
to start train use
```commandline
python validate.py
```

## to delete
conda create -n inverted_pendulum python=3.10.12
conda activate inverted_pendulum
#conda install -c conda-forge gym==0.25.2
#conda install -c conda-forge mujoco
#conda install -c conda-forge wandb
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install gym==0.25.2
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install mujoco
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install imageio
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install wandb
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install torch
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install tensorboard
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install free-mujoco-py
/home/reedgern/anaconda3/envs/inverted_pendulum/bin/pip install moviepy

conda remove -n inverted_pendulum --all

# troubleshooting
1. mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called
```commandline
export MUJOCO_GL=osmesa
```



