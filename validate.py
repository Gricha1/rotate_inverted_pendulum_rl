import time

import torch
import numpy as np
import gym
from gym.envs.registration import make, register, registry, spec

from utils import record_video
from utils import make_env, make_val_env
from models import Agent

def validate(eval_env=None, model=None, video_fps=30, writer=None, global_step=0, video_name='eval_trajectory', max_steps=200):
  # Generate a video
  video, info = record_video(eval_env, model, max_steps=max_steps, writer=writer, global_step=global_step)
  video = np.array(video)
  video = np.expand_dims(video, axis=0)
  video = np.transpose(video, (0, 1, 4, 2, 3))
  video = (video * 1).astype('uint8')
  print("video shape:", video.shape)
  # log data
  if writer is None:
    pass
  else:
    writer.add_video(
        tag=video_name,
        vid_tensor=torch.from_numpy(video),
        global_step=global_step,
        fps=4
    )
    writer.add_scalar("eval/episodic_return", info["r"], global_step)
    writer.add_scalar("eval/episodic_length", info["l"], global_step)
    print("add video to tensorboard")
  del video
  del info


if __name__ == "__main__":
    class Config:
        def __init__(self):
            pass

    args = Config()
    args.env_id = "custom_InvertedPendulum"
    args.cuda = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    register(
        id="custom_InvertedPendulum",
        entry_point="envs.inverted_pendulum:InvertedPendulumEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )

    # init train env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 1, 1, False, "test", 0.99) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)

    # init eval env
    eval_env = make_val_env(args.env_id, 1, 0.99, render_mode="single_rgb_array")
    validate(eval_env=eval_env, model=agent, writer=None, global_step=1)