import gym
import numpy as np
import torch


def record_video(env, policy, fps=30, mode="rgb_array`", max_steps=100, writer=None, global_step=0, device="cpu"):
  images = []
  val_info = {}
  val_info["r"] = 0
  done = False
  state = env.reset()
  img = env.render()
  images.append(img)
  steps = 0
  while not done:
    state = torch.Tensor(state).to(device)
    state = state[None, :]
    if not(writer is None):
      writer.add_scalar("eval" + f"_{global_step}" + "/angles", state[0][1].item(), steps)
      writer.add_scalar("eval" + f"_{global_step}" + "/x", state[0][0].item(), steps)
    action, _, _, _  = policy.get_action_and_value(state)
    action = action[0, :]
    if not(writer is None):
      writer.add_scalar("eval" + f"_{global_step}" + "/action", action.item(), steps)
    state, reward, done, info = env.step(action.cpu().numpy()) # We directly put next_state = state for recording logic
    steps += 1
    val_info["r"] += reward
    if steps <= max_steps:
      img = env.render()
      images.append(img)
  print("val steps:", steps)
  val_info["l"] = steps
  return [np.array(img) for i, img in enumerate(images)], val_info


def make_env(env_id, seed, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_val_env(env_id, seed, gamma, render_mode="single_rgb_array"):
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    #env = gym.wrappers.NormalizeObservation(env)
    #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env