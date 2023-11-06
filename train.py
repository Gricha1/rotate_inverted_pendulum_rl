import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
from gym.envs.registration import make, register, registry, spec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

class Config:
  def __init__(self):
    pass

args = Config()
args.exp_name = "testing_exp_1"
args.seed = 1
args.torch_deterministic = True # ?
args.cuda = True
args.track = True # wandb
args.wandb_project_name = "test_ppo_pendulum"
args.wandb_entity = None
args.capture_video = False
args.save_model = True
args.upload_model = True

args.env_id = "custom_InvertedPendulum"
#args.env_id = "InvertedPendulum-v4" #"HalfCheetah" #"InvertedPendulum-v4" # "HalfCheetah-v4"
args.total_timesteps = 10_000_000 # 600_000
args.learning_rate = 3e-4
args.num_envs = 1
args.eval_freq = 100_000 # 100_000
args.num_steps = 8196
args.anneal_lr = True # True
args.gamma = 0.99
args.gae_lambda = 0.95 # ?
args.num_minibatches = 4
args.update_epochs = 10
args.norm_adv = True
args.clip_coef = 0.2
args.clip_vloss = True
args.ent_coef = 0 # 1e-4
args.vf_coef = 0.5
args.max_grad_norm = 6.0
args.target_kl = None
args.num_layers = 32 # 32

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

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


def record_video(env, policy, fps=30, mode="rgb_array`", max_steps=100, writer=None, global_step=0):
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


register(
    id="custom_InvertedPendulum",
    entry_point="envs.inverted_pendulum:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)


# init train env
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

agent = Agent(envs).to(device)

# init eval env
eval_env = make_val_env(args.env_id, args.seed, args.gamma, render_mode="single_rgb_array")
validate(eval_env=eval_env, model=agent, writer=None, global_step=1)


import os
os.environ["WANDB_API_KEY"] = "e02adc96c6e6d09cc6555b77b6eda5c038be07ca"

if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# init eval env
eval_env = make_val_env(args.env_id, args.seed, args.gamma, render_mode="single_rgb_array")

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)


# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

# debug
debug_reward = 0
debug_steps = 0

for update in range(1, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        #debug
        debug_reward += reward
        debug_steps += 1

        #for item in info:
        if "episode" in info.keys():
            print(f"global_step={global_step}, episodic_return={info['episode'][0]['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"][0]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"][0]["l"], global_step)
            writer.add_scalar("train/episodic_reward", debug_reward, global_step)
            writer.add_scalar("train/episodic_length", debug_steps, global_step)

            # debug
            debug_reward = 0
            debug_steps = 0

        if global_step % args.eval_freq == 0:
          with torch.no_grad():
            validate(eval_env=eval_env, model=agent, writer=writer, global_step=0, video_name="eval_trajectory")

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

# init eval env
#eval_env = gym.make(args.env_id, render_mode="single_rgb_array")
eval_env = make_val_env(args.env_id, args.seed, args.gamma, render_mode="single_rgb_array")
validate(eval_env=eval_env, model=agent, writer=writer, global_step=0, video_name="eval_trajectory")

envs.close()
writer.close()
wandb.finish()



