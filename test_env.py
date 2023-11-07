import gym
from gym.envs.registration import make, register, registry, spec


register(
    id="custom_InvertedPendulum",
    entry_point="envs.inverted_pendulum:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

env = gym.make("custom_InvertedPendulum", render_mode="single_rgb_array")

# play episode
env.reset()
done = False
cumul_reward = 0
print("********************")
print("sum image:", env.custom_render().sum())
#print("sum image:", env.render().sum())
print("********************")
while not done:
    _, r, done, _ = env.step(env.action_space.sample())
    cumul_reward += r
    env.custom_render()
    #env.render()
print("cumul reward:", cumul_reward)

