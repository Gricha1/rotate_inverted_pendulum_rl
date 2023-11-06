import gym
from gym.envs.registration import make, register, registry, spec

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if "custom_InvertedPendulum" in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
register(
    id="custom_InvertedPendulum",
    entry_point="envs.inverted_pendulum:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

#env = gym.make("custom_InvertedPendulum")

#if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
#    bash ../xvfb start
#    os.environ['DISPLAY'] = ':1'
    
env = gym.make("custom_InvertedPendulum", render_mode="single_rgb_array")
#env = gym.make("custom_InvertedPendulum", render_mode="rgb_array")
# play episode
env.reset()
done = False
cumul_reward = 0
while not done:
    _, r, done, _ = env.step(env.action_space.sample())
    cumul_reward += r
    #env.custom_render()
    env.render()
print("cumul reward:", cumul_reward)

