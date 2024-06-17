from env import SumoPettingZooEnv
import gymnasium as gym


env = SumoPettingZooEnv()
env.reset()
for _ in range(50):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()
