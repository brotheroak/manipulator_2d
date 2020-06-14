import gym
import os
import numpy as np

# 예시 강화학습 알고리즘
from stable_baselines import DQN, DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

# 2D Arm 환경
from manipulator_2d import Manipulator2D

# Create environment
env = Manipulator2D()
param_noise =None
# Load the trained agent
# 예시로 DQN을 사용하였으므로, 실제 학습에 사용한 강화학습 알고리즘을 이용.
# 상대경로로 학습된 weight load
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

eval_callback = EvalCallback(env, best_model_save_path="./models",
                             log_path='./logs', eval_freq=500,
                             deterministic=True, render=False)

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

model.learn(total_timesteps=400000, callback=eval_callback)

model.save("best_reward")

model = DDPG.load(os.path.join("models", "best_reward"))

# Competition Code
# 이 아래 부분은 일괄적으로 덮어씌워질 예정.
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

env.render()