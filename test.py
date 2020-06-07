import numpy as np
from manipulator_2d import Manipulator2D

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

# Gym Environment 호출
env = Manipulator2D()

# Gym env의 action_space로부터 action의 개수를 알아낸다.
n_actions = env.action_space.shape[-1]
param_noise = None
# Exploration을 위한 Noise를 추가한다.
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# Stable Baseline이 제공하는 알고리즘 중 DDPG를 선택하고
model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

# 400,000 timestep 동안 시뮬레이션을 실행하며 학습한다.
model.learn(total_timesteps=400000)

# 학습된 결과를 저장한다.
# 연습 : 400,000 timestep 동안 학습한 결과가 아닌, 학습 도중 가장 좋은 reward 값을 반환한 policy network를 저장하려면 어떻게 해야할까요?
# Tip : learn 함수에 callback function을 사용해봅시다.
model.save("ddpg_manipulator2D")

# model 변수를 제거
del model  # remove to demonstrate saving and loading

# 저장된 학습 파일로부터 weight 등을 로드
model = DDPG.load("ddpg_manipulator2D")

# 시뮬레이션 환경을 초기화
obs = env.reset()

# Plot에 필요한 변수들 선언
states = []
rewards = []
timesteps = []

while True:
    # 학습된 모델로부터 observation 값을 넣어 policy network에서 action을 만들어냄
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    if done:
        break

env.render()