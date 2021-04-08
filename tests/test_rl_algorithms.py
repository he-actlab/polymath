import pytest
# import gym
# from tests.reference_implementations.rl_algorithms import create_actor_critic, TORCH_NP_FN_MAP,\
#     lognormalize, probs_to_logits, logits_to_probs
# from tests.reference_implementations.ppo import MLPActorCritic
import torch
import numpy as np
from torch import nn
import random

RANDOM_SEED = 0

# @pytest.mark.parametrize('env_name, hidden_sizes, activation', [
#     ('CartPole-v0', (64, 64), nn.ReLU)
# ])
# def test_ppo1(env_name, hidden_sizes, activation):
#     env = gym.make(env_name)
#     env.seed(RANDOM_SEED)
#
#     ac_ref = MLPActorCritic(env.observation_space, env.action_space,
#                         hidden_sizes=hidden_sizes, activation=activation)
#     np_act = TORCH_NP_FN_MAP[activation.__name__]
#     ac_np = create_actor_critic(env.observation_space, env.action_space,
#                                 hidden_sizes, np_act, ac_ref, env)
#     # torch.manual_seed(RANDOM_SEED)
#     # np.random.seed(RANDOM_SEED)
#     # random.seed(RANDOM_SEED)
#     # env.seed(RANDOM_SEED)
#     #
#     # o, ep_ret, ep_len = env.reset(), 0, 0
#     # o = np.float32(o)
#     # # [-0.08411545]
#     # # tensor(-0.7085)
#     # torch.set_deterministic(True)
#     # with torch.no_grad():
#     #
#     #     o_torch = torch.as_tensor(o, dtype=torch.float32)
#     #     a, v, logp = ac_ref.step(o_torch)
#     #
#     #     a_np, logp_np, v_np = ac_np.step(o)
#
#         # Numpy implementation









