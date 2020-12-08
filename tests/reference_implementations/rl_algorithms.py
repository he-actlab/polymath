import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1
from scipy.special import softmax
from gym.spaces import Box, Discrete
from typing import NamedTuple, Callable, List, Union
from stable_baselines3 import PPO
from torch import nn
from tests.util import np_relu


import numpy as np
import cv2

class Mlp(NamedTuple):
    layer_weights: List
    layer_biases: List
    layer_fns: List[Callable]
    activations: List[Callable]

class Actor(NamedTuple):
    sample: Callable
    log_prob_from_distribution: Callable
    actor_mlp: Mlp
    step: Callable

class Critic(NamedTuple):
    value_fn: Callable
    critic_mlp: Mlp

class ActorCritic(NamedTuple):
    actor: Actor
    critic: Callable
    step: Callable

def run_mlp(x, mlp: Mlp):
    for i in range(len(mlp.layer_fns)):
        w = mlp.layer_weights[i]
        b = mlp.layer_biases[i]
        act = mlp.activations[i]
        fn = mlp.layer_fns[i]
        x = act(fn(x, w, b))
    return x

def np_identity(x):
    return x

def np_linear(x, weight, bias=None):
    if bias is not None:
        return x.dot(weight.T) + bias
    else:
        return x.dot(weight.T)

def create_mlp(sizes, activation, output_activation=np_identity, pytorch_layers: Union[None, nn.Sequential]=None):
    layer_weights = []
    layer_fns = []
    layer_biases = []
    activations = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation

        if pytorch_layers:
            lidx = j*2
            lw = pytorch_layers[lidx].weight.detach().numpy()
            if hasattr(pytorch_layers[lidx], 'bias'):
                lb = pytorch_layers[lidx].bias.detach().numpy()
            else:
                lb = None
        else:
            torch_layer = nn.Linear(sizes[j], sizes[j+1])
            lw = torch_layer.weight.detach().numpy()
            lb = torch_layer.bias.detach().numpy()
        layer_weights.append(lw)
        layer_biases.append(lb)
        activations.append(act)
        layer_fns.append(np_linear)
    mlp = Mlp(layer_weights, layer_biases, layer_fns, activations)
    return mlp

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

def probs_to_logits(probs):
    return np.log(probs)

def logits_to_probs(logits):
    return softmax(logits)

def np_categorical_sample(obs, mlp_impl: Mlp):
    unnorm_probs = run_mlp(obs, mlp_impl)
    probs = lognormalize(unnorm_probs)
    return np.random.choice(probs.shape[-1], size=1, replace=True, p=probs)

def np_log_prob(a, log_probs):
    mg = np.meshgrid(
            *(range(i) for i in log_probs.shape[:-1]),
            indexing='ij')
    return log_probs[mg + [a.astype(np.int32)]]

def np_categorical_step(obs, mlp_impl):
    unnorm_probs = run_mlp(obs, mlp_impl)
    probs = lognormalize(unnorm_probs)
    log_probs = probs_to_logits(probs)
    sample = np.random.choice(probs.shape[-1], size=1, replace=True, p=probs)
    return sample, np_log_prob(sample, log_probs)

def mlp_categorical_dist(obs_dim, act_dim, hidden_sizes, activation, pytorch_mod=None, env_sample=None):
    pytorch_actor_layers = pytorch_mod.pi.logits_net
    pytorch_critic_layers = pytorch_mod.v.v_net

    layer_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
    actor_mlp_impl = create_mlp(layer_sizes, activation,
                          pytorch_layers=pytorch_actor_layers)
    actor_mlp = lambda x: run_mlp(x, actor_mlp_impl)
    step = lambda x: np_categorical_step(x, actor_mlp_impl)
    log_prob = lambda x, p: np_log_prob(x, p)
    actor = Actor(sample=lambda x: np_categorical_sample(x, actor_mlp_impl),
                  log_prob_from_distribution=log_prob,
                  actor_mlp=actor_mlp,
                  step=step)
    critic_mlp_impl = create_mlp([obs_dim] + list(hidden_sizes) + [1], activation,
                          pytorch_layers=pytorch_critic_layers)
    v_fnc = lambda x: run_mlp(x, critic_mlp_impl)
    step_fnc = lambda x: actor.step(x) + (v_fnc(x),)
    ac = ActorCritic(actor=actor, critic=v_fnc, step=step_fnc)
    return ac

def update(actor_critic, data):

    def compute_loss_pi(x):
        pass

    def compute_loss_v(x):
        pass

    pi_l_old = compute_loss_pi(data)
    v_l_old = compute_loss_v(data)


def mlp_gaussian(obs_dim, act_dim, hidden_sizes, activation, pytorch_layers=None, env_sample=None):
    layer_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
    mlp_impl = create_mlp(layer_sizes, activation,
                          pytorch_layers=pytorch_layers)
    dist = lambda x: np.random.choice(run_mlp(x, mlp_impl), replace=False)
    log_prob = lambda pi, lprob: pi * lprob


def create_actor_critic(obs_space, action_space, hidden_sizes, activation, pytorch_mod=None, env_sample=None):
    obs_dim = obs_space.shape[0]
    assert isinstance(hidden_sizes, tuple)
    if isinstance(action_space, Box):
        act_dim = action_space.shape[0]
        pi = mlp_gaussian(obs_dim, act_dim, hidden_sizes, activation, pytorch_mod, env_sample)
    else:
        assert isinstance(action_space, Discrete)
        act_dim = action_space.n
        pi = mlp_categorical_dist(obs_dim, act_dim, hidden_sizes, activation, pytorch_mod, env_sample)
    return pi

def ppo1_impl():

    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, verbose=1)
    return model, env

def value_func(obs):
    pass

def entropy(x):
    logp = np.log(x)
    return np.sum(-x * logp)

def compute_value_loss(data, value_func):
    obs, ret = data['obs'], data['ret']
    return np.mean((value_func(obs) - ret) ** 2)

def compute_loss_pi(data, clip_ratio, ac: ActorCritic):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    pi, logp = mlp_cat_actor(obs, act)
    ratio = np.exp(logp - logp_old)
    clip_adv = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -np.mean(np.min(ratio * adv, clip_adv))

    # Useful extra info
    approx_kl = np.mean(logp_old - logp)
    ent = np.mean(entropy(pi))
    clipped = (ratio > (1 + clip_ratio)) | (ratio < (1 - clip_ratio))
    clipfrac = np.mean(clipped)
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

TORCH_NP_FN_MAP = {
    "ReLU": np_relu,
    "Linear": np_linear,
    "Identity": np_identity
}