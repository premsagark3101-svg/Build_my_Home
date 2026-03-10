"""
ppo_agent.py
============
Proximal Policy Optimization (PPO) implemented from scratch using NumPy.
Mirrors the Stable-Baselines3 PPO interface for drop-in compatibility.

Architecture:
  Actor   : MLP  obs → logits over actions
  Critic  : MLP  obs → state value V(s)
  Both share a common feature trunk.

Key PPO hyperparameters:
  clip_epsilon  : 0.2  (trust-region clip)
  gamma         : 0.99 (discount)
  gae_lambda    : 0.95 (GAE λ)
  n_epochs      : 4    (update epochs per rollout)
  entropy_coef  : 0.01 (exploration bonus)
  vf_coef       : 0.5  (value loss weight)
"""

from __future__ import annotations

import math
import random
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network (pure NumPy MLP with Adam optimiser)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def log_softmax(x: np.ndarray) -> np.ndarray:
    return x - x.max() - np.log(np.exp(x - x.max()).sum())


class Layer:
    """Fully-connected layer with Adam optimiser."""

    def __init__(self, in_dim: int, out_dim: int, lr: float = 3e-4):
        scale = math.sqrt(2.0 / in_dim)
        self.W  = (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0
        # Cache for backprop
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        dW = self._x.T @ grad_out if self._x.ndim > 1 else np.outer(self._x, grad_out)
        db = grad_out.sum(axis=0) if grad_out.ndim > 1 else grad_out
        dx = grad_out @ self.W.T
        self._adam_update(dW, db)
        return dx

    def _adam_update(self, dW: np.ndarray, db: np.ndarray,
                     beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.t += 1
        # W
        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * dW ** 2
        mWh = self.mW / (1 - beta1 ** self.t)
        vWh = self.vW / (1 - beta2 ** self.t)
        self.W -= self.lr * mWh / (np.sqrt(vWh) + eps)
        # b
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * db ** 2
        mbh = self.mb / (1 - beta1 ** self.t)
        vbh = self.vb / (1 - beta2 ** self.t)
        self.b -= self.lr * mbh / (np.sqrt(vbh) + eps)


class ActorCriticNet:
    """
    Shared trunk + separate actor/critic heads.
    Input  : flat observation vector
    Output : (action_logits [n_actions], value [1])
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, lr: float = 3e-4):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        # Shared trunk (2 hidden layers)
        self.fc1    = Layer(obs_dim,  hidden, lr)
        self.fc2    = Layer(hidden,   hidden, lr)

        # Actor head
        self.actor  = Layer(hidden, n_actions, lr)

        # Critic head
        self.critic = Layer(hidden, 1, lr)

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """obs: flat float32 vector → (logits, value)"""
        h = relu(self.fc1.forward(obs))
        h = relu(self.fc2.forward(h))
        logits = self.actor.forward(h)
        value  = float(self.critic.forward(h).squeeze())
        return logits, value

    def get_action(self, obs: np.ndarray, deterministic: bool = False
                   ) -> tuple[int, float, float]:
        """Sample action → (action, log_prob, value)"""
        logits, value = self.forward(obs)
        log_probs     = log_softmax(logits)
        if deterministic:
            action = int(np.argmax(log_probs))
        else:
            probs  = np.exp(log_probs)
            probs /= probs.sum()   # renormalise
            action = int(np.random.choice(len(probs), p=probs))
        return action, float(log_probs[action]), value

    def evaluate(self, obs: np.ndarray, action: int
                 ) -> tuple[float, float, float]:
        """Evaluate obs/action → (log_prob, value, entropy)"""
        logits, value = self.forward(obs)
        log_probs = log_softmax(logits)
        probs     = np.exp(log_probs)
        probs    /= probs.sum()
        entropy   = float(-np.sum(probs * log_probs))
        return float(log_probs[action]), value, entropy

    def update_actor(self, grad: np.ndarray):
        """Backprop through actor head → trunk."""
        dh     = self.actor.backward(grad)
        dh    *= relu_grad(self.fc2._x @ self.fc2.W + self.fc2.b)
        dh     = self.fc2.backward(dh)
        dh    *= relu_grad(self.fc1._x @ self.fc1.W + self.fc1.b)
        self.fc1.backward(dh)

    def update_critic(self, grad: np.ndarray):
        """Backprop through critic head → trunk."""
        dh     = self.critic.backward(grad)
        dh    *= relu_grad(self.fc2._x @ self.fc2.W + self.fc2.b)
        dh     = self.fc2.backward(dh)
        dh    *= relu_grad(self.fc1._x @ self.fc1.W + self.fc1.b)
        self.fc1.backward(dh)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.obs:       list[np.ndarray] = []
        self.actions:   list[int]        = []
        self.log_probs: list[float]      = []
        self.rewards:   list[float]      = []
        self.values:    list[float]      = []
        self.dones:     list[bool]       = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation."""
        n          = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else self.values[t + 1]
            not_done  = 1.0 - float(self.dones[t])
            delta     = self.rewards[t] + gamma * next_val * not_done - self.values[t]
            gae       = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
            returns[t]    = gae + self.values[t]

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimisation agent.

    Parameters
    ----------
    obs_dim      : flattened observation dimension
    n_actions    : number of discrete actions
    lr           : learning rate
    gamma        : discount factor
    gae_lambda   : GAE lambda
    clip_epsilon : PPO clip ratio
    n_epochs     : gradient update epochs per rollout
    batch_size   : mini-batch size
    entropy_coef : entropy bonus coefficient
    vf_coef      : value function loss coefficient
    """

    def __init__(
        self,
        obs_dim:      int,
        n_actions:    int,
        lr:           float = 3e-4,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        clip_epsilon: float = 0.2,
        n_epochs:     int   = 4,
        batch_size:   int   = 64,
        entropy_coef: float = 0.01,
        vf_coef:      float = 0.5,
        hidden:       int   = 256,
    ):
        self.net          = ActorCriticNet(obs_dim, n_actions, hidden, lr)
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.entropy_coef = entropy_coef
        self.vf_coef      = vf_coef
        self.buffer       = RolloutBuffer(capacity=2048)

        # Metrics
        self.total_steps   = 0
        self.episode_count = 0
        self.losses: list[dict] = []

    def get_action(self, obs_flat: np.ndarray, deterministic: bool = False):
        return self.net.get_action(obs_flat, deterministic)

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.add(obs, action, log_prob, reward, value, done)
        self.total_steps += 1

    def update(self, last_value: float = 0.0) -> dict:
        """Run PPO update on collected rollout. Returns loss dict."""
        if len(self.buffer) == 0:
            return {}

        advantages, returns = self.buffer.compute_returns(
            last_value, self.gamma, self.gae_lambda
        )

        obs_arr     = np.array(self.buffer.obs,       dtype=np.float32)
        actions_arr = np.array(self.buffer.actions,   dtype=np.int32)
        old_lps_arr = np.array(self.buffer.log_probs, dtype=np.float32)
        n           = len(self.buffer)

        total_pg_loss  = 0.0
        total_vf_loss  = 0.0
        total_ent      = 0.0
        total_updates  = 0

        for epoch in range(self.n_epochs):
            indices = list(range(n))
            random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]
                if len(batch_idx) < 4:
                    continue

                # Accumulate gradients over mini-batch
                pg_loss_sum = 0.0
                vf_loss_sum = 0.0
                ent_sum     = 0.0

                actor_grads  = np.zeros(self.net.n_actions, dtype=np.float32)
                critic_grads = np.zeros(1, dtype=np.float32)

                for i in batch_idx:
                    obs    = obs_arr[i]
                    action = int(actions_arr[i])
                    old_lp = float(old_lps_arr[i])
                    adv    = float(advantages[i])
                    ret    = float(returns[i])

                    new_lp, value, entropy = self.net.evaluate(obs, action)

                    # Policy gradient loss (PPO clip)
                    ratio   = math.exp(new_lp - old_lp)
                    pg1     = ratio * adv
                    pg2     = np.clip(ratio, 1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) * adv
                    pg_loss = -min(pg1, pg2)

                    # Value function loss
                    vf_loss = 0.5 * (value - ret) ** 2

                    # Total loss
                    loss    = pg_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy

                    pg_loss_sum += pg_loss
                    vf_loss_sum += vf_loss
                    ent_sum     += entropy

                    # Approximate gradient for actor (policy gradient)
                    # ∂L/∂logit_a ≈ -adv * clip_ratio_indicator
                    clip_ok     = 1.0 if (1-self.clip_epsilon) <= ratio <= (1+self.clip_epsilon) else 0.0
                    grad_logit  = np.zeros(self.net.n_actions, dtype=np.float32)
                    grad_logit[action] = -clip_ok * adv / len(batch_idx)
                    actor_grads += grad_logit

                    # Critic gradient
                    critic_grads += np.array([2 * (value - ret) * self.vf_coef / len(batch_idx)])

                # Apply averaged gradients
                # Forward pass needed to set cache for backward
                for i in batch_idx[:1]:
                    self.net.forward(obs_arr[i])
                self.net.update_actor(actor_grads / len(batch_idx))

                for i in batch_idx[:1]:
                    self.net.forward(obs_arr[i])
                self.net.update_critic(critic_grads / len(batch_idx))

                b = len(batch_idx)
                total_pg_loss += pg_loss_sum / b
                total_vf_loss += vf_loss_sum / b
                total_ent     += ent_sum / b
                total_updates += 1

        self.buffer.clear()
        denom = max(total_updates, 1)
        metrics = {
            "pg_loss":  round(total_pg_loss / denom, 4),
            "vf_loss":  round(total_vf_loss / denom, 4),
            "entropy":  round(total_ent     / denom, 4),
            "steps":    self.total_steps,
        }
        self.losses.append(metrics)
        return metrics
