"""
train.py
========
PPO training loop for the FloorPlanEnv.

Usage:
    python train.py                     # train + visualise
    python train.py --episodes 500      # custom episode count
    python train.py --no-render         # skip visualization
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from floor_plan_env import FloorPlanEnv, GRID
from ppo_agent import PPOAgent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_obs(obs: dict) -> np.ndarray:
    """Flatten dict observation into a single 1-D vector."""
    grid_flat = obs["grid"].flatten()
    vec_flat  = obs["vec"].flatten()
    return np.concatenate([grid_flat, vec_flat]).astype(np.float32)


def rollout_one_episode(
    env: FloorPlanEnv, agent: PPOAgent, store: bool = True
) -> dict:
    """Run a single episode and optionally store transitions."""
    obs, info = env.reset()
    obs_flat  = flatten_obs(obs)
    ep_reward = 0.0
    ep_steps  = 0
    done      = False

    while not done:
        action, log_prob, value = agent.get_action(obs_flat)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done      = terminated or truncated
        next_flat = flatten_obs(next_obs)

        if store:
            agent.store(obs_flat, action, log_prob, reward, value, done)

        obs_flat   = next_flat
        ep_reward += reward
        ep_steps  += 1

    return {
        "reward":  ep_reward,
        "steps":   ep_steps,
        "placed":  info["placed"],
        "layout":  env.get_layout_json(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_episodes:   int  = 300,
    update_every: int  = 10,    # PPO update every N episodes
    seed:         int  = 42,
    verbose:      bool = True,
    save_path:    str  = "trained_agent.npz",
) -> tuple[PPOAgent, FloorPlanEnv, list[dict]]:

    np.random.seed(seed)

    env = FloorPlanEnv(seed=seed)

    # Observation dim: 3 channels × GRID × GRID + vec
    obs_dim   = 3 * GRID * GRID + 9   # 9 = 8 room types + 1 progress
    n_actions = GRID * GRID

    agent = PPOAgent(
        obs_dim      = obs_dim,
        n_actions    = n_actions,
        lr           = 3e-4,
        gamma        = 0.99,
        gae_lambda   = 0.95,
        clip_epsilon = 0.2,
        n_epochs     = 4,
        batch_size   = 64,
        entropy_coef = 0.02,
        vf_coef      = 0.5,
        hidden       = 128,
    )

    history:      list[dict] = []
    best_reward   = -9999.0
    best_layout:  list[dict] = []
    ep_rewards:   list[float] = []
    update_count  = 0

    print("=" * 60)
    print("  Floor Plan RL — PPO Training")
    print(f"  Episodes : {n_episodes}")
    print(f"  Grid     : {GRID}×{GRID}")
    print(f"  Actions  : {n_actions}")
    print(f"  Obs dim  : {obs_dim}")
    print("=" * 60)

    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        result = rollout_one_episode(env, agent, store=True)
        ep_reward = result["reward"]
        ep_rewards.append(ep_reward)

        # Track best layout
        if ep_reward > best_reward and result["layout"]:
            best_reward = ep_reward
            best_layout = result["layout"]

        # PPO update every N episodes
        if ep % update_every == 0:
            _, last_val, _ = agent.get_action(
                flatten_obs(env.reset()[0]), deterministic=True
            )
            metrics = agent.update(last_value=last_val)
            update_count += 1

            avg_r  = np.mean(ep_rewards[-update_every:])
            elapsed = time.time() - t0
            rec = {
                "episode":    ep,
                "avg_reward": round(float(avg_r), 3),
                "best_reward":round(float(best_reward), 3),
                "update":     update_count,
                **metrics,
                "elapsed_s":  round(elapsed, 1),
            }
            history.append(rec)

            if verbose:
                print(
                    f"  Ep {ep:>4d}/{n_episodes}"
                    f"  avg_r={avg_r:>7.2f}"
                    f"  best={best_reward:>7.2f}"
                    f"  pg={metrics.get('pg_loss', 0):>6.3f}"
                    f"  vf={metrics.get('vf_loss', 0):>6.3f}"
                    f"  ent={metrics.get('entropy', 0):>5.3f}"
                    f"  [{elapsed:.0f}s]"
                )

    total_time = time.time() - t0
    print("=" * 60)
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best episode reward : {best_reward:.2f}")
    print(f"  Best layout rooms   : {len(best_layout)}")
    print("=" * 60)

    return agent, env, history, best_layout


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (greedy rollout)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent: PPOAgent, env: FloorPlanEnv, n_eval: int = 5) -> list[dict]:
    """Run greedy evaluation episodes."""
    results = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=100 + i)
        obs_flat = flatten_obs(obs)
        done = False
        total_r = 0.0
        while not done:
            action, _, _ = agent.get_action(obs_flat, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            obs_flat = flatten_obs(obs)
            total_r += r
            done = terminated or truncated
        layout = env.get_layout_json()
        results.append({"reward": round(total_r, 2), "layout": layout})
        print(f"  Eval {i+1}: reward={total_r:.2f}  rooms_placed={len(layout)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,  default=300)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    agent, env, history, best_layout = train(
        n_episodes=args.episodes,
        seed=args.seed,
    )

    print("\n── Evaluation (greedy) ──")
    eval_results = evaluate(agent, env)

    # Save outputs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "best_layout.json", "w") as f:
        json.dump(best_layout, f, indent=2)

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nBest layout:\n{json.dumps(best_layout, indent=2)}")

    if not args.no_render:
        from visualize import visualize_layout, plot_training_curves
        visualize_layout(best_layout, title="Best Layout Found by PPO",
                         save_path=str(out_dir / "best_layout.png"))
        plot_training_curves(history,
                             save_path=str(out_dir / "training_curves.png"))
