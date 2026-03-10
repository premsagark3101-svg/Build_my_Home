"""
multifloor_train.py
===================
Hierarchical PPO training for multi-floor building generation.

Two specialised agents:
  L1Agent — trained on ground floor planning (living spaces)
  L2Agent — trained on upper floor planning (bedrooms, alignment)

Training regime:
  1. Train L1 for N episodes on ground floor only
  2. Train L2 for N episodes using L1's best ground floor
     as the structural template
  3. Joint evaluation: L1 plans ground, L2 plans all upper floors
"""

from __future__ import annotations

import argparse
import json
import time
import numpy as np
from pathlib import Path

from multifloor_env import MultiFloorEnv, GRID
from ppo_agent import PPOAgent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate([
        obs["grid"].flatten(),
        obs["vec"].flatten(),
    ]).astype(np.float32)


def make_agent(obs_dim: int, seed: int = 42) -> PPOAgent:
    np.random.seed(seed)
    return PPOAgent(
        obs_dim      = obs_dim,
        n_actions    = GRID * GRID,
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


def run_floor_episode(
    env:       MultiFloorEnv,
    agent:     PPOAgent,
    floor_num: int,
    store:     bool = True,
) -> dict:
    """Run one episode on a single floor. Returns metrics dict."""
    obs, info = env.reset_floor(floor_num)
    obs_flat  = flatten_obs(obs)
    ep_reward = 0.0
    ep_steps  = 0
    done      = False

    while not done:
        action, log_prob, value = agent.get_action(obs_flat)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if store:
            agent.store(obs_flat, action, log_prob, reward, value, done)

        obs_flat   = flatten_obs(next_obs)
        ep_reward += reward
        ep_steps  += 1

    return {
        "reward":  ep_reward,
        "steps":   ep_steps,
        "layout":  info["layout"],
        "placed":  len([r for r in info["layout"] if not r.get("structural")]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Train Level-1 agent (ground floor)
# ─────────────────────────────────────────────────────────────────────────────

def train_l1(
    n_episodes:   int  = 200,
    update_every: int  = 10,
    seed:         int  = 42,
    n_floors:     int  = 3,
    verbose:      bool = True,
) -> tuple[PPOAgent, list[dict], list[dict]]:

    print("\n" + "═"*62)
    print("  PHASE 1 — Level-1 Agent: Ground Floor Planning")
    print("═"*62)

    env   = MultiFloorEnv(n_floors=n_floors, seed=seed)
    agent = make_agent(obs_dim=env.obs_dim, seed=seed)

    best_reward = -9999.0
    best_layout: list[dict] = []
    ep_rewards:  list[float] = []
    history:     list[dict]  = []
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        result = run_floor_episode(env, agent, floor_num=1, store=True)
        ep_rewards.append(result["reward"])

        if result["reward"] > best_reward and result["placed"] > 0:
            best_reward = result["reward"]
            best_layout = result["layout"]

        if ep % update_every == 0:
            obs_r, _ = env.reset_floor(1)
            _, lv, _ = agent.get_action(flatten_obs(obs_r), deterministic=True)
            m  = agent.update(last_value=lv)
            avg = np.mean(ep_rewards[-update_every:])
            rec = {
                "phase": "L1", "episode": ep,
                "avg_reward": round(float(avg), 3),
                "best_reward": round(float(best_reward), 3),
                **m,
            }
            history.append(rec)
            if verbose:
                print(
                    f"  [L1] Ep {ep:>3d}/{n_episodes}"
                    f"  avg={avg:>6.2f}  best={best_reward:>6.2f}"
                    f"  placed={result['placed']}"
                    f"  ent={m.get('entropy',0):.3f}"
                    f"  [{time.time()-t0:.0f}s]"
                )

    print(f"\n  L1 best reward : {best_reward:.2f}")
    print(f"  L1 rooms placed: {len(best_layout)} (incl. structural)")
    return agent, best_layout, history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Train Level-2 agent (upper floors)
# ─────────────────────────────────────────────────────────────────────────────

def train_l2(
    ground_layout: list[dict],
    n_episodes:    int  = 200,
    update_every:  int  = 10,
    seed:          int  = 42,
    n_floors:      int  = 3,
    verbose:       bool = True,
) -> tuple[PPOAgent, dict, list[dict]]:

    print("\n" + "═"*62)
    print("  PHASE 2 — Level-2 Agent: Upper Floor Planning")
    print("═"*62)

    env   = MultiFloorEnv(n_floors=n_floors, seed=seed)
    agent = make_agent(obs_dim=env.obs_dim, seed=seed+100)

    # Inject best ground floor as the structural template
    # We do this by running a deterministic ground floor episode
    # and then calling finalise
    env.reset_floor(1)
    env.floors[1].placed_rooms = []
    env.floors[1].occupancy    = np.zeros((GRID, GRID), dtype=np.float32)
    env.floors[1].room_id_map  = np.zeros((GRID, GRID), dtype=np.float32)
    for r in ground_layout:
        if not r.get("structural"):
            from floor_plan_env import PlacedRoom
            pr = PlacedRoom(r["room"], r["x"], r["y"], r["w"], r["h"])
            env.floors[1]._place(pr)
    env.finalise_ground_floor()

    best_reward = -9999.0
    best_building: dict = {}
    ep_rewards: list[float] = []
    history: list[dict] = []
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        # Train on floor 2 (representative upper floor)
        floor_num  = 2 if n_floors >= 2 else 1
        result     = run_floor_episode(env, agent, floor_num=floor_num, store=True)
        ep_rewards.append(result["reward"])

        if result["reward"] > best_reward and result["placed"] > 0:
            best_reward = result["reward"]

        if ep % update_every == 0:
            obs_r, _ = env.reset_floor(floor_num)
            _, lv, _ = agent.get_action(flatten_obs(obs_r), deterministic=True)
            m   = agent.update(last_value=lv)
            avg = np.mean(ep_rewards[-update_every:])
            rec = {
                "phase": "L2", "episode": ep,
                "avg_reward": round(float(avg), 3),
                "best_reward": round(float(best_reward), 3),
                **m,
            }
            history.append(rec)
            if verbose:
                print(
                    f"  [L2] Ep {ep:>3d}/{n_episodes}"
                    f"  avg={avg:>6.2f}  best={best_reward:>6.2f}"
                    f"  placed={result['placed']}"
                    f"  ent={m.get('entropy',0):.3f}"
                    f"  [{time.time()-t0:.0f}s]"
                )

    print(f"\n  L2 best reward : {best_reward:.2f}")
    return agent, best_building, history


# ─────────────────────────────────────────────────────────────────────────────
# Full building generation (joint evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def generate_building(
    l1_agent:  PPOAgent,
    l2_agent:  PPOAgent,
    n_floors:  int  = 3,
    seed:      int  = 99,
    deterministic: bool = True,
) -> dict:
    """
    Use trained L1 + L2 agents to generate a complete multi-floor building.
    """
    print(f"\n── Generating {n_floors}-floor building (seed={seed}) ──")
    env = MultiFloorEnv(n_floors=n_floors, seed=seed)

    # ── Floor 1: L1 agent ────────────────────────────────────────────
    obs, _ = env.reset_floor(1)
    done   = False
    while not done:
        action, _, _ = l1_agent.get_action(flatten_obs(obs), deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.save_floor_layout(1)

    core = env.finalise_ground_floor()
    print(f"  Ground floor: {len(env.layouts[1])} rooms placed")
    print(f"  Staircase   : {core.staircase}")
    print(f"  Elevator    : {core.elevator}")
    print(f"  Plumbing stacks: {core.plumbing_stacks}")

    # ── Floors 2+: L2 agent ──────────────────────────────────────────
    for floor_num in range(2, n_floors + 1):
        obs, _ = env.reset_floor(floor_num)
        done   = False
        while not done:
            action, _, _ = l2_agent.get_action(flatten_obs(obs), deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.save_floor_layout(floor_num)
        print(f"  Floor {floor_num}: {len(env.layouts[floor_num])} rooms placed")

    building = env.get_building_layout()
    return building


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,  default=200)
    parser.add_argument("--floors",    type=int,  default=3)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    # Train
    l1_agent, best_ground, l1_history = train_l1(
        n_episodes=args.episodes, n_floors=args.floors, seed=args.seed
    )
    l2_agent, _, l2_history = train_l2(
        ground_layout=best_ground,
        n_episodes=args.episodes, n_floors=args.floors, seed=args.seed
    )

    # Generate full building
    building = generate_building(l1_agent, l2_agent, n_floors=args.floors)

    # Save
    out = Path("outputs"); out.mkdir(exist_ok=True)
    with open(out / "building_layout.json", "w") as f:
        json.dump(building, f, indent=2)

    history = l1_history + l2_history
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBuilding layout saved → outputs/building_layout.json")
    print(json.dumps({k: len(v) for k, v in building.items()
                      if k.startswith("floor")}, indent=2))

    if not args.no_render:
        from multifloor_visualize import visualize_building
        visualize_building(building, save_path=str(out/"building.png"))
