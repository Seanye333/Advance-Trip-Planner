"""
RL Trip Agent — Stable-Baselines3 Wrapper
==========================================
Trains and runs a PPO (Proximal Policy Optimisation) agent on TripPlanningEnv.

PPO is chosen because:
  - Works well with continuous observations + discrete actions
  - Sample-efficient and stable
  - Handles the sparse reward structure of trip planning

Training tip: run for ~500k steps with a learning rate of 3e-4.
The agent will learn to balance exploration vs exploitation of attractions.

Alternative: use DQN for fully discrete action/observation spaces,
or SAC for continuous action spaces.
"""
from __future__ import annotations

import os
from pathlib import Path

from trip_ai.core.models import TravelerProfile, TravelNode
from trip_ai.graph_engine.travel_graph import TravelGraph
from trip_ai.rl_agent.trip_env import TripPlanningEnv


class RLTripAgent:
    """
    Wrapper around Stable-Baselines3 PPO for trip planning.

    Usage:
        agent = RLTripAgent(attractions, profile, graph)
        agent.train(total_timesteps=200_000)
        route = agent.plan()
    """

    def __init__(
        self,
        attractions: list[TravelNode],
        profile: TravelerProfile,
        graph: TravelGraph,
        model_dir: str = "models/rl",
    ) -> None:
        self.attractions = attractions
        self.profile = profile
        self.graph = graph
        self.model_dir = model_dir
        self._model = None
        self._env = TripPlanningEnv(attractions, profile, graph)

    def train(
        self,
        total_timesteps: int = 200_000,
        save_path: str | None = None,
        verbose: int = 1,
    ) -> None:
        """Train a PPO agent on the trip environment."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed. Run: pip install stable-baselines3"
            )

        env = DummyVecEnv([lambda: Monitor(
            TripPlanningEnv(self.attractions, self.profile, self.graph)
        )])

        self._model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=verbose,
            tensorboard_log=f"{self.model_dir}/tb_logs",
        )

        self._model.learn(total_timesteps=total_timesteps)

        if save_path:
            os.makedirs(self.model_dir, exist_ok=True)
            self._model.save(save_path)
            print(f"Model saved to {save_path}")

    def load(self, model_path: str) -> None:
        """Load a previously trained model."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("stable-baselines3 not installed")
        self._model = PPO.load(model_path, env=self._env)

    def plan(self, max_steps: int = 200) -> list[str]:
        """
        Run the trained policy greedily and return the list of visited attraction IDs.
        """
        if self._model is None:
            raise RuntimeError("Agent not trained. Call .train() or .load() first.")

        obs, _ = self._env.reset()
        visited: list[str] = []
        n = len(self.attractions)

        for _ in range(max_steps):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = self._env.step(int(action))

            if int(action) < n:
                node = self.attractions[int(action)]
                if node.id not in visited:
                    visited.append(node.id)

            if done or truncated:
                break

        return visited

    def evaluate(self, n_episodes: int = 10) -> dict[str, float]:
        """Evaluate the policy over multiple episodes, returning avg reward + stats."""
        if self._model is None:
            raise RuntimeError("Agent not trained.")

        try:
            from stable_baselines3.common.evaluation import evaluate_policy
        except ImportError:
            raise ImportError("stable-baselines3 not installed")

        mean_reward, std_reward = evaluate_policy(
            self._model,
            self._env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )
        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_episodes": n_episodes,
        }
