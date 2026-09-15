"""
train_phase1.py

Phase 1: 도망자는 규칙기반으로 고정, 술래(chaser)만 PPO로 학습.
로컬에서도, Colab에서도 그대로 돌아가게 만듦.

사용법:
    python train_phase1.py --timesteps 300000 --out models/chaser_phase1.zip
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from drone_tag_env import DroneTagEnv


def make_env():
    return DroneTagEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=8, help="병렬 환경 개수")
    parser.add_argument("--out", type=str, default="models/chaser_phase1.zip")
    parser.add_argument("--logdir", type=str, default="logs/phase1")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    train_env = make_vec_env(make_env, n_envs=args.n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best_model"),
        log_path=args.logdir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,  # 탐험 장려 (추격 전략이 한 가지로 고착되지 않게)
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(args.out)
    print(f"학습 완료. 모델 저장: {args.out}")


if __name__ == "__main__":
    main()
