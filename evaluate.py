"""
evaluate.py

학습된 술래 모델을 N 에피소드 돌려서 catch율을 확인하고,
마지막 1개 에피소드의 궤적을 그림으로 저장.

사용법:
    python evaluate.py --model models/chaser_phase1.zip --episodes 50
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from drone_tag_env import DroneTagEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--plot_out", type=str, default="last_episode_trajectory.png")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = DroneTagEnv()

    results = {"caught": 0, "chaser_out_of_bounds": 0, "evader_out_of_bounds": 0, "time_up_evader_wins": 0}

    chaser_traj, evader_traj = [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        traj_c, traj_e = [env.chaser_pos.copy()], [env.evader_pos.copy()]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            traj_c.append(env.chaser_pos.copy())
            traj_e.append(env.evader_pos.copy())
            if term or trunc:
                results[info["result"]] += 1
                break
        chaser_traj, evader_traj = traj_c, traj_e  # 마지막 에피소드만 남김

    print("=== 결과 ===")
    for k, v in results.items():
        print(f"{k}: {v}/{args.episodes} ({100*v/args.episodes:.1f}%)")

    catch_rate = results["caught"] / args.episodes
    print(f"\ncatch율: {catch_rate*100:.1f}%")

    # 마지막 에피소드 궤적 시각화 (x, y 평면)
    chaser_traj = np.array(chaser_traj)
    evader_traj = np.array(evader_traj)

    plt.figure(figsize=(6, 6))
    plt.plot(chaser_traj[:, 0], chaser_traj[:, 1], "b-", label="chaser (술래)")
    plt.plot(evader_traj[:, 0], evader_traj[:, 1], "r-", label="evader (도망자)")
    plt.scatter(*chaser_traj[0, :2], c="blue", marker="o", label="chaser start")
    plt.scatter(*evader_traj[0, :2], c="red", marker="o", label="evader start")
    plt.legend()
    plt.title("마지막 에피소드 궤적 (x-y 평면)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.savefig(args.plot_out, dpi=150)
    print(f"궤적 이미지 저장: {args.plot_out}")


if __name__ == "__main__":
    main()
