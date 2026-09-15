"""
index.py

프로젝트 메인 진입점. train / evaluate를 서브커맨드로 실행.

사용법:
    python index.py train --timesteps 300000 --out models/chaser_phase1.zip
    python index.py evaluate --model models/chaser_phase1.zip --episodes 50
"""

import argparse
import os

from drone_tag_env import DroneTagEnv


def cmd_train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    train_env = make_vec_env(lambda: DroneTagEnv(), n_envs=args.n_envs)
    eval_env = make_vec_env(lambda: DroneTagEnv(), n_envs=1)

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
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(args.out)
    print(f"학습 완료. 모델 저장: {args.out}")


def cmd_evaluate(args):
    import json

    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO

    model = PPO.load(args.model)
    env = DroneTagEnv()

    results = {
        "caught": 0,
        "chaser_out_of_bounds": 0,
        "evader_out_of_bounds": 0,
        "time_up_evader_wins": 0,
    }

    episodes_data = []  # HTML 애니메이션용 - 매 에피소드 궤적 전부 저장
    chaser_traj, evader_traj = [], []

    for ep_idx in range(args.episodes):
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
        chaser_traj, evader_traj = traj_c, traj_e
        episodes_data.append(
            {
                "index": ep_idx,
                "result": info["result"],
                "chaser": [p.tolist() for p in traj_c],
                "evader": [p.tolist() for p in traj_e],
            }
        )

    print("=== 결과 ===")
    for k, v in results.items():
        print(f"{k}: {v}/{args.episodes} ({100*v/args.episodes:.1f}%)")
    print(f"\ncatch율: {100*results['caught']/args.episodes:.1f}%")

    chaser_traj = np.array(chaser_traj)
    evader_traj = np.array(evader_traj)
    plt.figure(figsize=(6, 6))
    plt.plot(chaser_traj[:, 0], chaser_traj[:, 1], "b-", label="chaser (술래)")
    plt.plot(evader_traj[:, 0], evader_traj[:, 1], "r-", label="evader (도망자)")
    plt.scatter(*chaser_traj[0, :2], c="blue", marker="o")
    plt.scatter(*evader_traj[0, :2], c="red", marker="o")
    plt.legend()
    plt.title("마지막 에피소드 궤적 (x-y 평면)")
    plt.axis("equal")
    plt.savefig(args.plot_out, dpi=150)
    print(f"궤적 이미지 저장: {args.plot_out}")

    if args.html_out:
        _write_html_viewer(episodes_data, env.arena_xy, env.catch_radius, args.html_out)
        print(f"결과 뷰어(HTML) 저장: {args.html_out} — 학습 없이 브라우저로 그냥 열면 됨")


def _write_html_viewer(episodes_data, arena_xy, catch_radius, out_path):
    """
    학습 결과 궤적을 재생하는 완전 독립형 HTML 파일 생성.
    - 서버 필요 없음, 데이터가 HTML 안에 그대로 박혀있음 (file:// 로 그냥 열면 됨)
    - 에피소드 선택 드롭다운 + 재생/일시정지/속도 조절
    """
    data_json = json.dumps(episodes_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>드론 술래잡기 - 결과 뷰어</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #12141c; color: #eee; margin: 0; padding: 20px; }}
  h1 {{ font-size: 18px; font-weight: 600; }}
  #canvas-wrap {{ display: flex; justify-content: center; margin-top: 16px; }}
  canvas {{ background: #1c1f2b; border-radius: 8px; }}
  .controls {{ display: flex; gap: 12px; align-items: center; justify-content: center; margin-top: 16px; flex-wrap: wrap; }}
  select, button {{ background: #2a2e3d; color: #eee; border: 1px solid #444; border-radius: 6px; padding: 6px 12px; font-size: 14px; }}
  button {{ cursor: pointer; }}
  button:hover {{ background: #3a3f52; }}
  .legend {{ display: flex; gap: 16px; justify-content: center; margin-top: 10px; font-size: 13px; color: #aaa; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
  #result-badge {{ text-align: center; margin-top: 8px; font-size: 13px; color: #9aa; }}
</style>
</head>
<body>
  <h1>드론 술래잡기 결과 뷰어</h1>
  <div class="legend">
    <span><span class="dot" style="background:#4da3ff"></span>술래 (chaser)</span>
    <span><span class="dot" style="background:#ff5c7a"></span>도망자 (evader)</span>
  </div>
  <div id="canvas-wrap"><canvas id="c" width="600" height="600"></canvas></div>
  <div id="result-badge"></div>
  <div class="controls">
    <select id="epSelect"></select>
    <button id="playBtn">▶ 재생</button>
    <button id="resetBtn">⏮ 처음으로</button>
    <label>속도:
      <select id="speedSelect">
        <option value="1">1x</option>
        <option value="2" selected>2x</option>
        <option value="4">4x</option>
      </select>
    </label>
  </div>

<script>
const EPISODES = {data_json};
const ARENA_XY = {arena_xy};
const CATCH_R = {catch_radius};

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const scale = (W / 2) / (ARENA_XY * 1.1);
const cx = W / 2, cy = H / 2;

function toScreen(x, y) {{
  return [cx + x * scale, cy - y * scale]; // y축 반전 (화면 좌표계)
}}

const epSelect = document.getElementById('epSelect');
EPISODES.forEach((ep, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `에피소드 ${{ep.index}} — ${{ep.result}}`;
  epSelect.appendChild(opt);
}});

let currentEp = EPISODES[0];
let frame = 0;
let playing = false;
let speed = 2;

function drawArena() {{
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#3a3f52';
  ctx.lineWidth = 2;
  const [x0, y0] = toScreen(-ARENA_XY, ARENA_XY);
  const [x1, y1] = toScreen(ARENA_XY, -ARENA_XY);
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
}}

function drawTrail(traj, upTo, color) {{
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= upTo; i++) {{
    const [sx, sy] = toScreen(traj[i][0], traj[i][1]);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }}
  ctx.stroke();
  ctx.globalAlpha = 1.0;
}}

function drawDot(pos, color, radius) {{
  const [sx, sy] = toScreen(pos[0], pos[1]);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(sx, sy, radius, 0, Math.PI * 2);
  ctx.fill();
}}

function render() {{
  drawArena();
  const c = currentEp.chaser, e = currentEp.evader;
  const upTo = Math.min(frame, c.length - 1);
  drawTrail(c, upTo, '#4da3ff');
  drawTrail(e, upTo, '#ff5c7a');
  drawDot(c[upTo], '#4da3ff', 8);
  drawDot(e[upTo], '#ff5c7a', 8);

  // catch radius 표시 (술래 주변)
  const [sx, sy] = toScreen(c[upTo][0], c[upTo][1]);
  ctx.strokeStyle = 'rgba(77,163,255,0.3)';
  ctx.beginPath();
  ctx.arc(sx, sy, CATCH_R * scale, 0, Math.PI * 2);
  ctx.stroke();

  document.getElementById('result-badge').textContent =
    `스텝 ${{upTo}} / ${{c.length - 1}} — 결과: ${{currentEp.result}}`;
}}

function tick() {{
  if (!playing) return;
  frame += speed;
  if (frame >= currentEp.chaser.length - 1) {{
    frame = currentEp.chaser.length - 1;
    playing = false;
    document.getElementById('playBtn').textContent = '▶ 재생';
  }}
  render();
  if (playing) requestAnimationFrame(tick);
}}

document.getElementById('playBtn').onclick = () => {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ 일시정지' : '▶ 재생';
  if (playing) requestAnimationFrame(tick);
}};
document.getElementById('resetBtn').onclick = () => {{ frame = 0; render(); }};
document.getElementById('speedSelect').onchange = (e) => {{ speed = Number(e.target.value); }};
epSelect.onchange = (e) => {{
  currentEp = EPISODES[Number(e.target.value)];
  frame = 0;
  playing = false;
  document.getElementById('playBtn').textContent = '▶ 재생';
  render();
}};

render();
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def cmd_export(args):
    """
    에피소드 궤적을 JSON으로 저장 (웹 뷰어에서 애니메이션 재생용).
    --model이 없으면 랜덤 액션으로, 있으면 학습된 모델로 실행.
    """
    import json

    env = DroneTagEnv()
    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

    episodes_data = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        chaser_traj = [env.chaser_pos.tolist()]
        evader_traj = [env.evader_pos.tolist()]
        result = "unknown"
        for _ in range(env.max_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            chaser_traj.append(env.chaser_pos.tolist())
            evader_traj.append(env.evader_pos.tolist())
            if term or trunc:
                result = info.get("result", "unknown")
                break
        episodes_data.append(
            {"chaser": chaser_traj, "evader": evader_traj, "result": result}
        )

    out_data = {
        "arena": {"xy": env.arena_xy, "z": env.arena_z},
        "catch_radius": env.catch_radius,
        "dt": env.dt,
        "episodes": episodes_data,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f)
    print(f"궤적 {len(episodes_data)}개 에피소드 저장: {args.out}")


def cmd_demo(args):
    """환경만 빠르게 확인하고 싶을 때 (랜덤 액션, 학습 X)."""
    env = DroneTagEnv(seed=0)
    for ep in range(3):
        obs, _ = env.reset()
        total_r = 0.0
        for t in range(env.max_steps):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                print(f"episode {ep}: steps={t+1}, reward={total_r:.2f}, {info}")
                break


def main():
    parser = argparse.ArgumentParser(description="drone tag RL - 메인 진입점")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Phase 1 술래(chaser) PPO 학습")
    p_train.add_argument("--timesteps", type=int, default=300_000)
    p_train.add_argument("--n_envs", type=int, default=8)
    p_train.add_argument("--out", type=str, default="models/chaser_phase1.zip")
    p_train.add_argument("--logdir", type=str, default="logs/phase1")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="학습된 모델 평가 + 궤적 시각화")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--episodes", type=int, default=50)
    p_eval.add_argument("--plot_out", type=str, default="last_episode_trajectory.png")
    p_eval.add_argument(
        "--html_out",
        type=str,
        default="results.html",
        help="궤적 애니메이션 웹뷰어(HTML) 저장 경로. 빈 문자열이면 생성 안 함",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_demo = sub.add_parser("demo", help="환경 동작 확인 (랜덤 액션, 학습 없음)")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
