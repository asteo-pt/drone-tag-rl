"""
drone_tag_env_pybullet.py

FPV 드론 술래잡기 - 실제 물리 버전 (gym-pybullet-drones 기반)
- Crazyflie 2.X 쿼드콥터 모델(cf2x) 사용, 4개 로터 각각의 추력/토크를
  PyBullet 강체 시뮬레이션으로 계산 (뉴턴-오일러 방정식, 공기저항,
  지면효과(ground effect) 등 실제 물리 반영)
- 드론 0 = 술래(chaser, RL 에이전트가 제어)
- 드론 1 = 도망자(evader, 규칙기반)
- 액션 타입: VEL (목표 방향 + 속력 -> 내부 PID 컨트롤러가 4개 모터 RPM으로 변환)
  -> RL은 "어느 방향으로 얼마나 빠르게"만 출력하면 되고, 그걸 실제 로터 RPM으로
     바꾸는 저수준 제어는 라이브러리가 검증된 PID로 처리함 (실감나는 비행 특성 유지)

의존성: gym-pybullet-drones (pip install --upgrade git+https://github.com/utiasDSL/gym-pybullet-drones.git)
"""

import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class DroneTagPyBulletEnv(BaseRLAviary):
    """
    관측값 (12차원, 기존 point-mass 버전과 동일한 형식 유지):
        [chaser_pos(3), chaser_vel(3), rel_pos_to_evader(3), rel_vel_to_evader(3)]

    액션 (4차원): VEL 타입 - [dir_x, dir_y, dir_z, speed_fraction], 각 [-1, 1]
    """

    def __init__(
        self,
        arena_xy: float = 7.5,       # 경기장 x,y 반경 (총 15m x 15m) - 실제 물리라 30m는 너무 넓어서 축소 추천
        arena_z: float = 3.0,        # 경기장 최대 고도
        catch_radius: float = 0.3,
        episode_len_sec: int = 20,   # 제한시간(초)
        evader_speed_frac: float = 0.55,  # 도망자 속력 비율 (술래보다 살짝 느리게)
        gui: bool = False,           # Colab에서는 항상 False (헤드리스)
        seed: int | None = None,
    ):
        self.arena_xy = arena_xy
        self.arena_z = arena_z
        self.catch_radius = catch_radius
        self.evader_speed_frac = evader_speed_frac
        self._np_rng = np.random.default_rng(seed)

        init_xyzs = np.array(
            [
                [arena_xy * 0.4, 0.0, 1.0],   # 술래 시작 위치
                [-arena_xy * 0.4, 0.0, 1.0],  # 도망자 시작 위치
            ]
        )

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=2,
            initial_xyzs=init_xyzs,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=30,
            gui=gui,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )
        self.EPISODE_LEN_SEC = episode_len_sec

        # BaseRLAviary는 기본적으로 드론 2대 모두 RL 액션을 받는 구조인데,
        # 여기선 술래(드론0)만 RL로 제어하고 도망자(드론1)는 규칙기반이라
        # 외부에 노출되는 action_space는 술래 1대분(4차원)만 보이게 덮어씀
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        high = np.array([np.inf] * 12, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.chaser_pos = init_xyzs[0].copy()
        self.evader_pos = init_xyzs[1].copy()
        self.max_steps = int(episode_len_sec * self.CTRL_FREQ)
        self.step_count = 0

    # ------------------------------------------------------------------ #
    # 도망자 규칙기반 정책 (VEL 액션 포맷으로 반환)
    # ------------------------------------------------------------------ #
    def _evader_action(self):
        chaser_state = self._getDroneStateVector(0)
        evader_state = self._getDroneStateVector(1)
        chaser_pos = chaser_state[0:3]
        evader_pos = evader_state[0:3]

        away = evader_pos - chaser_pos
        dist = np.linalg.norm(away) + 1e-6
        away_dir = away / dist

        to_center = np.array(
            [-evader_pos[0], -evader_pos[1], (self.arena_z / 2) - evader_pos[2]]
        )
        to_center = to_center / (np.linalg.norm(to_center) + 1e-6)

        edge_dist = min(
            self.arena_xy - abs(evader_pos[0]),
            self.arena_xy - abs(evader_pos[1]),
            evader_pos[2],
            self.arena_z - evader_pos[2],
        )
        edge_urgency = np.clip(1.0 - edge_dist / 0.8, 0.0, 1.0)

        jitter = self._np_rng.normal(0, 0.2, size=3)
        direction = (1 - edge_urgency) * away_dir + edge_urgency * to_center + jitter
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        return np.array(
            [direction[0], direction[1], direction[2], self.evader_speed_frac],
            dtype=np.float32,
        )

    def _out_of_bounds(self, pos):
        return (
            abs(pos[0]) > self.arena_xy
            or abs(pos[1]) > self.arena_xy
            or pos[2] < 0.05
            or pos[2] > self.arena_z
        )

    # ------------------------------------------------------------------ #
    # Gymnasium 오버라이드
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        # 매 에피소드 시작 위치를 랜덤화 (경기장 안쪽, 너무 가깝지 않게)
        while True:
            c_xy = self._np_rng.uniform(-self.arena_xy * 0.7, self.arena_xy * 0.7, size=2)
            e_xy = self._np_rng.uniform(-self.arena_xy * 0.7, self.arena_xy * 0.7, size=2)
            c_pos = np.array([c_xy[0], c_xy[1], self._np_rng.uniform(0.8, self.arena_z * 0.7)])
            e_pos = np.array([e_xy[0], e_xy[1], self._np_rng.uniform(0.8, self.arena_z * 0.7)])
            if np.linalg.norm(c_pos - e_pos) > 1.5:
                break
        self.INIT_XYZS = np.array([c_pos, e_pos])

        obs, info = super().reset(seed=seed, options=options)
        self.step_count = 0
        self.chaser_pos = self._getDroneStateVector(0)[0:3].copy()
        self.evader_pos = self._getDroneStateVector(1)[0:3].copy()
        return self._get_obs(), info

    def step(self, action):
        chaser_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        evader_action = self._evader_action()
        full_action = np.vstack([chaser_action, evader_action])

        # 부모 클래스의 step()이 실제 물리(로터 추력/토크 -> 강체 시뮬레이션)를 전부 처리함
        _, _, _, _, info = super().step(full_action)
        self.step_count += 1

        chaser_state = self._getDroneStateVector(0)
        evader_state = self._getDroneStateVector(1)
        self.chaser_pos = chaser_state[0:3].copy()
        self.evader_pos = evader_state[0:3].copy()

        dist = np.linalg.norm(self.chaser_pos - self.evader_pos)
        chaser_oob = self._out_of_bounds(self.chaser_pos)
        evader_oob = self._out_of_bounds(self.evader_pos)

        terminated = False
        truncated = False
        reward = -0.1 * dist - 0.01
        info = {}

        if dist < self.catch_radius:
            reward += 50.0
            terminated = True
            info["result"] = "caught"
        elif chaser_oob:
            reward -= 20.0
            terminated = True
            info["result"] = "chaser_out_of_bounds"
        elif evader_oob:
            reward += 5.0
            terminated = True
            info["result"] = "evader_out_of_bounds"
        elif self.step_count >= self.max_steps:
            reward -= 5.0
            truncated = True
            info["result"] = "time_up_evader_wins"

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        chaser_state = self._getDroneStateVector(0)
        evader_state = self._getDroneStateVector(1)
        chaser_pos = chaser_state[0:3]
        chaser_vel = chaser_state[10:13]
        evader_pos = evader_state[0:3]
        evader_vel = evader_state[10:13]
        rel_pos = evader_pos - chaser_pos
        rel_vel = evader_vel - chaser_vel
        return np.concatenate([chaser_pos, chaser_vel, rel_pos, rel_vel]).astype(np.float32)

    # BaseRLAviary가 내부적으로 요구하는 추상 메서드들 - 우리는 step()에서 직접
    # 보상/종료를 계산하므로 최소한의 형태로만 채워둠 (실제로 쓰이지 않음)
    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}


if __name__ == "__main__":
    env = DroneTagPyBulletEnv(seed=0)
    for ep in range(2):
        obs, _ = env.reset()
        total_r = 0.0
        for t in range(env.max_steps):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                print(f"episode {ep}: steps={t+1}, reward={total_r:.2f}, {info}")
                break
    env.close()
