"""
drone_tag_env.py

FPV 드론 술래잡기(Tag) 강화학습 환경 - Phase 1
- 술래(chaser)는 RL 에이전트, 도망자(evader)는 규칙기반(rule-based)
- 경량 점질량(point-mass) 동역학: 가속도를 액션으로 받아 속도/위치를 적분
- 나중에 gym-pybullet-drones 같은 실제 물리엔진으로 교체하기 쉽도록
  관측값/리워드/종료조건 인터페이스를 최대한 깔끔하게 분리해둠

의존성: gymnasium, numpy
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DroneTagEnv(gym.Env):
    """
    술래(chaser) 1대를 RL로 학습시키는 환경.
    도망자(evader)는 규칙기반 정책으로 움직임.

    관측값 (12차원):
        [chaser_pos(3), chaser_vel(3), rel_pos_to_evader(3), rel_vel_to_evader(3)]

    액션 (3차원):
        chaser에게 가할 가속도 명령 (x, y, z), [-1, 1] 범위로 정규화됨

    좌표계: 경기장은 원점 중심의 박스
        x, y: [-ARENA_XY, ARENA_XY]
        z(고도): [0, ARENA_Z]  (바닥 아래로는 못 감)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        arena_xy: float = 15.0,      # 경기장 x,y 반경 (총 30m x 30m)
        arena_z: float = 30.0,       # 경기장 최대 고도 (30m)
        max_accel: float = 4.0,      # 최대 가속도 (m/s^2)
        max_speed: float = 3.0,      # 최대 속도 (m/s)
        drag: float = 0.25,          # 공기저항 계수 (속도에 비례한 감쇠)
        catch_radius: float = 0.3,   # 이 거리 이내면 catch
        dt: float = 0.05,            # 스텝당 시간(s) - 20Hz 제어
        max_steps: int = 1200,       # 60초 제한시간 (0.05 * 1200) - 경기장이 커져서 시간 늘림
        evader_speed: float = 2.2,   # 도망자 최고 속도 (술래보다 살짝 느리게 시작)
        seed: int | None = None,
    ):
        super().__init__()
        self.arena_xy = arena_xy
        self.arena_z = arena_z
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.drag = drag
        self.catch_radius = catch_radius
        self.dt = dt
        self.max_steps = max_steps
        self.evader_speed = evader_speed

        # 관측값: 12차원, 대략적인 범위로 정규화 없이 raw 값 사용 (필요시 나중에 정규화 추가)
        high = np.array([np.inf] * 12, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # 내부 유틸
    # ------------------------------------------------------------------ #
    def _random_pos(self):
        xy = self._rng.uniform(-self.arena_xy * 0.8, self.arena_xy * 0.8, size=2)
        z = self._rng.uniform(0.5, self.arena_z * 0.8)
        return np.array([xy[0], xy[1], z], dtype=np.float32)

    def _out_of_bounds(self, pos):
        return (
            abs(pos[0]) > self.arena_xy
            or abs(pos[1]) > self.arena_xy
            or pos[2] < 0.0
            or pos[2] > self.arena_z
        )

    def _clip_speed(self, vel, max_speed):
        speed = np.linalg.norm(vel)
        if speed > max_speed:
            vel = vel / speed * max_speed
        return vel

    def _evader_policy(self):
        """
        규칙기반 도망자:
        - 기본적으로 술래 반대 방향으로 도망
        - 경계에 가까워지면 중심 쪽으로 방향 보정 (구석에 몰리지 않게)
        - 약간의 랜덤 지터를 섞어서 예측 불가능하게
        """
        away = self.evader_pos - self.chaser_pos
        dist = np.linalg.norm(away) + 1e-6
        away_dir = away / dist

        # 경계 회피: 중심 방향 벡터를 섞어줌
        to_center = -self.evader_pos.copy()
        to_center[2] = (self.arena_z / 2) - self.evader_pos[2]
        center_norm = np.linalg.norm(to_center) + 1e-6
        to_center_dir = to_center / center_norm

        # 경계에 가까울수록 to_center 비중을 높임
        edge_dist = min(
            self.arena_xy - abs(self.evader_pos[0]),
            self.arena_xy - abs(self.evader_pos[1]),
            self.evader_pos[2],
            self.arena_z - self.evader_pos[2],
        )
        edge_urgency = np.clip(1.0 - edge_dist / 0.8, 0.0, 1.0)  # 0.8m 이내면 긴급

        jitter = self._rng.normal(0, 0.3, size=3)
        direction = (1 - edge_urgency) * away_dir + edge_urgency * to_center_dir + jitter
        norm = np.linalg.norm(direction) + 1e-6
        direction = direction / norm

        target_vel = direction * self.evader_speed
        accel = (target_vel - self.evader_vel) * 3.0  # 단순 P 제어로 가속도 산출
        accel = np.clip(accel, -self.max_accel, self.max_accel)
        return accel.astype(np.float32)

    def _get_obs(self):
        rel_pos = self.evader_pos - self.chaser_pos
        rel_vel = self.evader_vel - self.chaser_vel
        return np.concatenate(
            [self.chaser_pos, self.chaser_vel, rel_pos, rel_vel]
        ).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 술래/도망자가 너무 가까이서 시작하지 않도록 거리 보장
        while True:
            self.chaser_pos = self._random_pos()
            self.evader_pos = self._random_pos()
            if np.linalg.norm(self.chaser_pos - self.evader_pos) > 1.5:
                break

        self.chaser_vel = np.zeros(3, dtype=np.float32)
        self.evader_vel = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        chaser_accel = action * self.max_accel

        # 술래 동역학 적분
        self.chaser_vel += (chaser_accel - self.drag * self.chaser_vel) * self.dt
        self.chaser_vel = self._clip_speed(self.chaser_vel, self.max_speed)
        self.chaser_pos = self.chaser_pos + self.chaser_vel * self.dt

        # 도망자 동역학 적분 (규칙기반)
        evader_accel = self._evader_policy()
        self.evader_vel += (evader_accel - self.drag * self.evader_vel) * self.dt
        self.evader_vel = self._clip_speed(self.evader_vel, self.evader_speed)
        self.evader_pos = self.evader_pos + self.evader_vel * self.dt

        self.step_count += 1

        dist = np.linalg.norm(self.chaser_pos - self.evader_pos)
        chaser_oob = self._out_of_bounds(self.chaser_pos)
        evader_oob = self._out_of_bounds(self.evader_pos)

        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        # 리워드: 거리를 좁힐수록 + (이전 스텝 대비 개선량 기반이 더 안정적이지만 우선 단순화)
        reward += -0.1 * dist          # 매 스텝 거리에 비례한 페널티(가까울수록 덜 깎임)
        reward += -0.01                # 시간 페널티 (빨리 잡을수록 유리하게)

        if dist < self.catch_radius:
            reward += 50.0
            terminated = True
            info["result"] = "caught"
        elif chaser_oob:
            reward -= 20.0
            terminated = True
            info["result"] = "chaser_out_of_bounds"
        elif evader_oob:
            # 도망자가 실수로 나가면 술래에게도 약한 보너스만 (직접 잡은 게 아니므로)
            reward += 5.0
            terminated = True
            info["result"] = "evader_out_of_bounds"
        elif self.step_count >= self.max_steps:
            reward -= 5.0
            truncated = True
            info["result"] = "time_up_evader_wins"

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # 환경이 정상 동작하는지 간단히 확인 (랜덤 액션으로 몇 에피소드 굴려보기)
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
