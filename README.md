# drone-tag-rl

FPV 드론 술래잡기(Tag) 강화학습 프로젝트. Colab에서 어디서든 학습하고 GitHub에 계속 반영하는 걸 목표로 함.

## 규칙

- **경기장**: 5m × 5m × 3m 박스형 공중 구역 1개
- **드론**: 술래(chaser) 1대 + 도망자(evader) 1대
- **승리 조건**
  - 술래가 도망자와 거리 0.3m 이내 접근 → 술래 승 (catch)
  - 제한시간(300 step = 15초, 20Hz 제어) 내 못 잡으면 → 도망자 승
  - 경기장 밖으로 나가면 해당 드론 즉시 페널티
- **리워드**: 거리 기반 shaping + catch/out-of-bounds/time-up 이벤트 보너스·페널티
- 자세한 리워드 수식은 `drone_tag_env.py`의 `step()` 참고

## 학습 단계

- **Phase 1 (현재)**: 도망자는 규칙기반(추격자 반대방향 + 경계 회피), 술래만 PPO로 학습
- **Phase 2 (다음)**: 술래가 어느 정도 잡으면 도망자도 PPO로 바꿔서 self-play

## 구조

```
index.py           # 메인 진입점 (train / evaluate / demo 서브커맨드)
drone_tag_env.py    # Gymnasium 환경 (점질량 동역학, Phase 1: 도망자 규칙기반)
train_phase1.py     # (index.py train과 동일 로직의 단독 실행 버전, 참고용)
evaluate.py          # (index.py evaluate와 동일 로직의 단독 실행 버전, 참고용)
```

## 실행 (Codespaces 기준)

```bash
pip install gymnasium stable-baselines3 numpy matplotlib

# 환경 동작만 빠르게 확인 (학습 없음)
python index.py demo

# 학습 (기본 30만 스텝, ~10-20분 소요, GPU 없어도 됨 - MLP라 CPU로 충분)
python index.py train --timesteps 300000 --out models/chaser_phase1.zip

# 평가 + 궤적 시각화
python index.py evaluate --model models/chaser_phase1.zip --episodes 50
```

## GitHub Codespaces에서 작업하기

1. 레포 페이지 → `Code` 버튼 → `Codespaces` 탭 → `Create codespace on main`
2. 브라우저 안에서 VS Code + 터미널 바로 사용 가능, 깃도 이미 연결돼있어서 토큰 설정 불필요
3. 작업 끝나면 좌측 소스컨트롤 탭에서 커밋/푸시

## 다음 단계 (Phase 2 예정)

- 도망자도 PPO로 학습 → self-play (상대 정책을 주기적으로 고정/교체하는 opponent pool 방식 추천)
- 잘 되면 gym-pybullet-drones 같은 실제 물리엔진으로 업그레이드해서 sim-to-real 준비
