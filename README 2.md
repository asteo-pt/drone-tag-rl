# drone-tag-rl

FPV 드론 술래잡기(Tag) 강화학습 프로젝트.
실제 물리(4개 로터 추력/토크, PyBullet 강체 시뮬레이션) 기반으로 술래 드론을 PPO로 학습시킴.
최종 목표는 커스텀 FPV 드론(Betaflight 펌웨어 + 컴패니언 컴퓨터)에 실제로 탑재하는 것.

## 규칙

- **경기장**: 15m × 15m × 3m 박스형 공중 구역
- **드론**: 술래(chaser) 1대(RL 학습) + 도망자(evader) 1대(규칙기반)
- **승리 조건**
  - 술래가 도망자와 거리 0.3m 이내로 정확히 충돌 → 술래 승 (catch, 둘 다 파괴되는 걸로 간주해 즉시 에피소드 종료)
  - 제한시간(20초) 내 못 잡으면 → 도망자 승
  - 경기장 밖으로 나가면 해당 드론 페널티
- 리워드/판정 로직은 `drone_tag_env_pybullet.py`의 `step()` 참고

## 물리 / 하드웨어 방향

- 시뮬레이터: [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) — 로터 추력·토크를 실제로 계산하는 PyBullet 기반 쿼드콥터 시뮬
- 액션: VEL 타입(방향+속력) — 실제 Betaflight의 자세제어 PID에 넘길 velocity setpoint와 인터페이스가 유사해서, 나중에 실제 하드웨어로 옮길 때 액션 구조를 크게 안 바꿔도 되도록 설계함
- 아직 안 된 것: 실제 드론 스펙(5인치 FPV 프리스타일급)으로 URDF 커스터마이징, 실기체 상태추정(비전/텔레메트리) 설계

## 실행 (Colab)

전부 `drone_tag_colab.ipynb` 하나로 돌아감. 터미널 필요 없이 노트북 열고 위에서부터 칸(▶)을 순서대로 누르면 됨.

1. 라이브러리 설치 (gymnasium, stable-baselines3, gym-pybullet-drones)
2. 환경 코드 저장
3. 랜덤 액션으로 환경 동작 확인 (에러 나면 여기서 잡힘)
4. PPO 학습 — **반복 실행 가능**: 누를 때마다 저장된 모델을 이어서 추가 학습함, Google Drive에 자동 백업
5. 평가 — 결과 궤적을 `results.html`(독립형 애니메이션 뷰어)로 생성
6. (선택) GitHub Pages에 결과 자동 반영

## GitHub Pages로 실시간 결과 보기

**최초 1회 설정**: 레포 `Settings → Pages → Source: Deploy from a branch → Branch: main, 폴더: /docs`

그 다음부터는 노트북 마지막 칸에 본인 레포 주소 + GitHub 토큰만 채워서 실행하면 `results.html`이 `docs/index.html`로 자동 반영됨.
`https://<username>.github.io/<repo>/` 새로고침하면 항상 최신 결과 확인 가능.

## 다음 단계

- 드론 스펙을 5인치 FPV 프리스타일급으로 URDF 커스터마이징
- 술래가 어느 정도 잘 잡으면 도망자도 PPO로 바꿔서 self-play
- 실기체 상태추정(비전 기반 상대 위치 탐지 or 텔레메트리 공유) 설계
