#!/home/junhyun/miniconda3/envs/deepsee/bin/python3
"""
hds_input.py — HDS 실시간 상황 입력 인터페이스

별도 터미널에서 실행:
  python3 ~/SEESys/DeepSEE/Training/hds_input.py

입력한 상황이 LLM을 거쳐 HDS 점수에 즉시 반영됩니다.
"""
from __future__ import annotations
import os, sys, json, time

CONDITION_FILE = "/tmp/hds_user_condition.json"

EXAMPLES = [
    "어두운 복도로 진입 중",
    "카메라 빠르게 이동 중",
    "텍스처 없는 흰 벽 앞",
    "역광 환경 (창문 방향)",
    "카메라 일부 가려짐",
    "좁은 공간에서 급격한 회전",
    "low-texture environment",
    "rapid rotation causing motion blur",
]

def write_condition(cond: str | None, delta_override: float | None = None):
    data = {
        "condition": cond,
        "timestamp": time.time(),
        "delta_override": delta_override,  # None이면 LLM 자동 계산
    }
    with open(CONDITION_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False)

def main():
    print("=" * 55)
    print("  HDS 실시간 상황 입력  (live_hds_demo.py 연동)")
    print("=" * 55)
    print("현재 SLAM 환경 상황을 자유롭게 입력하세요.")
    print("입력된 텍스트 → LLM → risk_delta → HDS 점수 반영\n")
    print("예시 상황:")
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"  {i}. {ex}")
    print()
    print("명령어:")
    print("  숫자 (1-8) : 예시 상황 선택")
    print("  clear / 0  : 상황 초기화 (자동 감지로 복귀)")
    print("  +0.5       : 델타값 직접 지정 (0.0 ~ 1.0)")
    print("  Ctrl+C     : 종료")
    print("-" * 55)
    print()

    # 시작 시 초기화
    write_condition(None)

    while True:
        try:
            raw = input("상황 입력 > ").strip()
            if not raw:
                continue

            # 숫자 입력 → 예시 선택
            if raw.isdigit():
                idx = int(raw)
                if idx == 0:
                    write_condition(None)
                    print("  → [초기화] 자동 감지 모드로 복귀\n")
                elif 1 <= idx <= len(EXAMPLES):
                    cond = EXAMPLES[idx - 1]
                    write_condition(cond)
                    print(f"  → 전송: '{cond}'\n")
                else:
                    print(f"  ⚠  1-{len(EXAMPLES)} 사이 숫자를 입력하세요\n")
                continue

            # clear / 0
            if raw.lower() in ("clear", "0", "없음", "정상"):
                write_condition(None)
                print("  → [초기화] 자동 감지 모드로 복귀\n")
                continue

            # +숫자: 델타 직접 지정
            if raw.startswith("+") or raw.startswith("-"):
                try:
                    delta = float(raw)
                    delta = max(0.0, min(1.0, delta))
                    write_condition("manual delta override", delta_override=delta)
                    print(f"  → 델타 직접 설정: {delta:.2f}\n")
                except ValueError:
                    print("  ⚠  형식 오류. 예: +0.5\n")
                continue

            # 자유 텍스트 → LLM 위임
            write_condition(raw)
            print(f"  → 전송: '{raw}' (LLM이 risk_delta 계산 중...)\n")

        except (KeyboardInterrupt, EOFError):
            write_condition(None)
            print("\n[hds_input] 종료 — 조건 초기화됨")
            break

if __name__ == "__main__":
    main()
