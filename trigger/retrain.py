"""재학습 (담당: MLOps). 노션 '사용자 피드백으로 계속 배우는 음식 분류 서비스' 2장 규칙을 따른다.

1. metadata/uploads.csv + pending/ 검토 → 승인분은 data/, 처리 끝난 회차는 backupdata/
2. 학습 = 기존 Food-101 Train 일부 + 과거 피드백 Train 일부 + 새 피드백 Train (매 회차 같은 코드)
   Val/Test 사진은 절대 학습에 쓰지 않음, 증강은 Train 읽을 때만
3. 새 음식이 승인되면 model/classes.txt 끝에 추가 → 출력층 크기 증가
   + knowledge/class_map.csv 에 행 추가 요청(룰베이스 담당)
4. 평가 → 승격 기준 통과 시 model/best.pt 교체, model/version.json +1, metrics/history.csv 기록
   승격 기준(안): 기존 음식 정확도 하락 1%p 이내
"""

if __name__ == '__main__':
    raise SystemExit('아직 구현 전 (MLOps 담당)')
