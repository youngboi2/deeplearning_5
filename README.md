# 찍먹 🍽️ — 사진 한 장으로 알레르기·영양 정보 알려주는 서비스

해외여행 중 처음 보는 음식을 찍으면 무슨 음식인지 맞히고, 그 음식에 흔히 들어가는 알레르기 재료와 영양 정보를 알려준다.

## 한눈에 보는 흐름

```
① 사진 찍기
   ↓
② 분류 모델이 음식 이름 후보 3개를 맞힘            ← modeling/   (한솔)
   ↓
③ 룰베이스 표에서 그 음식의 알레르기·영양 정보를 찾음 ← knowledge/  (중현)
   ↓
④ LLM이 결과를 쉬운 말로 설명 (선택)                ← llm/        (중현)
   ↓
⑤ 사용자가 "맞아요/틀려요/목록에 없어요" 피드백
   ↓
⑥ 피드백 사진을 모아 다시 학습, 새 음식도 추가     ← trigger/    (유라, MLOps)
```

화면(①~⑤)은 `streamlit_app/` 하나에서 모두 처리한다 (유라).

## 역할 분담 (제안 — 팀 회의에서 확정)

| 담당 | 폴더 | 하는 일 |
|---|---|---|
| **유라** · MLOps | `streamlit_app/`, `trigger/`, `share_storage/` | 화면, 피드백 저장, 재학습·승격 자동화, 학과 서버 배포 |
| **한솔** · 모델 | `modeling/` | 분류 모델 학습 코드·가중치, `predict()` 함수 맞추기, 잊어버림 실험 |
| **중현** · 룰베이스·LLM | `knowledge/`, `llm/` | 음식별 알레르기·영양 표, 경고 판정 규칙, LLM 설명 |
| **서준영** · 데이터·평가 | `evaluation/` (만들 예정) | 데이터 분할, 사용자 사진 평가셋, 알레르기 정답지로 경고 정확도 검증 |

## 폴더 설명

```
deeplearning_5/
├─ streamlit_app/app.py     화면 (사진 → 결과 → 경고 → 피드백)
├─ modeling/model.py        모델 불러오기·예측 (학습 코드는 한솔님 코드가 여기로)
├─ trigger/retrain.py       재학습 (아직 비어 있음, 노션 규칙만 적어 둠)
├─ knowledge/               룰베이스
│  ├─ rules.py              음식 이름↔레시피 매칭 규칙 + 재료→알레르기 19항목 규칙
│  ├─ build_table.py        1M_Final 레시피로 음식별 표를 만드는 스크립트 (Colab에서 실행)
│  ├─ food_table.csv        결과 표: 101종 × 알레르기 비율 (+ 영양 중앙값 예정) + 레시피 수
│  ├─ class_map.csv         새 음식 이름이 표 이름과 다를 때 연결해 주는 표 (지금은 비어 있음)
│  └─ lookup.py             음식 이름 → 경고 정보 (판정은 여기서만)
├─ llm/explain.py           판정 결과를 쉬운 말로 설명 (LLM 아직 미연결)
├─ notebooks/               01 EDA·전처리(사진·레시피 확인, 알레르기·나트륨), 02 새 음식 후보 집계
├─ share_storage/           사진·기록·모델 보관 (깃에는 폴더 모양만)
│  ├─ pending/   아직 검토 안 한 피드백 사진
│  ├─ data/      승인된 피드백 사진
│  ├─ backupdata/ 지난 회차 피드백
│  ├─ metadata/  uploads.csv (누가 언제 무슨 사진에 어떤 정답을 줬는지)
│  ├─ model/     best.pt(깃 제외), classes.txt(음식 목록), version.json
│  └─ metrics/   회차별 평가 결과
└─ docs/plan.md             단계별 할 일
```

`share_storage/` 구조는 노션 「사용자 피드백으로 계속 배우는 음식 분류 서비스」, 폴더 이름은 유라님 [docker-mlops-pipeline](https://github.com/yura103/docker-mlops-pipeline)을 따랐다.

## 서로 지켜야 할 약속 (이것만 맞으면 각자 따로 작업 가능)

```python
predict(model, img)   -> [("ramen", 0.82), ("pho", 0.10), ("udon", 0.03)]       # modeling/model.py
lookup("ramen")       -> {"status": "표 있음",                                   # knowledge/lookup.py
                          "allergens": [("밀", 0.81, "높음"), ("대두", 0.42, "가능")],
                          "nutrition": {"kcal": 450.0, "sodium_mg": 1800.0},
                          "n_recipes": 1520, "source": "1M_Final 레시피 통계"}
explain("ramen", info) -> "라멘은 보통 밀 면을 써서 ..." 또는 None                # llm/explain.py
```

- **경고 판정은 `lookup`만 한다. LLM은 설명만 한다.** (LLM이 "들어있지 않아요"라고 지어내는 걸 막기 위해)
- 어떤 화면에서도 **"안전하다"고 표시하지 않는다.** 항상 "가게에 확인하세요".
- 레시피 30개 미만 = `정보 부족`, 표에 없는 음식 = `정보 없음` → 둘 다 확인 안내.

## 실행

```bash
pip install -r requirements.txt
# 모델 가중치(best.pt)를 share_storage/model/ 에 복사한 뒤
streamlit run streamlit_app/app.py
```

## 깃에 올리지 않는 것
- 사진, 1M_Final.csv, 모델 가중치(.pt) → 학과 서버 `food101/` 또는 드라이브에 보관
- 서버 계정·비밀번호·API 키 → `.env`, `.streamlit/secrets.toml` (gitignore 되어 있음)

## 데이터 출처
- Food-101 (Bossard et al., ECCV 2014) — 학술 이용
- 1M_Final (Recipe1M+ 파생, Kaggle) — 연구용, 재배포 금지 (표에는 통계만 담음)
- 알레르기 기준: 「식품 등의 표시·광고에 관한 법률 시행규칙」 [별표 2] 19개 항목
