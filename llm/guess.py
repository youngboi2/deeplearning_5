"""LLM 추정 (담당: 김유라). 판정은 하지 않는다 — 결과는 knowledge 쪽 규칙이 판정한다.

- 연결 전·실패 시 None을 돌려준다. 앱은 None이어도 동작해야 한다.
- 키는 .env에만 (os.environ으로 읽기). 코드·출력에 쓰지 않는다.
"""


def guess_name(img):
    """PIL 사진 → [(영어 음식 이름, 0~1 확신)] 확신 높은 순, 또는 None.
    2단계에서 CLIP 검색 결과와 교차 확인용 (search.py가 사용)."""
    return None


def guess_ingredients(food_name):
    """음식 이름 → 영어 재료 목록 ['flour', 'egg', ...] 또는 None.
    레시피가 아예 없는 음식만 사용. 결과는 rules.allergen_flags로 판정하고 '추정' 표시."""
    return None
