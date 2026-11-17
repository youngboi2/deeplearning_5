"""LLM 추정 (담당: 김유라). 판정은 하지 않는다 — 결과는 knowledge 쪽 규칙이 판정한다.

- 연결 전·실패 시 None을 돌려준다. 앱은 None이어도 동작해야 한다.
- 키는 .env에만 (os.environ으로 읽기). 코드·출력에 쓰지 않는다.
"""
import csv
import re
from pathlib import Path

from llm.client import ask

ROOT = Path(__file__).parent.parent


def _known_names():
    """우리 표가 아는 이름 175개 (분류 모델 101종 + 새 후보 74종). LLM이 이 표기로 답하게 해서 CLIP 결과와 바로 비교."""
    names = (ROOT / 'modeling/classes.txt').read_text(encoding='utf-8').split()
    with open(ROOT / 'knowledge/candidates.csv', encoding='utf-8-sig') as f:
        names += [r['class'] for r in csv.DictReader(f)]
    return sorted(set(names))


KNOWN = _known_names()


def _snake(name):
    """'Pad Thai' → 'pad_thai' (표 이름 표기)"""
    return re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')


def guess_name(img):
    """PIL 사진 → [(영어 음식 이름, 0~1 확신)] 확신 높은 순, 또는 None.
    2단계에서 CLIP 검색 결과와 교차 확인용 (search.py가 사용).
    이름은 가능하면 표 이름(KNOWN)과 같은 표기 (예: 'pad_thai'). 목록에 없으면 영어 snake_case."""
    prompt = (
        'Look at this photo of a dish and guess what food it is.\n'
        'Return JSON only: {"guesses": [{"name": "...", "confidence": 0.0}]} with up to 3 guesses, '
        'most likely first. confidence is a probability between 0 and 1.\n'
        'If a guess matches one of these known names, use exactly that name: ' + ', '.join(KNOWN) + '.\n'
        'Otherwise use a short English name in snake_case (e.g. "khao_soi"). '
        'If the photo is not food, return {"guesses": []}.'
    )
    out = ask(prompt, img=img, want_json=True)
    if not isinstance(out, dict) or not isinstance(out.get('guesses'), list):
        return None
    res = []
    for g in out['guesses'][:3]:
        try:
            name, conf = _snake(g['name']), min(max(float(g['confidence']), 0.0), 1.0)
        except (KeyError, TypeError, ValueError):
            continue
        if name:
            res.append((name, conf))
    return sorted(res, key=lambda x: -x[1]) or None


def guess_ingredients(food_name):
    """음식 이름 → 영어 재료 목록 ['flour', 'egg', ...] 또는 None.
    레시피가 아예 없는 음식만 사용. 결과는 rules.allergen_flags로 판정하고 '추정' 표시.
    숨은 재료(육수·소스·반죽·기름)도 빠뜨리지 않게 물어본다 — 알레르기는 놓치는 쪽이 더 위험."""
    name = str(food_name).replace('_', ' ').strip()
    if not name:
        return None
    prompt = (
        f'List the ingredients usually used in "{name}", as cooked in typical recipes.\n'
        'Include hidden ingredients: broth or stock, sauces (e.g. soy sauce, fish sauce, oyster sauce), '
        'batter or dough, cooking fat, garnishes, nuts and seeds.\n'
        'Use simple lowercase English ingredient words (e.g. "peanuts", "shrimp", "wheat flour", "egg").\n'
        'Return JSON only: {"ingredients": ["...", "..."]}. If you do not know this dish, return {"ingredients": []}.'
    )
    out = ask(prompt, want_json=True)
    if not isinstance(out, dict) or not isinstance(out.get('ingredients'), list):
        return None
    items = [str(x).strip().lower() for x in out['ingredients'] if str(x).strip()]
    return items or None
