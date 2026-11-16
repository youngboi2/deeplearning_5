"""음식 이름 → '알레르기 비율 줄' (1단계 룰베이스 표 찾기).

- food_table.csv : 음식별 알레르기 비율(19항목) + 레시피 수 (+ 영양 중앙값, build_table.py로 추가)
- class_map.csv  : 이름이 표 이름과 다를 때만 연결 (예: 김치전 → 표의 다른 이름)

비율 줄 모양 (2단계 RAG도 같은 모양을 만든다):
    {'밀': 0.81, '대두': 0.42, ... 19개, 'n_recipes': 1520, 'nutrition': {'kcal': 450.0, ...}, 'source': '레시피 통계'}
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
NUTRI = ['kcal', 'protein_g', 'fat_g', 'carbs_g', 'fiber_g', 'sodium_mg']


def _read(name):
    with open(HERE / name, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


_TABLE = {r['class']: r for r in _read('food_table.csv')}
_MAP = {r['class']: r for r in _read('class_map.csv')}
ALLERGENS = [k for k in next(iter(_TABLE.values())) if k not in ('class', 'n_recipes', *NUTRI)]  # 표 머리글 순서


def row(name):
    """표에 있으면 비율 줄, 없으면 None (→ 2단계)."""
    m = _MAP.get(name, {})
    r = _TABLE.get(m.get('table_key') or name)
    if r is None:
        return None
    out = {a: float(r[a]) for a in ALLERGENS}
    out['n_recipes'] = int(r['n_recipes'])
    out['nutrition'] = {k: float(r[k]) for k in NUTRI if r.get(k)}
    out['source'] = m.get('source') or '레시피 통계'
    return out
