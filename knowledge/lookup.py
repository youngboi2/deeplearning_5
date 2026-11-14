"""클래스 이름 → 알레르기·영양 정보 (담당: 룰베이스·LLM).

판정은 여기서만 한다. LLM은 이 결과를 설명만 한다(llm/explain.py).
- food_table.csv : 음식별 알레르기 포함 비율 + 레시피 수 (EDA v2 결과, 1M_Final 레시피 통계)
                   영양 칸(kcal, protein_g, fat_g, carbs_g, fiber_g, sodium_mg = 1인분 중앙값)은
                   build_table.py 를 Colab에서 돌리면 추가됨
- class_map.csv  : 모델 클래스 이름이 표 이름과 다를 때만 한 줄 추가 (새 클래스 온보딩 때)
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
MIN_RECIPES = 30        # 이보다 적으면 '정보 부족'
HIGH, MAYBE = 0.5, 0.1  # ponytail: 알레르기 공통 고정 임계값. 사람 정답지로 검증 후 항목별로 조정
NUTRI = ['kcal', 'protein_g', 'fat_g', 'carbs_g', 'fiber_g', 'sodium_mg']
NOT_ALLERGEN = {'class', 'n_recipes', *NUTRI}


def _read(name):
    with open(HERE / name, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


_TABLE = {r['class']: r for r in _read('food_table.csv')}
_MAP = {r['class']: r for r in _read('class_map.csv')}


def lookup(cls):
    """반환: {status: '표 있음'|'정보 부족'|'정보 없음', allergens: [(이름, 비율, '높음'|'가능')],
             nutrition: {칸: 값}, n_recipes, source}"""
    m = _MAP.get(cls, {})
    row = _TABLE.get(m.get('table_key') or cls)
    if row is None:
        return dict(status='정보 없음', allergens=[], nutrition={}, n_recipes=0, source=m.get('source', ''))
    n = int(row['n_recipes'])
    found = sorted(((k, float(v)) for k, v in row.items() if k not in NOT_ALLERGEN and float(v) >= MAYBE),
                   key=lambda x: -x[1])
    return dict(status='정보 부족' if n < MIN_RECIPES else '표 있음',
                allergens=[(k, v, '높음' if v >= HIGH else '가능') for k, v in found],
                nutrition={k: float(row[k]) for k in NUTRI if row.get(k)},
                n_recipes=n, source=m.get('source') or '1M_Final 레시피 통계')


if __name__ == '__main__':
    assert ('우유', 0.7487, '높음') in lookup('apple_pie')['allergens']
    assert lookup('takoyaki')['status'] == '정보 부족'
    assert lookup('없는음식')['status'] == '정보 없음'
    print('lookup OK')
