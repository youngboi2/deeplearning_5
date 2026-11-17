"""나트륨 '높음' 기준 고르기 (담당: 서준영)

knowledge/food_table.csv의 1인분 나트륨(sodium_mg, 레시피 중앙값)을 하루 기준 2,000mg과 비교해
기준 후보마다 몇 종이 '높음'이 되는지 센다. 결정과 이유는 evaluation/나트륨_기준.md.
실행 (저장소 루트):  python -m evaluation.sodium_threshold
"""
import csv
import statistics
from pathlib import Path

TABLE = Path(__file__).parent.parent / 'knowledge' / 'food_table.csv'
DAY = 2000        # 하루 기준 (mg): 식품 영양표시 1일 기준치, WHO 권고(하루 2,000mg 미만)
PICK = 1 / 2      # 정한 기준. frontend/index.html의 SODIUM_HIGH와 같아야 한다
CANDIDATES = [('한 끼 몫 (하루의 1/3)', 1 / 3), ('절반 (하루의 1/2)', 1 / 2), ('하루치 (하루의 100%)', 1.0)]
FEW = 30          # knowledge/judge.py MIN_RECIPES와 같은 값. 이보다 적으면 값이 흔들림


def load():
    with open(TABLE, encoding='utf-8-sig') as f:
        return [(r['class'], float(r['sodium_mg']), int(r['n_recipes'])) for r in csv.DictReader(f) if r.get('sodium_mg')]


def main():
    foods = load()
    mg = [v for _, v, _ in foods]
    n = len(foods)
    print(f'음식 {n}종 · 1인분 나트륨 중앙값 {statistics.median(mg):,.0f}mg '
          f'(가장 적음 {min(mg):,.0f}, 가장 많음 {max(mg):,.0f})\n')

    print('| 기준 | 1인분 | 높음 | 비율 |')
    print('|---|---|---|---|')
    for name, k in CANDIDATES:
        hi = sum(v >= DAY * k for v in mg)
        mark = ' ← 선택' if k == PICK else ''
        print(f'| {name}{mark} | {DAY * k:,.0f}mg 이상 | {hi}종 | {hi / n:.0%} |')

    high = sorted((f for f in foods if f[1] >= DAY * PICK), key=lambda f: -f[1])
    print(f'\n선택한 기준({DAY * PICK:,.0f}mg 이상)으로 높음 {len(high)}종, 많은 순:')
    print(', '.join(f'{c} {v:,.0f}' + (f'(레시피 {r}개)' if r < FEW else '') for c, v, r in high))


if __name__ == '__main__':
    main()
