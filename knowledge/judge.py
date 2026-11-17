"""판정: 모델 후보 + 내 알레르기 → 🔴 주의 / 🟡 확인 필요 / ⚪ 해당 없음.

흐름 (README '우리 내부 흐름'):
  decide()  : 1등 확률로 1단계(1등만 / 후보 합치기) 또는 2단계(처음 보는 음식) 결정
  merge()   : 후보 여러 개의 비율 줄을 '가장 위험한 쪽'으로 합침
  judge()   : 비율 줄 + 내 알레르기 → 항목별 단계
  analyze() : 위를 순서대로. 처음 보는 음식은 stage2(= CLIP 검색 + LLM 교차 확인) 결과를 같은 judge()로 판정.
              애매한 1단계(합치기)도 stage2 결과를 후보와 함께 '가장 위험한 쪽'으로 합침 (착각 대비)

자체 검사: 저장소 루트에서  python -m knowledge.judge
"""
from knowledge.lookup import ALLERGENS, row

# EDA v2 임시 기준. 정답지·검증 데이터로 다시 정한다 (docs/plan.md 6, 9번)
CAUTION, CHECK = 0.6, 0.2   # 레시피 비율 60% 이상 주의, 20% 이상 확인 필요
MIN_RECIPES = 30            # 이보다 적으면 '해당 없음' 대신 '정보 부족'
T_ONE, T_KNOWN = 0.95, 0.6  # 보정된(T) 1등 확률 ≥0.95 → 1등만 / ≥0.6 → 상위 3개 합치기(+CLIP) / 그 아래 → 2단계. 근거: notebooks/05 (Base), docs/진행상황.md 12


def decide(cands):
    """cands: [(음식, 확률)] 확률 높은 순 → 'one' | 'merge' | 'unknown'"""
    p1 = cands[0][1]
    return 'one' if p1 >= T_ONE else 'merge' if p1 >= T_KNOWN else 'unknown'


def merge(rows):
    """후보 중 하나라도 높으면 높게 (위험을 놓치지 않는 쪽). 레시피 수는 가장 적은 쪽."""
    out = {a: max(r[a] for r in rows) for a in ALLERGENS}
    out['n_recipes'] = min(r['n_recipes'] for r in rows)
    out['nutrition'] = rows[0]['nutrition'] if len(rows) == 1 else {}  # 섞인 음식의 영양은 의미 없음
    out['source'] = ' / '.join(sorted({r['source'] for r in rows}))
    return out


def level(v):
    return '주의' if v >= CAUTION else '확인 필요' if v >= CHECK else '해당 없음'


def judge(r, my_allergens):
    """r: 비율 줄. 반환: {verdict: {알레르기: 단계}, other_caution: [...], n_recipes, nutrition, source}"""
    bad = [a for a in my_allergens if a not in ALLERGENS]
    if bad:
        raise ValueError(f'모르는 알레르기 항목: {bad} (가능: {ALLERGENS})')
    lack = r['n_recipes'] < MIN_RECIPES
    verdict = {}
    for a in my_allergens:
        lv = level(r[a])
        verdict[a] = '정보 부족' if lack and lv == '해당 없음' else lv  # 근거가 적으면 '없다'고 말하지 않음
    return dict(verdict=verdict,
                other_caution=[a for a in ALLERGENS if a not in my_allergens and r[a] >= CAUTION],
                n_recipes=r['n_recipes'], nutrition=r['nutrition'], source=r['source'])


def analyze(cands, my_allergens, stage2=None):
    """모델 후보 → 판정.
    stage2: 처음 보는 음식일 때 부르는 함수 () -> (비율 줄, [(음식, 점수)]) — 백엔드가 search.search(사진)를 감싸 넘김.
    없거나(검색 목록 없음) None을 돌려주면 '확인 필요'로 돌려줌."""
    route = decide(cands)
    use = cands[:1] if route == 'one' else cands[:3]
    rows = [r for c, _ in use if (r := row(c))] if route != 'unknown' else []
    if not rows:  # 처음 보는 음식 → 2단계 (CLIP 검색, 같은 규칙으로 판정)
        hit = stage2() if stage2 else None
        if hit:
            r, foods = hit
            out = judge(r, my_allergens)
            names = [c for c, _ in foods[:3]]
            out.update(route='2단계', foods=names, similar=names)
            return out
        return dict(route='2단계', foods=[], similar=[], verdict={a: '확인 필요' for a in my_allergens},
                    other_caution=[], n_recipes=0, nutrition={}, source='정보 없음 (검색 목록 없음)')
    similar = []
    if route == 'merge' and stage2:  # 애매함 = 처음 보는 음식을 착각했을 수도 → CLIP 검색 결과도 같이 합침 (06: 카오소이→치킨 커리 0.61)
        hit = stage2()
        if hit:
            rows.append(hit[0])
            similar = [c for c, _ in hit[1][:3]]
    out = judge(merge(rows), my_allergens)
    out.update(route='1단계-1등' if route == 'one' else '1단계-합치기' + ('+CLIP' if similar else ''),
               foods=[c for c, _ in use], similar=similar)
    return out


if __name__ == '__main__':
    r = analyze([('ramen', 0.97), ('pho', 0.05), ('udon', 0.01)], ['밀', '땅콩'])
    assert r['route'] == '1단계-1등' and r['foods'] == ['ramen'] and r['verdict']['밀'] in ('주의', '확인 필요')

    # 라멘(밀 높음) ↔ 쌀국수: 애매하면 합쳐서 밀을 놓치지 않아야 함
    m = analyze([('pho', 0.65), ('ramen', 0.3), ('pad_thai', 0.05)], ['밀'])
    assert m['route'] == '1단계-합치기' and m['verdict']['밀'] == judge(row('ramen'), ['밀'])['verdict']['밀']

    assert analyze([('takoyaki', 0.95), ('gyoza', 0.02), ('dumplings', 0.01)], ['고등어'])['verdict']['고등어'] == '정보 부족'
    assert analyze([('cheesecake', 0.2), ('tiramisu', 0.15), ('cannoli', 0.1)], ['땅콩'])['route'] == '2단계'

    # 2단계: 검색 결과(같은 모양의 비율 줄)도 같은 규칙으로 판정
    fake = {a: 0.0 for a in ALLERGENS} | {'새우': 0.8, 'n_recipes': 30, 'nutrition': {}, 'source': 'CLIP 검색'}
    s = analyze([('cheesecake', 0.2), ('tiramisu', 0.15), ('cannoli', 0.1)], ['새우', '땅콩'],
                stage2=lambda: (fake, [('pad_thai', 20.1), ('pho', 3.2)]))
    assert s['route'] == '2단계' and s['foods'] == ['pad_thai', 'pho'] and s['source'] == 'CLIP 검색'
    assert s['verdict'] == {'새우': '주의', '땅콩': '해당 없음'} and s['similar'] == ['pad_thai', 'pho']

    # 애매한 1단계(합치기)도 CLIP 결과를 같이 합침: 후보에 없던 새우를 놓치지 않음
    amb = [('pho', 0.65), ('ramen', 0.3), ('pad_thai', 0.05)]
    assert m['verdict'] == analyze(amb, ['밀'])['verdict'] and m['similar'] == []      # 검색 없으면 그대로
    mc = analyze(amb, ['새우'], stage2=lambda: (fake, [('pad_thai', 20.1)]))
    assert mc['route'] == '1단계-합치기+CLIP' and mc['verdict']['새우'] == '주의' and mc['similar'] == ['pad_thai']
    assert analyze([('ramen', 0.97)], ['새우'], stage2=lambda: (fake, []))['route'] == '1단계-1등'  # 확신 높으면 검색 안 함
    try:
        judge(row('ramen'), ['peanut'])
        raise AssertionError('잘못된 항목을 받아들임')
    except ValueError:
        pass
    print('judge OK', r)
