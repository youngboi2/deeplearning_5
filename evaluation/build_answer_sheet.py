"""알레르기 정답지 만들기 (담당: 서준영)

정답지_레시피기록.tsv  음식마다 레시피 3개, 레시피별로 들어간 알레르기 항목과 그 재료
  → 정답지.csv        음식 × 19항목 1/0 (레시피 3개 중 2개 이상에 들어가면 1)
  → 정답지_근거.md     사람이 확인할 때 보는 표 (칸마다 어느 레시피의 어떤 재료인지)

기준과 레시피 고르는 방법은 evaluation/README.md.
레시피 기록을 고쳤으면 저장소 루트에서 다시 실행:  python -m evaluation.build_answer_sheet
"""
import csv
import re
from collections import Counter
from pathlib import Path

from knowledge.lookup import ALLERGENS, row

HERE = Path(__file__).parent
SRC, OUT, MD = HERE / '정답지_레시피기록.tsv', HERE / '정답지.csv', HERE / '정답지_근거.md'
SLOTS = ('R1', 'R2', 'R3')  # 음식마다 레시피 3개
NEED = 2                    # 이 개수 이상에 들어가면 '들어감(1)'


def parse(cell):
    """'알류(egg);밀(flour, soy sauce)' → {'알류': 'egg', '밀': 'flour, soy sauce'}"""
    out = {}
    for part in filter(None, (p.strip() for p in cell.split(';'))):
        m = re.fullmatch(r'(\S+?)\((.+)\)', part)
        if not m or m[1] not in ALLERGENS:
            raise ValueError(f'형식 오류: {part!r} (예: 밀(flour). 항목은 {ALLERGENS})')
        out[m[1]] = m[2]
    return out


def load():
    foods = {}
    with open(SRC, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            foods.setdefault(r['class'], {})[r['recipe']] = dict(site=r['site'], url=r['url'], found=parse(r['allergens']))
    dup = [u for u, n in Counter(x['url'] for recs in foods.values() for x in recs.values()).items() if n > 1]
    assert not dup, f'같은 레시피가 두 번 들어감: {dup}'
    for c, recs in foods.items():
        assert sorted(recs) == list(SLOTS), f'{c}: 레시피가 R1·R2·R3 3개가 아님 ({sorted(recs)})'
        assert row(c) is not None, f'{c}: knowledge/food_table.csv에 없는 이름'
    return foods


def build():
    foods = load()
    sheet, sections, n_pos, n_edge = [], [], Counter(), 0
    for c, recs in foods.items():
        cnt = {a: sum(a in recs[s]['found'] for s in SLOTS) for a in ALLERGENS}
        ans = {a: int(cnt[a] >= NEED) for a in ALLERGENS}
        n_pos.update(a for a in ALLERGENS if ans[a])
        sheet.append({'class': c, **ans, '근거': ' | '.join(recs[s]['url'] for s in SLOTS)})

        lines = [f'### {c}', '',
                 '레시피: ' + ' · '.join(f"{s} [{recs[s]['site']}]({recs[s]['url']})" for s in SLOTS), '',
                 '| 항목 | R1 | R2 | R3 | 개수 | 정답 |', '|---|---|---|---|---|---|']
        for a in ALLERGENS:
            if cnt[a]:
                cells = [recs[s]['found'].get(a, '—') for s in SLOTS]
                mark = '**1**' if ans[a] else '0 (경계)'
                n_edge += not ans[a]
                lines.append(f"| {a} | {' | '.join(cells)} | {cnt[a]}/3 | {mark} |")
        sections.append('\n'.join(lines + ['', '표에 없는 항목은 세 레시피 모두에 없음 → 0', '']))

    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['class', *ALLERGENS, '근거'], lineterminator='\n')
        w.writeheader()
        w.writerows(sheet)

    total = len(sheet) * len(ALLERGENS)
    head = [
        '# 알레르기 정답지 근거',
        '',
        '> 자동 생성 파일. 직접 고치지 말고 `정답지_레시피기록.tsv`를 고친 뒤 `python -m evaluation.build_answer_sheet`.',
        '> 기준은 [`README.md`](README.md).',
        '',
        f'- 음식 {len(sheet)}종 × 19항목 = {total}칸 → 정답 1: **{sum(n_pos.values())}칸**, '
        f'경계(3개 중 1개에만 있음 → 0): **{n_edge}칸**',
        '- 항목별 정답 1인 음식 수: ' + ', '.join(f'{a} {n_pos[a]}' for a in ALLERGENS),
        '',
        '**확인하는 법:** 음식마다 링크를 열어 표의 재료가 레시피 재료 목록에 정말 있는지 본다. '
        '틀린 곳은 `정답지_레시피기록.tsv`에서 고친다. 경계 칸은 "확인 필요"가 맞는 자리라 따로 표시했다.',
        '',
    ]
    MD.write_text('\n'.join(head + sections), encoding='utf-8')
    print(f'정답지 {len(sheet)}종 저장 → {OUT.name}, {MD.name} (정답 1: {sum(n_pos.values())}칸, 경계: {n_edge}칸)')


if __name__ == '__main__':
    build()
