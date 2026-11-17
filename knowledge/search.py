"""2단계 검색: 사진 → 기본 CLIP → 비슷한 레시피 K개 → 19항목 비율 줄 (담당: 박중현).

04·04-2 결과로 학습 안 한 openai/clip-vit-base-patch32를 쓴다 (docs/진행상황.md).
검색 목록은 notebooks/04-3이 만든 드라이브 찍먹_실험기록/best/clip_base/의 두 파일을
knowledge/index/에 복사해서 쓴다 (레시피 제목이 들어 있어 깃 제외):
    index_emb.npy       레시피 글 벡터 (N×512, fp16, 정규화됨)
    index_meta.parquet  recipe_idx, label, title, 19항목(0/1). 위와 같은 순서

자체 검사 (torch 없이 돌아감):  python -m knowledge.search
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from knowledge.lookup import ALLERGENS

INDEX = Path(__file__).parent / 'index'
NAME = 'openai/clip-vit-base-patch32'
K = 30  # judge.MIN_RECIPES(30)와 같게: 레시피 30개로 계산한 비율이라 '정보 부족'이 아님


def available():
    return (INDEX / 'index_emb.npy').exists() and (INDEX / 'index_meta.parquet').exists()


@lru_cache(maxsize=1)
def _load():
    """첫 검색 때 한 번만 (CLIP 약 600MB, 처음엔 인터넷에서 받음)."""
    import pandas as pd
    import torch
    from torchvision import transforms as T
    from transformers import CLIPModel

    emb = torch.from_numpy(np.load(INDEX / 'index_emb.npy').astype(np.float32))
    meta = pd.read_parquet(INDEX / 'index_meta.parquet')
    assert len(meta) == len(emb), '검색 목록 두 파일의 줄 수가 다름 (04-3을 다시 돌릴 것)'
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])])   # 04와 같은 전처리
    return CLIPModel.from_pretrained(NAME).eval(), tf, emb, meta


def aggregate(sims, labels, flags):
    """K개 검색 결과 → (비율 줄, [(음식, 유사도 합)] 높은 순). flags: K × 19 (ALLERGENS 순서)."""
    score = {}
    for s, c in zip(sims, labels):
        score[c] = score.get(c, 0.0) + float(s)
    foods = sorted(score.items(), key=lambda x: -x[1])
    row = {a: round(float(v), 4) for a, v in zip(ALLERGENS, np.asarray(flags, dtype=float).mean(0))}
    row.update(n_recipes=len(labels), nutrition={}, source='CLIP 검색')  # 섞인 레시피의 영양은 의미 없어서 비움
    return row, foods


T_LLM = 0.5     # LLM 이름 확신이 이보다 낮으면 무시
LLM_ONLY = 0.5  # LLM 근거로 올릴 때의 값 = '확인 필요' (CHECK 0.2 ≤ 0.5 < CAUTION 0.6)


def crosscheck(hit, guesses, ingredients_of=lambda name: None):
    """CLIP 결과 + LLM 이름 추정 → (비율 줄, foods). 판정은 그대로 judge가 한다.
    - LLM 이름이 CLIP 상위 3개 안에 있음 → 그대로 (일치)
    - 다르면 LLM이 말한 음식의 근거(표에 있으면 표 비율, 없으면 LLM 재료 추정을 같은 룰로)에서
      '확인 필요' 이상인 항목만 '확인 필요'까지 올림. LLM 근거로 '주의'는 만들지 않는다 (AGENTS.md 규칙 1).
    guesses: guess_name() 결과 [(이름, 확신)] 또는 None. ingredients_of: guess_ingredients."""
    from knowledge.judge import CHECK
    from knowledge.lookup import row as table_row
    r, foods = hit
    if not guesses or guesses[0][1] < T_LLM:
        return hit
    name, _ = guesses[0]
    if name in [c for c, _ in foods[:3]]:
        return {**r, 'source': r['source'] + ' (LLM 이름 일치)'}, foods
    other, how = table_row(name), '이름'
    if other is None:
        ing = ingredients_of(name)
        if not ing:
            return hit
        import pandas as pd
        from knowledge.rules import allergen_flags
        other, how = allergen_flags(pd.Series([' , '.join(ing).lower()])).iloc[0].astype(float), '재료'
    out = {a: max(r[a], LLM_ONLY) if other[a] >= CHECK else r[a] for a in ALLERGENS}
    out.update(n_recipes=r['n_recipes'], nutrition={}, source=f"{r['source']} + LLM {how} 추정 ({name})")
    return out, foods


def search(img, k=K):
    """PIL 사진 → (비율 줄, [(음식, 점수)]). 1단계 lookup.row와 같은 모양의 줄."""
    import torch
    import torch.nn.functional as F

    model, tf, emb, meta = _load()
    with torch.no_grad():
        px = tf(img.convert('RGB'))[None]
        f = F.normalize(model.visual_projection(model.vision_model(pixel_values=px).pooler_output), dim=-1)[0]
    sim, idx = (emb @ f).topk(k)
    hit = meta.iloc[idx.numpy()]
    return aggregate(sim.numpy(), hit.label.values, hit[ALLERGENS].values)


if __name__ == '__main__':
    i = ALLERGENS.index('새우')
    fl = np.zeros((4, len(ALLERGENS)))
    fl[:3, i] = 1
    r, foods = aggregate([0.9, 0.8, 0.7, 0.2], ['pad_thai', 'pad_thai', 'pho', 'pho'], fl)
    assert r['새우'] == 0.75 and r['n_recipes'] == 4 and r['source'] == 'CLIP 검색'
    assert [c for c, _ in foods] == ['pad_thai', 'pho']

    # 교차 확인: 일치 / 표에 있는 다른 음식(합침) / 표에 없는 음식(재료 추정은 '확인 필요'까지) / 확신 낮음(무시)
    from knowledge.judge import judge
    from knowledge.lookup import row as table_row
    hit = (r, foods)
    assert crosscheck(hit, None) is hit and crosscheck(hit, [('ramen', 0.3)]) is hit
    assert 'LLM 이름 일치' in crosscheck(hit, [('pho', 0.9)])[0]['source']
    m, _ = crosscheck(hit, [('ramen', 0.9)])
    assert table_row('ramen')['밀'] >= 0.6 and m['밀'] == LLM_ONLY and m['n_recipes'] == 4  # 표는 '주의'여도 '확인 필요'까지만
    g, _ = crosscheck(hit, [('not_in_table_dish', 0.9)], lambda n: ['egg noodles', 'peanuts'])
    assert g['땅콩'] == LLM_ONLY and g['새우'] == 0.75 and '재료 추정' in g['source']
    assert judge({**g, 'n_recipes': 30}, ['땅콩'])['verdict']['땅콩'] == '확인 필요'
    assert crosscheck(hit, [('not_in_table_dish', 0.9)]) is hit  # 재료 추정 실패 → CLIP만
    print('search 자체 검사 통과')
