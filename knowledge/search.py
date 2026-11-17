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
    print('search 자체 검사 통과')
