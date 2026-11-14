"""1M_Final 레시피 → 음식별 '알레르기 비율 + 영양 중앙값' 표 (knowledge/food_table.csv).

notebooks/01_eda_알레르기_나트륨.ipynb C·D·E 파트를 스크립트로 옮김 (규칙은 rules.py).
1M_Final이 1.4GB라 Colab이나 학과 서버에서 실행한다:

    cd knowledge
    python build_table.py --csv "/content/drive/MyDrive/26-2 딥러닝/data1/1M_Final.csv"

새 클래스가 생기면 share_storage/model/classes.txt 에 추가된 이름으로 다시 돌리면 된다
(이름이 레시피 제목과 다르면 rules.py 의 ALIAS 에 별칭 추가).
"""
import argparse
from pathlib import Path

import pandas as pd

from rules import allergen_flags, ingredients_text, match

HERE = Path(__file__).parent
NUTRI = {'energy_kcal_per_serving': 'kcal', 'protein_g_per_serving': 'protein_g', 'fat_g_per_serving': 'fat_g',
         'carbs_g_per_serving': 'carbs_g', 'fiber_g_per_serving': 'fiber_g', 'sodium_mg_per_serving': 'sodium_mg'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='1M_Final.csv 경로')
    ap.add_argument('--classes', default=HERE.parent / 'share_storage/model/classes.txt')
    ap.add_argument('--out', default=HERE / 'food_table.csv')
    a = ap.parse_args()

    classes = Path(a.classes).read_text(encoding='utf-8').split()
    df = pd.read_csv(a.csv, usecols=['title', 'NER', *NUTRI], encoding='utf-8', encoding_errors='replace')
    final = {c: fin for c, (_, fin) in match(df.title, classes).items()}
    pairs = pd.concat([pd.DataFrame({'idx': m[m].index, 'label': c}) for c, m in final.items()], ignore_index=True)

    # 알레르기: 그 음식 레시피 중 해당 재료가 1개 이상 들어간 비율
    ing = df.loc[pairs.idx.unique(), 'NER'].map(ingredients_text)
    flags = allergen_flags(ing)
    ratio = flags.loc[pairs.idx.values].set_index(pairs.label.values).groupby(level=0).mean().reindex(classes).fillna(0)

    # 영양: 1인분 값의 중앙값 (평균은 이상치에 흔들려서)
    nut = df.loc[pairs.idx.values, list(NUTRI)].apply(pd.to_numeric, errors='coerce').rename(columns=NUTRI)
    nut = nut.set_index(pairs.label.values).groupby(level=0).median().reindex(classes)

    out = ratio.round(4).join(nut.round(1))
    out['n_recipes'] = pd.Series({c: int(m.sum()) for c, m in final.items()})
    out.index.name = 'class'
    assert len(out) == len(classes)
    out.to_csv(a.out)
    print(f'{a.out} 저장: {len(out)}종, 레시피 30개 미만 {(out.n_recipes < 30).sum()}종')


if __name__ == '__main__':
    main()
