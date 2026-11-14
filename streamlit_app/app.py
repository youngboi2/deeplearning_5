"""찍먹 Streamlit 앱 (담당: MLOps). 실행: streamlit run streamlit_app/app.py

흐름: 사진 → modeling.predict(top-3) → knowledge.lookup(알레르기·영양) → llm.explain(설명, 선택)
      → 사용자 정답 확인 → share_storage/pending/ 에 사진, metadata/uploads.csv 에 기록 (재학습은 trigger/)
"""
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from knowledge.lookup import lookup  # noqa: E402
from llm.explain import explain  # noqa: E402
from modeling.model import load_model, predict  # noqa: E402

SHARE = Path(os.getenv('SHARE_DIR', ROOT / 'share_storage'))
LOG = SHARE / 'metadata' / 'uploads.csv'
LOG_COLS = ['ts', 'time', 'user', 'file', 'pred', 'conf', 'correct', 'label']  # 참고 MNIST 프로젝트와 같은 칸


@st.cache_resource  # 서버 전체에서 한 번만 로드
def get_model():
    return load_model(SHARE / 'model')


def save_feedback(img, user, pred, conf, label):
    ts = int(time.time())
    name = f'{ts}_{uuid.uuid4().hex[:8]}.jpg'
    (SHARE / 'pending').mkdir(parents=True, exist_ok=True)
    img.save(SHARE / 'pending' / name)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        w.writerow([ts, datetime.now().isoformat(timespec='seconds'), user, name, pred, round(conf, 4),
                    int(pred == label), label])


st.title('찍먹 🍽️')
if not (SHARE / 'model' / 'best.pt').exists():
    st.error('share_storage/model/best.pt 가 없습니다. 드라이브 찍먹_실험기록/best/convnext_tiny/best.pt 를 복사하세요.')
    st.stop()

user = st.text_input('이름(선택)')
f = st.camera_input('사진 찍기') or st.file_uploader('사진 올리기', type=['jpg', 'jpeg', 'png'])
if f:
    img = Image.open(f).convert('RGB')
    st.image(img, width=320)
    top3 = predict(get_model(), img)
    cls, conf = top3[0]
    st.subheader(f'{cls.replace("_", " ")}  ({conf:.0%})')
    if conf < 0.5:
        st.warning('확신이 낮아요. 아래 후보 중에서 골라 주세요.')

    info = lookup(cls)
    st.caption(f'정보: {info["status"]} · 레시피 {info["n_recipes"]}개 · 출처 {info["source"]}')
    for name, ratio, level in info['allergens']:
        st.write(f'⚠️ **{name}** — {level} (레시피 {ratio:.0%})')
    if info['nutrition']:
        st.write(info['nutrition'])
    st.info('레시피 통계 기반 참고 정보예요. 알레르기가 있다면 꼭 가게에 확인하세요.')

    if st.button('왜 그런가요? (AI 설명)'):
        st.session_state['why'] = explain(cls, info) or 'LLM 연결 전입니다.'
    if 'why' in st.session_state:
        st.write(st.session_state['why'])

    opts = [c for c, _ in top3] + ['목록에 없어요']
    pick = st.radio('맞는 음식은?', opts)
    label = st.text_input('음식 이름 입력') if pick == '목록에 없어요' else pick
    if st.button('정답 보내기') and label:
        save_feedback(img, user, cls, conf, label)
        st.success('고마워요! 검토 후 학습에 반영돼요.')
