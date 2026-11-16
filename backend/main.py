"""백엔드: 사진 + 내 알레르기 → 판정 결과 JSON.

실행 (저장소 루트에서):  uvicorn backend.main:app --reload
문서/테스트 화면:        http://localhost:8000/docs
"""
import io
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from knowledge.judge import analyze
from knowledge.lookup import ALLERGENS
from modeling.model import WEIGHTS, load_model, predict

MAX_BYTES = 10 * 1024 * 1024  # 폰 사진 넉넉히

app = FastAPI(title='찍먹')


@lru_cache(maxsize=1)
def model():
    """첫 요청 때 한 번만 불러옴 (서버 켜는 속도 유지)."""
    if not WEIGHTS.exists():
        raise HTTPException(503, f'모델 가중치 없음: 드라이브 찍먹_실험기록/best/convnext_tiny/best.pt → {WEIGHTS}')
    return load_model()


@app.get('/allergens')
def allergens():
    """화면의 알레르기 체크박스 목록 (19개)."""
    return ALLERGENS


@app.post('/analyze')
def analyze_photo(image: UploadFile = File(...), allergens: str = Form('')):
    """allergens: 쉼표로 구분 (예: '밀,땅콩'). async 아님 → 모델 계산이 다른 요청을 막지 않음."""
    data = image.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, '사진이 10MB보다 큼')
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, '사진 파일이 아님')

    mine = [a.strip() for a in allergens.split(',') if a.strip()]
    cands = predict(model(), img)
    try:
        result = analyze(cands, mine)
    except ValueError as e:  # 모르는 알레르기 이름
        raise HTTPException(400, str(e))
    result['candidates'] = [{'food': c, 'prob': round(p, 4)} for c, p in cands]
    return result
