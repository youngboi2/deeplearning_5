"""분류 모델 불러오기 · 예측 (담당: 모델).

modeling/weights/best.pt(git 제외)와 modeling/classes.txt(Food-101 meta 순서)를 쓴다. 두 종류를 자동으로 구분:
- convnext_base : 주한솔 hansol_train_convnext.ipynb (timm `convnext_base.fb_in22k_ft_in1k`, EMA 가중치,
                  {'model': state_dict, 'cfg': {...}}로 저장). Food-101 test top-1 93.4%  ← 지금 앱 기본
- convnext_tiny : 처음 학습 노트북 (torchvision convnext_tiny, state_dict만). test top-1 88.8%
확률 보정값(temperature)은 modeling/calibration.json (notebooks/05 결과). 그 파일의 model이 불러온 모델과 같을 때만 쓴다.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms as T

HERE = Path(__file__).parent
WEIGHTS = HERE / 'weights' / 'best.pt'
CALIB = HERE / 'calibration.json'
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])   # tiny 학습 때 전처리


def load_model(weights=WEIGHTS, classes=HERE / 'classes.txt'):
    """→ (모델, 클래스 목록). 모델에 .kind, .preprocess(전처리), .temperature(보정값)를 붙여 둔다."""
    classes = Path(classes).read_text(encoding='utf-8').split()
    try:
        ck = torch.load(weights, map_location='cpu', weights_only=True)
    except Exception:  # 한솔 체크포인트의 cfg에 기본형 아닌 값이 있으면 (팀원 파일이라 믿고 엶)
        ck = torch.load(weights, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model' in ck:
        import timm
        name = (ck.get('cfg') or {}).get('timm_name', 'convnext_base.fb_in22k_ft_in1k')
        m = timm.create_model(name, pretrained=False, num_classes=len(classes))
        m.load_state_dict(ck['model'])
        m.kind = name.split('.')[0]
        # 한솔 평가 전처리와 같게: timm create_transform(224, crop_pct=0.875, bicubic) = 짧은 변 256 → 가운데 224
        m.preprocess = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                                  T.ToTensor(), NORM])
    else:
        m = models.convnext_tiny()
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(classes))
        m.load_state_dict(ck)
        m.kind, m.preprocess = 'convnext_tiny', TF
    cal = json.loads(CALIB.read_text(encoding='utf-8')) if CALIB.exists() else {}
    m.temperature = cal['temperature'] if cal.get('model', 'convnext_tiny') == m.kind else 1.0
    return m.eval(), classes


@torch.no_grad()
def predict(model, img, k=3):
    """PIL 이미지 → [(클래스, 확률)] 상위 k개. 확률은 temperature로 보정된 값 (judge의 T_ONE·T_KNOWN도 보정 기준)."""
    m, classes = model
    conf, idx = (m(m.preprocess(img.convert('RGB'))[None]) / m.temperature).softmax(1)[0].topk(k)
    return [(classes[i], float(c)) for c, i in zip(conf, idx)]
