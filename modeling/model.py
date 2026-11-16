"""분류 모델 불러오기 · 예측 (담당: 모델).

Colab 학습 노트북(찍먹_모델학습.ipynb)과 같은 구조: torchvision convnext_tiny, 마지막 층만 클래스 수로 교체.
modeling/weights/best.pt(state_dict, git 제외)와 modeling/classes.txt(클래스 순서)를 쓴다.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms as T

TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


HERE = Path(__file__).parent
WEIGHTS = HERE / 'weights' / 'best.pt'


def load_model(weights=WEIGHTS, classes=HERE / 'classes.txt'):
    classes = Path(classes).read_text(encoding='utf-8').split()
    m = models.convnext_tiny()
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(classes))
    m.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return m.eval(), classes


@torch.no_grad()
def predict(model, img, k=3):
    """PIL 이미지 → [(클래스, 확률)] 상위 k개."""
    m, classes = model
    conf, idx = m(TF(img.convert('RGB'))[None]).softmax(1)[0].topk(k)
    return [(classes[i], float(c)) for c, i in zip(conf, idx)]
