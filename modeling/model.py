"""분류 모델 불러오기 · 예측 (담당: 모델).

Colab 학습 노트북(찍먹_모델학습.ipynb)과 같은 구조: torchvision convnext_tiny, 마지막 층만 클래스 수로 교체.
share_storage/model/ 에 best.pt(state_dict, git 제외)와 classes.txt(클래스 순서)가 있어야 한다.
새 클래스가 승인되면 classes.txt 끝에 추가 → 재학습한 best.pt와 함께 교체 (MLOps 담당).
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms as T

TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_model(model_dir):
    model_dir = Path(model_dir)
    classes = (model_dir / 'classes.txt').read_text(encoding='utf-8').split()
    m = models.convnext_tiny()
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(classes))
    m.load_state_dict(torch.load(model_dir / 'best.pt', map_location='cpu'))
    return m.eval(), classes


@torch.no_grad()
def predict(model, img, k=3):
    """PIL 이미지 → [(클래스, 확률)] 상위 k개."""
    m, classes = model
    conf, idx = m(TF(img.convert('RGB'))[None]).softmax(1)[0].topk(k)
    return [(classes[i], float(c)) for c, i in zip(conf, idx)]
