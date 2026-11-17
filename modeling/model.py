"""분류 모델 불러오기 · 예측 (담당: 모델).

지금 best.pt는 timm ConvNeXt-Base 체크포인트({'model','epoch','val_top1','cfg'}). 예전 torchvision Tiny state_dict도 읽는다.
모델 크기(tiny/small/base/large)는 가중치 모양으로 알아낸다.
modeling/weights/best.pt(git 제외)와 modeling/classes.txt(클래스 순서)를 쓴다.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms as T

def _tf(size=224):
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


TF = _tf()


HERE = Path(__file__).parent
WEIGHTS = HERE / 'weights' / 'best.pt'


SIZES = {(96, 9): 'convnext_tiny', (96, 27): 'convnext_small', (128, 27): 'convnext_base', (192, 27): 'convnext_large'}


def _arch(sd, timm_keys):
    """가중치 모양으로 tiny/small/base/large를 알아낸다 (첫 층 채널 수, 3번째 단계 블록 수)."""
    first, stage = ('stem.0.weight', 'stages.2.blocks.') if timm_keys else ('features.0.0.weight', 'features.5.')
    depth = len({k[len(stage):].split('.')[0] for k in sd if k.startswith(stage)})
    return SIZES[(sd[first].shape[0], depth)]


def load_model(weights=WEIGHTS, classes=HERE / 'classes.txt'):
    """두 가지 체크포인트를 모두 읽는다.
    - torchvision state_dict (예전 Tiny): 키가 features.* / classifier.*
    - 학습 노트북 체크포인트 {'model','epoch','val_top1','cfg'}: timm ConvNeXt, 키가 stem.* / stages.* / head.*"""
    global TF
    classes = Path(classes).read_text(encoding='utf-8').split()
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)  # 팀이 만든 파일만 넣을 것
    cfg = ckpt.get('cfg') if isinstance(ckpt, dict) and 'model' in ckpt else None
    sd = ckpt['model'] if cfg is not None or (isinstance(ckpt, dict) and 'model' in ckpt) else ckpt
    sd = {k.removeprefix('_orig_mod.').removeprefix('module.'): v for k, v in sd.items()}  # compile/DataParallel 접두어
    timm_keys = 'stem.0.weight' in sd
    name = _arch(sd, timm_keys)
    size = next((cfg[k] for k in ('img_size', 'image_size', 'input_size') if isinstance(cfg, dict) and isinstance(cfg.get(k), int)), None)
    if timm_keys:  # timm으로 학습한 모델 (지금 best.pt: ConvNeXt-Base)
        import timm
        m = timm.create_model(name, pretrained=False, num_classes=len(classes))
    else:                      # torchvision 모델
        m = getattr(models, name)()
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(classes))
    m.load_state_dict(sd)
    if size:
        TF = _tf(size)
    return m.eval(), classes


@torch.no_grad()
def predict(model, img, k=3):
    """PIL 이미지 → [(클래스, 확률)] 상위 k개."""
    m, classes = model
    conf, idx = m(TF(img.convert('RGB'))[None]).softmax(1)[0].topk(k)  # TF는 load_model이 크기를 맞춤
    return [(classes[i], float(c)) for c, i in zip(conf, idx)]
