"""LLM 설명

규칙
- 판정(주의/확인 필요/해당 없음/정보 부족)은 knowledge.judge 가 이미 정했다. LLM은 그 결과를 쉬운 말로 풀어 주고
  질문에 답하기만 한다. 판정을 바꾸거나 새로 하지 않는다.
- '안전하다'고 말하지 않는다 → 프롬프트로 막고, 답에 들어 있으면 None으로 버린다.
- 근거는 우리 표(food_table.csv)의 숫자만. 표에 없는 정보는 '추정'이라고 밝힌다.
"""
import json

from knowledge.lookup import ALLERGENS, row
from llm.client import ask

BANNED = ('안전', 'safe')   # 이 말이 나오면 설명을 쓰지 않는다 (AGENTS.md 절대 규칙 2)


def _facts(cls, info):
    """analyze() 결과 + 표 비율 → LLM에게 줄 근거 (숫자는 표에서만)."""
    # 표 비율은 1단계(표에서 찾은 음식)일 때만 맞는 숫자. 2단계는 CLIP이 찾은 레시피 30개로 따로 계산해서 표 숫자를 쓰면 안 됨
    stage1 = str(info.get('route', '')).startswith('1단계')
    rows = [r for f in info.get('foods') or [cls] if (r := row(f))] if stage1 else []
    pct = {a: round(max(r[a] for r in rows) * 100) for a in ALLERGENS} if rows else {}
    mine = {a: {'판정': v, '레시피 중 들어간 비율(%)': pct.get(a)} for a, v in (info.get('verdict') or {}).items()}
    return {
        '음식(영어 이름)': info.get('foods') or [cls],
        '찾은 방법': '분류 모델이 알아본 음식의 레시피 통계' if stage1 else '처음 보는 음식이라 사진과 비슷한 레시피를 찾아서 계산',
        '판정 경로': info.get('route'),
        '근거 출처': info.get('source'),
        '근거 레시피 수': info.get('n_recipes'),
        '내 알레르기 판정': mine,
        '다른 주의 재료': {a: pct.get(a) for a in info.get('other_caution') or []},
    }


def explain(cls, info, question=None):
    """cls: 1등 음식 이름, info: knowledge.judge.analyze() 결과, question: 사용자가 물어본 말(선택).
    반환: 한국어 설명 문자열 (2~4문장), 연결 전·실패·금지어 포함 시 None."""
    if not info:
        return None
    facts = _facts(cls, info)
    guess = '추정' in str(info.get('source')) or 'LLM' in str(info.get('source'))
    prompt = (
        '너는 여행 중 처음 보는 음식의 알레르기 정보를 설명해 주는 도우미야. 아래 [근거]만 사용해서 한국어로 답해.\n'
        '규칙:\n'
        '1. 판정(주의/확인 필요/해당 없음/정보 부족)은 이미 정해졌다. 절대 바꾸거나 새로 판정하지 마.\n'
        "2. '안전하다', '먹어도 된다', '괜찮다'는 말을 절대 쓰지 마. '해당 없음'은 '레시피에 드물게 들어간다' 정도로만 말해.\n"
        '3. 숫자는 [근거]에 있는 것만 쓰고 지어내지 마. 음식 이름은 한국어로 바꿔 불러도 돼.\n'
        "4. '정보 부족'이면 레시피가 적어서 판단하기 어렵다고 말해.\n"
        + ("5. 이 결과는 LLM이 재료를 추정한 것이라 '추정'이라고 꼭 밝혀.\n" if guess else '')
        + '6. 마지막에 걱정되면 가게에 직접 물어보라고 한 문장 덧붙여.\n'
        + ('사용자 질문에 2~4문장으로 답해: ' + str(question) + '\n' if question
           else '왜 이런 판정이 나왔는지 2~4문장으로 쉽게 설명해.\n')
        + '[근거]\n' + json.dumps(facts, ensure_ascii=False)
    )
    text = ask(prompt)
    if not text or any(b in text.lower() for b in BANNED):
        return None
    return text
