"""LLM 설명 (담당: 룰베이스·LLM).

규칙
- 경고 여부는 knowledge.lookup이 정한다. LLM은 그 결과를 쉬운 말로 풀어 주고 질문에 답하기만 한다.
- '안전하다'고 말하지 않는다. 표에 없는 정보는 '추정'이라고 밝힌다.
- 제공자(API / 학과 GPU 오픈소스)는 아직 미정 → 정해지면 이 함수 안만 채운다. 키는 .env에만.
"""


def explain(cls, info, question=None):
    """cls: 클래스 이름, info: lookup(cls) 결과. 반환: 설명 문자열, 연결 전이면 None."""
    return None
