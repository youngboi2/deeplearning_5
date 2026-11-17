"""LLM 연결 확인 (판정 코드와 별개). 저장소 루트에서:

    python -m llm.check                 # 재료 추정 + 설명만
    python -m llm.check 사진.jpg         # 사진 이름 추정까지

키가 없거나 실패하면 'None'과 이유가 나온다. 키 값은 출력하지 않는다.
"""
import sys

from PIL import Image

from knowledge.judge import analyze
from llm import client
from llm.explain import explain
from llm.guess import guess_ingredients, guess_name


def show(title, value):
    print(f'\n[{title}]\n{value}')
    if value is None:
        print('  → 실패 이유:', client.last_error)


if __name__ == '__main__':
    print(f'제공자: {client.provider()} / 모델: {client.model_name()} / 키 있음: {client.ready()}')

    if len(sys.argv) > 1:
        show('이름 추정 guess_name', guess_name(Image.open(sys.argv[1])))

    show('재료 추정 guess_ingredients("pad_thai")', guess_ingredients('pad_thai'))

    info = analyze([('ramen', 0.92), ('pho', 0.05), ('udon', 0.01)], ['밀', '땅콩'])
    show('설명 explain("ramen", 판정 결과)', explain('ramen', info))
