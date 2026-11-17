"""LLM 부르는 곳. guess.py · explain.py는 여기 ask()만 쓴다.

- 제공자는 .env의 LLM_PROVIDER 로 고른다: gemini(기본) | groq. 다른 곳으로 바꿔도 이 파일만 고치면 된다.
- 키·모델 이름은 .env 에서 읽는다 (os.environ). 코드·출력에 키를 쓰지 않는다.
- 실패하면(키 없음, 인터넷 끊김, 하루 횟수 초과, 답이 이상함) 예외 대신 None → 앱은 LLM 없이도 동작.
- 같은 질문은 llm/_cache/ 에 저장해서 다시 묻지 않는다 (무료 횟수 아끼기). 폴더는 gitignore.

.env 예시
    GEMINI_API_KEY=...
    # 선택: LLM_PROVIDER=groq, GROQ_API_KEY=..., GEMINI_MODEL=..., GROQ_MODEL=...
"""
import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

try:  # .env 를 os.environ 으로 (없어도 동작)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

TIMEOUT = 60          # 초. 너무 오래 걸리면 포기하고 None
RETRY_WAITS = (0, 2, 5)  # 서버가 붐빌 때(503) 기다렸다 다시 묻는 간격(초). 하루 횟수 초과(429)는 다시 묻지 않음
CACHE = Path(__file__).parent / '_cache'
DEFAULT_MODEL = {'gemini': 'gemini-3.8-flash', 'groq': 'qwen/qwen3.8-27b'}
last_error = None     # 마지막 실패 이유 (확인용, 키는 들어가지 않음)


def provider():
    return os.environ.get('LLM_PROVIDER', 'gemini').strip().lower()


def model_name():
    p = provider()
    return os.environ.get(f'{p.upper()}_MODEL') or DEFAULT_MODEL.get(p, '')


def ready():
    """키가 있으면 True (실제로 되는지는 ask로 확인)."""
    return bool(os.environ.get(f'{provider().upper()}_API_KEY'))


def _jpeg(img, side=768):
    """PIL 사진 → 작게 줄인 JPEG bytes (전송량·토큰 줄이기)."""
    im = img.convert('RGB')
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85)
    return buf.getvalue()


def _cache_get(key):
    f = CACHE / f'{key}.json'
    if f.exists():
        try:
            return json.loads(f.read_text(encoding='utf-8'))['answer']
        except (OSError, ValueError, KeyError):
            return None
    return None


def _cache_put(key, answer):
    try:
        CACHE.mkdir(exist_ok=True)
        (CACHE / f'{key}.json').write_text(json.dumps({'answer': answer}, ensure_ascii=False), encoding='utf-8')
    except OSError:
        pass


def _gemini(prompt, jpg, want_json):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'],
                          http_options=types.HttpOptions(timeout=TIMEOUT * 1000))
    parts = [types.Part.from_bytes(data=jpg, mime_type='image/jpeg')] if jpg else []
    cfg = types.GenerateContentConfig(temperature=0, max_output_tokens=2048,
                                      response_mime_type='application/json' if want_json else None,
                                      thinking_config=types.ThinkingConfig(thinking_level='low'))  # 오래 고민 안 해도 되는 일 → 빠르게
    r = client.models.generate_content(model=model_name(), contents=parts + [prompt], config=cfg)
    return r.text


def _groq(prompt, jpg, want_json):
    import requests
    content = [{'type': 'text', 'text': prompt}]
    if jpg:
        url = 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode()
        content.append({'type': 'image_url', 'image_url': {'url': url}})
    body = {'model': model_name(), 'temperature': 0, 'max_tokens': 1024,
            'messages': [{'role': 'user', 'content': content}]}
    if want_json:
        body['response_format'] = {'type': 'json_object'}
    r = requests.post('https://api.groq.com/openai/v1/chat/completions', json=body, timeout=TIMEOUT,
                      headers={'Authorization': f"Bearer {os.environ['GROQ_API_KEY']}"})
    if not r.ok:  # 이유(어떤 모델이 없는지 등)까지 보이게
        raise RuntimeError(f'{r.status_code} {r.text[:300]}')
    return r.json()['choices'][0]['message']['content']


CALL = {'gemini': _gemini, 'groq': _groq}


def ask(prompt, img=None, want_json=False):
    """LLM에 묻고 답을 돌려준다. want_json=True면 dict/list, 아니면 문자열. 실패하면 None."""
    global last_error
    p = provider()
    if p not in CALL:
        last_error = f'모르는 LLM_PROVIDER: {p} (가능: {list(CALL)})'
        return None
    if not ready():
        last_error = f'.env 에 {p.upper()}_API_KEY 가 없음'
        return None
    jpg = _jpeg(img) if img is not None else None
    key = hashlib.sha256(f'{p}|{model_name()}|{want_json}|{prompt}'.encode() + (jpg or b'')).hexdigest()[:32]
    hit = _cache_get(key)
    if hit is not None:
        return hit
    answer = None
    for wait in RETRY_WAITS:  # 503(사람 많음)처럼 잠깐 막힌 경우만 조금 쉬었다가 다시
        time.sleep(wait)
        try:
            text = CALL[p](prompt, jpg, want_json)
            text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)  # 생각 과정은 빼고 답만
            answer = _parse_json(text) if want_json else (text or '').strip()
            break
        except Exception as e:  # 인터넷·횟수 초과·키 오류 등 → 앱은 계속 돌아야 함
            last_error = f'{type(e).__name__}: {str(e)[:200]}'
            if not any(s in str(e) for s in ('503', 'UNAVAILABLE', '504', 'DEADLINE', 'overloaded', '500 ')):
                return None
    else:
        return None
    if not answer:
        last_error = '빈 답 또는 JSON 아님'
        return None
    last_error = None
    _cache_put(key, answer)
    return answer


def _parse_json(text):
    """'```json ... ```' 처럼 감싸서 와도 JSON만 꺼낸다. 못 읽으면 None."""
    t = (text or '').strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[-1].rsplit('```', 1)[0]
    try:
        return json.loads(t)
    except ValueError:
        return None
