# Histolog AI

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| 언어 / 런타임 | Python 3.11 |
| 웹 프레임워크 | FastAPI + Uvicorn |
| 임베딩 | `BAAI/bge-m3` (sentence-transformers) |
| 벡터 인덱스 | FAISS (CPU) |
| LLM | Google Gemini `gemini-2.5-flash` |
| 토큰 카운트 | tiktoken (`cl100k_base` fallback) |
| 컨테이너 | Docker (python:3.11-slim) |

## 프로젝트 구조

```
.
├── main.py                 # FastAPI 앱 / 엔드포인트 정의
├── rag_pipeline.py         # FAISS 로드 + retrieve + 프롬프트 + Gemini 호출
├── requirements.txt
├── Dockerfile              # python:3.11-slim 기반
├── docker-compose.yml      # shared-network 외부 네트워크에 붙음
├── jeongjo_faiss.index     # FAISS 인덱스 (정조)
├── jeongjo_meta.pkl
├── danjong_faiss.index     # FAISS 인덱스 (단종)
└── danjong_meta.pkl
```

## API

### `GET /` — 헬스체크

```json
{ "health": "ok" }
```

### `POST /histolog/ai/query` — RAG 질의

- **Request Body**

```json
{
  "message": "string",
  "king": "JEONGJO"
}
```

| 필드 | 타입 | 비고 |
|---|---|---|
| message | string | 사용자 질문 |
| king | enum (`JEONGJO`, `DANJONG`) | 지원하지 않는 값이면 400 |

- **Response 200 OK**

```json
{
  "query": "string",
  "king": "JEONGJO",
  "answer": "string",
  "usage": 1234
}
```

`usage` 는 Gemini `usage_metadata.total_token_count` 값으로, Histolog-be 가 `users.token_usage` 에 누적한다.

- **Error**: `400` (지원하지 않는 `king`)

## RAG 파이프라인 흐름

`rag_pipeline.ask(question, king)` 한 호출 안에서 아래 단계가 순서대로 실행된다.

1. **Retrieve** — `BAAI/bge-m3` 로 query를 임베딩하고 해당 임금의 FAISS 인덱스에서 top-2 chunk 를 가져온다.
2. **Prompt 조립** — 조선왕조실록 발췌 + 질문 + 지침("위 기록을 근거로 답하라", "임금 입장에서 대화하듯", "300자 이내 + 문장 완결")을 하나의 프롬프트로 합친다. context 는 1000자에서 자른다.
3. **Generate** — `gemini-2.5-flash` 호출. `max_output_tokens=1600`, `temperature=0.7`, `top_p=0.9`. 시스템 역할로 "너는 조선왕 {임금}이다." 가 앞단에 붙는다.
4. **Usage 회수** — 응답의 `usage_metadata.total_token_count` 를 그대로 리턴값에 실어 BE 가 사용량 누적에 쓰도록 한다.

## 임금 추가 방법

`rag_pipeline.py` 의 `KINGS` dict 에 새 항목을 추가하면 자동으로 같은 파이프라인을 탄다.

```python
KINGS = {
    "JEONGJO": { "index_path": "jeongjo_faiss.index", "meta_path": "jeongjo_meta.pkl", "display_name": "정조" },
    "DANJONG": { "index_path": "danjong_faiss.index", "meta_path": "danjong_meta.pkl", "display_name": "단종" },
    # 새 임금을 추가하려면 동일 형식 + 미리 빌드한 *.index / *.pkl 필요
}
```

BE 쪽도 `com.example.histologbe.domain.chat.King` enum 과 FE `src/constants/kings.js` 에 동일 key 를 추가해야 한다.

## 외부 의존성

- **Google Generative AI (Gemini)** — `GOOGLE_API_KEY` 필수. 요청별 토큰 사용량이 응답에 함께 내려와 BE 사용량 집계의 기준이 된다.
- **사전 빌드된 FAISS 인덱스** — 별도 빌드 스크립트가 이 리포에는 없다. `.index` / `.pkl` 은 외부에서 생성해서 루트에 두는 방식.

## 관련 저장소

- **Histolog-be** — Spring Boot 백엔드. `POST /histolog/ai/query` 를 호출하는 유일한 클라이언트.
- **Histolog-fe** — Expo / React Native 클라이언트. AI 서버를 직접 부르지는 않고 BE 를 통해 간접 호출.
