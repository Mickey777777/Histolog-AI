import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import tiktoken
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# =========================
# 0. 왕별 설정
# =========================
KINGS = {
    "JEONGJO": {
        "index_path": "jeongjo_faiss.index",
        "meta_path": "jeongjo_meta.pkl",
        "display_name": "정조",
    },
    "DANJONG": {
        "index_path": "danjong_faiss.index",
        "meta_path": "danjong_meta.pkl",
        "display_name": "단종",
    },
}

# =========================
# 1. FAISS + 메타데이터 로드 (왕별)
# =========================
indexes = {}
chunk_texts_by_king = {}

for king, cfg in KINGS.items():
    indexes[king] = faiss.read_index(cfg["index_path"])
    with open(cfg["meta_path"], "rb") as f:
        meta = pickle.load(f)
    chunk_texts_by_king[king] = meta["chunk_texts"]
    print(f"[INFO] {king} ({cfg['display_name']}) FAISS 로드 완료 (ntotal={indexes[king].ntotal})")

# =========================
# 2. 임베딩 모델 로드 (공용)
# =========================
embed_model = SentenceTransformer("BAAI/bge-m3")

# =========================
# 3. 토큰 추정 함수 (fallback)
# =========================
def estimate_tokens(text):
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        # fallback: 대략 1 token ≈ 4 chars
        return len(text) // 4

# =========================
# 4. 검색 함수
# =========================
def retrieve(query, king, k=2):
    q_emb = embed_model.encode([query])
    D, I = indexes[king].search(np.array(q_emb).astype(np.float32), k)

    return [chunk_texts_by_king[king][i] for i in I[0]]

# =========================
# 5. 프롬프트 생성 함수
# =========================
def build_prompt(question, retrieved_docs, king):
    display_name = KINGS[king]["display_name"]
    context = "\n\n".join(retrieved_docs)[:1000]

    prompt = f"""
다음은 조선왕조실록 {display_name} 관련 기록이다:

{context}

질문: {question}

지침:
- 반드시 위 기록을 근거로 답하라
- {display_name}의 입장에서 대화하듯 답하라
- 300자 이내를 목표로 하되, 반드시 완전한 문장으로 마무리하라. 절대 중간에 끊지 마라.

답변:
"""
    return prompt

# =========================
# 6. Gemini 응답 + 토큰 디버깅
# =========================
def generate_answer(prompt, king):
    display_name = KINGS[king]["display_name"]

    model = genai.GenerativeModel("gemini-2.5-flash")

    full_input = [
        f"너는 조선왕 {display_name}이다.",
        "반드시 문장을 끝까지 완결하여 작성하라. 절대 중간에 끊지 마라.",
        prompt
    ]

    response = model.generate_content(
        full_input,
        generation_config={
            "max_output_tokens": 1600,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )

    # =====================
    # 텍스트 추출
    # =====================
    try:
        parts = response.candidates[0].content.parts
        result = "".join([p.text for p in parts if hasattr(p, "text")])
    except:
        result = ""

    # =====================
    # 토큰 디버깅
    # =====================
    print("\n[DEBUG] ===== Token Usage =====")

    # 1. API usage (있을 경우)
    try:
        usage = response.usage_metadata
        print(f"Input tokens  : {usage.prompt_token_count}")
        print(f"Output tokens : {usage.candidates_token_count}")
        print(f"Total tokens  : {usage.total_token_count}")
    except:
        print("[INFO] usage_metadata 없음 → 추정값 사용")

        input_text = "\n".join(full_input)

        est_input = estimate_tokens(input_text)
        est_output = estimate_tokens(result)

        print(f"[EST] Input tokens  : {est_input}")
        print(f"[EST] Output tokens : {est_output}")
        print(f"[EST] Total tokens  : {est_input + est_output}")

    print("[DEBUG] =========================\n")

    return result.strip(), usage.total_token_count

# =========================
# 7. 전체 파이프라인
# =========================
def ask(question, king):
    if king not in KINGS:
        raise ValueError(f"지원하지 않는 왕입니다: {king}. 사용 가능: {list(KINGS.keys())}")
    retrieved_docs = retrieve(question, king, k=2)
    prompt = build_prompt(question, retrieved_docs, king)
    answer = generate_answer(prompt, king)
    return answer
