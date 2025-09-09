# -*- coding: utf-8 -*-
"""
LegalMate – AI Legal Assistant (Hugging Face + FAISS)
說明：
- 從 data/sample_clauses.json 載入條文
- 使用 sentence-transformers 產生向量，做單位向量正規化（cosine）
- 以 FAISS IndexFlatIP（內積）檢索最相關條文
- 用 Hugging Face 的 Flan-T5 產生條文解釋
- 以 Gradio 提供簡易網頁介面
"""

import json
import numpy as np
import faiss
import gradio as gr
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------------
# 路徑處理：本機有 __file__，Colab 沒有
# ------------------------------
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "data" / "sample_clauses.json"

# ------------------------------
# 讀取 JSON 內容
# ------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Sample data not found at {DATA_PATH}. Please ensure data/sample_clauses.json exists.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    legal_docs = json.load(f)

if not isinstance(legal_docs, list) or not all(isinstance(x, str) for x in legal_docs):
    raise ValueError("data/sample_clauses.json 必須是一個只包含字串的 list。")

# ------------------------------
# 建立 Embeddings + FAISS 索引（cosine 相似度做法）
# ------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = embedder.encode(
    legal_docs,
    convert_to_numpy=True,        # 確保拿到 numpy，而不是 list
    normalize_embeddings=True     # 把向量正規化（單位長度 = 1）
).astype("float32")               # FAISS 只吃 float32

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # 用內積 (Inner Product)，配合正規化等於 cosine
index.add(embeddings)

# ------------------------------
# 文字生成模型（建議先用 base；要更強可改 large）
# ------------------------------
GEN_MODEL = "google/flan-t5-base"  # 可改為 "google/flan-t5-large"
qa_pipeline = pipeline("text2text-generation", model=GEN_MODEL)

# ------------------------------
# 問答函式
# ------------------------------
def search_and_answer(user_question: str) -> str:
    try:
        if not user_question or not user_question.strip():
            return "請輸入問題 / Please enter a question."

        query_vec = embedder.encode([user_question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_vec, k=1)
        top_idx = int(indices[0][0])
        matched_clause = legal_docs[top_idx]

        prompt = (
            "You are a legal assistant for a U.S. law firm. "
            "Explain the following contract clause in clear, professional, plain English.\n\n"
            f"Clause:\n\"{matched_clause}\"\n\n"
            f"Question:\n{user_question}\n\n"
            "Answer:"
        )

        response = qa_pipeline(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"].strip()
        return "📄 Matched clause:\n" + matched_clause + "\n\n" + "💡 Answer:\n" + response

    except Exception as e:
        return f"❌ Error: {type(e).__name__}: {e}"

# ------------------------------
# Gradio 介面
# ------------------------------
import gradio as gr

def launch_app(share=False, debug=False):
    demo = gr.Interface(
        fn=search_and_answer,
        inputs=gr.Textbox(lines=2, label="Enter your legal question:"),
        outputs=gr.Textbox(label="AI Answer"),
        title="LegalMate – AI Legal Assistant (Hugging Face Version)"
    )
    demo.launch(share=share, debug=debug)

if __name__ == "__main__":
    # 本機建議 share=False；Colab 想要外網連結就改成 True
    launch_app(share=True, debug=True)
