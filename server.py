from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

app = FastAPI()

# ==============================
# 🌐 CORS (IMPORTANT)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🔑 GROQ CLIENT
# ==============================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ Set GROQ_API_KEY")

client = Groq(api_key=api_key)

# ==============================
# 🧠 MEMORY (LEVEL 2)
# ==============================
chat_memory = []

# ==============================
# 📚 KNOWLEDGE BASE (LEVEL 4)
# ==============================
docs = [
    "Paras Jain is an Integrated M.Tech student in Computer Science (Computational and Data Science) at VIT Bhopal with GPA 7.56.",
    "Paras Jain is NOT a student of IIT Bombay but contributed to IIT Bombay FOSSEE Osdag project.",
    "Paras Jain built AI Gym Trainer using OpenCV and MediaPipe running at 20 FPS.",
    "Paras Jain built a Healthcare system using Django with JWT authentication and RBAC.",
    "Paras Jain built an eSim tool manager CLI for IIT Bombay supporting Linux, Windows, macOS.",
    "Paras Jain skills include Python, C++, TensorFlow, PyTorch, OpenCV, Django, REST APIs, AWS, GCP.",
    "Paras Jain GitHub is https://github.com/parasjain238",
    "Paras Jain LinkedIn is https://linkedin.com/in/paras-jain-148b97297"
]

# ==============================
# 🔍 EMBEDDINGS SETUP
# ==============================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(docs)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))

# ==============================
# 📦 REQUEST MODEL
# ==============================
class Req(BaseModel):
    message: str

# ==============================
# 🔎 SEARCH FUNCTION
# ==============================
def search_context(query):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k=3)
    return "\n".join([docs[i] for i in I[0]])

# ==============================
# 🤖 SYSTEM PROMPT
# ==============================
SYSTEM_PROMPT = """
You are Paras Jain's AI assistant.

RULES:
- Only answer from provided context
- Never hallucinate
- If not found → say "Not mentioned in portfolio"
- Paras is NOT a student of IIT Bombay
- Keep answers short (2–3 lines max)
- Speak professionally and confidently
"""

# ==============================
# 🧠 MODELS (LEVEL 3)
# ==============================
MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768"
]

# ==============================
# 🚀 CHAT API
# ==============================
@app.post("/chat")
async def chat(req: Req):
    try:
        user_msg = req.message
        print("User:", user_msg)

        # 🔥 MEMORY UPDATE
        chat_memory.append({"role": "user", "content": user_msg})
        if len(chat_memory) > 6:
            chat_memory.pop(0)

        # 🔥 CONTEXT FROM EMBEDDINGS
        context = search_context(user_msg)

        # 🔥 BUILD MESSAGES
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Context:\n{context}"}
        ] + chat_memory

        last_error = None

        # 🔥 TRY MULTIPLE MODELS
        for model_name in MODELS:
            try:
                print(f"Trying model: {model_name}")

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=300
                )

                if response and response.choices:
                    reply = response.choices[0].message.content.strip()

                    # 🔥 SAVE AI RESPONSE TO MEMORY
                    chat_memory.append({"role": "assistant", "content": reply})

                    print("AI:", reply)

                    return {
                        "reply": reply
                    }

            except Exception as e:
                print(f"❌ Model failed: {model_name} → {e}")
                last_error = e
                continue

        # ❌ ALL MODELS FAILED
        return {
            "reply": "⚠️ AI temporarily unavailable. Try again."
        }

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return {"reply": f"❌ Server error: {str(e)}"}


# ==============================
# 🧪 ROUTES
# ==============================
@app.get("/")
def home():
    return {"message": "🔥 FINAL AI SERVER RUNNING"}

@app.get("/memory")
def memory():
    return {"memory": chat_memory}