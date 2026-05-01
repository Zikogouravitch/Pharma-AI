from fastapi import Form, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # ← Ajouter
from pydantic import BaseModel
import shutil
import os

from rag_medicaments import (
    search,
    generate_answer,
    init_rag,
    detect_drug_from_image
)

app = FastAPI()

# ← Ajouter ce bloc
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou mettez l'URL exacte de votre app ASP.NET
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data, texts, index = init_rag()
class QuestionRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(
    question: str = Form(None),
    file: UploadFile = File(None)
):
    # ❌ Cas interdit
    if question and file:
        return {"response": "Veuillez envoyer soit une image soit un texte, pas les deux."}

    # ❌ Rien envoyé
    if not question and not file:
        return {"response": "Veuillez entrer une question ou envoyer une image."}

    # ✅ Cas IMAGE
    if file:
        path = f"temp_{file.filename}"

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc, extracted = detect_drug_from_image(path, data, texts, index)
        os.remove(path)

        if doc:
            answer = generate_answer(
                f"Identifie ce médicament et explique son usage : {extracted}",
                [doc]
            )
            return {"response": answer}

        return {"response": "Médicament non reconnu"}

    # ✅ Cas TEXTE
    if question:
        docs = search(question, data, texts, index)
        answer = generate_answer(question, docs)
        return {"response": answer}