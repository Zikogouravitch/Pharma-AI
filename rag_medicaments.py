import json
import faiss
import numpy as np
import unicodedata
import re
from sentence_transformers import SentenceTransformer

from rapidfuzz import fuzz
import requests

SYMPTOM_MAP = {
    "fièvre": [
        "PARACETAMOL",
        "IBUPROFENE",
        "ACIDE ACETYLSALICYLIQUE",
        "ASPIRINE"
    ],

    "mal de gorge": [
        "PARACETAMOL",
        "IBUPROFENE"
    ],

    "angine": [
        "PARACETAMOL",
        "IBUPROFENE"
    ],

    "toux grasse": [
        "AMBROXOL",
        "ACETYLCYSTEINE",
        "CARBOCISTEINE"
    ],

    "toux sèche": [
        "DEXTROMETHORPHANE",
        "CLOPERASTINE"
    ],

    "allergie": [
        "CETIRIZINE",
        "LEVOCETIRIZINE",
        "LORATADINE"
    ],

    "douleur": [
        "PARACETAMOL",
        "IBUPROFENE",
        "MELOXICAM",
        "PIROXICAM"
    ],

    "brulure estomac": [
        "OMEPRAZOLE",
        "ESOMEPRAZOLE",
        "LANSOPRAZOLE"
    ],

    "reflux": [
        "OMEPRAZOLE",
        "PANTOPRAZOLE"
    ]
}

BANNED_TERMS = [
    "VACCIN",
    "INJECTABLE",
    "PERFUSION",
    "IMPLANT",
    "LYOPHILISAT"
]

import pytesseract
from PIL import Image
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



def normalize_text(text):
    text = text.lower()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def format_medicament(med):
    return f"""
Nom: {med.get('drug_name') or 'N/A'}
Présentation: {med.get('presentation') or 'N/A'}
Fabricant: {med.get('manufacturer') or 'N/A'}
Substance active: {', '.join(med.get('composition') or [])}
Classe thérapeutique: {med.get('therapeutic_class') or 'N/A'}
Statut: {med.get('status') or 'N/A'}
Prix: {(med.get('price') or {}).get('ppv') or 'N/A'} MAD
Indications: {', '.join(med.get('indications') or []) if med.get('indications') else 'Non spécifié'}
""".strip()

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w] 

    
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced
def extract_text_from_image(image_path):
    try:
        processed = preprocess_image(image_path)

        results = []
        for psm in [6, 11, 3]:
            config = f"--oem 3 --psm {psm} -l fra+eng"
            text = pytesseract.image_to_string(processed, config=config)
            text = normalize_text(text)
            if len(text) > len(results[-1] if results else ""):
                results.append(text)

        return max(results, key=len) if results else None

    except Exception as e:
        print(f"Erreur OCR : {e}")
        return None
        


def detect_drug_from_image(image_path, data, texts, index):
    print("🔍 Extraction du texte depuis l'image...")
    extracted_text = extract_text_from_image(image_path)

    if not extracted_text or len(extracted_text) < 3:
        return None, "Impossible d'extraire du texte."

    print(f"📝 Texte OCR : [{extracted_text}]")

    words = [w for w in extracted_text.split() if len(w) >= 4]

    best_match = None
    best_score = 0

    for i, med in enumerate(data):
        med_name = normalize_text(med["drug_name"])
        first_word = med_name.split()[0]

        for word in words:
            score = fuzz.ratio(word, first_word)
            if score > best_score:
                best_score = score
                best_match = i

    print(f"🎯 Meilleur score : {best_score} → {data[best_match]['drug_name'] if best_match is not None else 'Rien'}")

    if best_score >= 75 and best_match is not None:
        return texts[best_match], extracted_text

    docs = search(extracted_text, data, texts, index)
    return (docs[0] if docs else None), extracted_text


print("🔄 Chargement du modèle...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts):
    return model.encode(texts)


def create_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def detect_symptom(query):
    q = normalize_text(query)

    for med in data:
        med_name = normalize_text(med["drug_name"])
        if med_name in q or q in med_name:
            return None

    best_symptom = None
    best_score = 0

    for symptom in SYMPTOM_MAP:
        score = fuzz.token_set_ratio(q, normalize_text(symptom))
        if score > best_score:
            best_score = score
            best_symptom = symptom

    if best_score >= 70: 
        return best_symptom

    return None

def is_invalid_for_reco(med):
    txt = normalize_text(
        (med.get("presentation") or "") + " " +
        " ".join(med.get("composition") or []) + " " +
        (med.get("therapeutic_class") or "")
    )
    return any(normalize_text(term) in txt for term in BANNED_TERMS)

def filter_by_symptom(symptom, data):
    allowed_substances = SYMPTOM_MAP[symptom]
    scored = []

    for med in data:
        substance = " ".join(med.get("composition") or []).upper()

        if is_invalid_for_reco(med):
            continue

        for rank, target in enumerate(allowed_substances):
            score_match = fuzz.partial_ratio(target, substance)
            if score_match >= 80:
                score = 100 - rank * 10
                scored.append((score, med))
                break

    scored.sort(reverse=True, key=lambda x: x[0])
    return [med for score, med in scored]

def looks_like_gibberish(text):
    vowels = set("aeiouy")
    count = sum(1 for c in text.lower() if c in vowels)

    return count == 0
def search(query, data, texts, index, k=3):
    query_norm = normalize_text(query)

    if len(query_norm) < 3:
        return []

    if looks_like_gibberish(query_norm):
        return []
    SMALL_TALK = ["hello", "salut", "bonjour", "hi"]

    if query_norm in SMALL_TALK:
       return ["Bonjour  Comment puis-je vous aider concernant un médicament ?"]
    
    best_match = None
    best_length = 0

    for i, med in enumerate(data):
        med_name = normalize_text(med["drug_name"])

        if med_name in query_norm:
            if len(med_name) > best_length:
                best_match = i
                best_length = len(med_name)

    if best_match is not None:
        return [texts[best_match]]

    
    symptom = detect_symptom(query)

    if symptom:
        filtered = filter_by_symptom(symptom, data)

        if filtered:
            return [format_medicament(filtered[0])]

    
    q_emb = model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)

    if distances[0][0] > 1.2:
        return []

    return [texts[indices[0][0]]]


def generate_answer(query, docs):
    if not docs:
        return "Je ne sais pas."

    context = docs[0]

    prompt = f"""
Tu es PharmaBot, un assistant pharmaceutique professionnel.

RÈGLES IMPORTANTES :

1. Réponds uniquement avec les informations du contexte fourni.
2. Réponds toujours en français naturel, clair et professionnel.
3. Ne dis jamais que tu n'es pas médecin ou assistant.
4. Si la question est vague, demande une précision.
5. Ne jamais inventer d'information.
6. Si information absente du contexte, dis :
Je ne sais pas.
7. Si symptôme inquiétant ou douleur forte :
conseille de consulter un professionnel de santé.
8. Donne des réponses courtes et utiles.
9. Si un médicament est trouvé, explique son usage simplement.

QUESTION UTILISATEUR :
{query}

CONTEXTE :
{context}

RÉPONSE :
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }
)

        return response.json()["response"]

    except Exception as e:
        return f"Erreur Ollama : {str(e)}"


data = load_data("medicaments.jsonl")

data = sorted(
    data,
    key=lambda x: len(x["drug_name"]),
    reverse=True
)

texts = [format_medicament(m) for m in data]

print(" Création des embeddings...")
embeddings = embed(texts)

print(" Création de l'index...")
index = create_index(embeddings)


if __name__ == "__main__":
    print(" Système prêt ! Pose tes questions ")
    print(" Tape 'image' pour analyser une photo de médicament\n")

    while True:
        query = input(" Question: ").strip()

        if normalize_text(query) in ["exit", "quit"]:
            break

        # Mode image
        if normalize_text(query) == "image":
            image_path = input(" Chemin de l'image : ").strip()

            doc, extracted = detect_drug_from_image(image_path, data, texts, index)

            if doc:
                answer = generate_answer(
                    f"Identifie ce médicament et explique son usage : {extracted}",
                    [doc]
                )
                print("\n Réponse :", answer)
            else:
                print(" Médicament non identifié depuis l'image.")

            continue

        
        docs = search(query, data, texts, index)
        answer = generate_answer(query, docs)
        print("\n", answer)