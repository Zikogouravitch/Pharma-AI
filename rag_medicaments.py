import json
import faiss
import numpy as np
import unicodedata
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import fuzz
import requests
import os
import pickle

EMBEDDINGS_PATH = "embeddings.npy"
INDEX_PATH = "faiss.index"
TEXTS_PATH = "texts.pkl"

SYMPTOM_EMBEDDINGS_PATH = "symptom_embeddings.pkl"
SIMILARITY_THRESHOLD = 0.25   # seuil sémantique abaissé (requêtes courtes)
FUZZY_THRESHOLD       = 70    # seuil fuzzy sur les mots-clés des indications

BANNED_TERMS = [
    "VACCIN",
    "INJECTABLE",
    "PERFUSION",
    "IMPLANT",
    "LYOPHILISAT"
]

# ── Salutations & remerciements ────────────────────────────────────────────────
GREETINGS = [
    "bonjour", "salut", "hello", "hi", "bonsoir", "coucou",
    "hey", "salam", "ahlan", "bjr"
]

THANKS = [
    "merci", "thanks", "thank you", "شكرا", "merci beaucoup",
    "merci bien", "parfait merci", "ok merci", "super merci"
]

INTENT_GREETING = "__GREETING__"
INTENT_THANKS   = "__THANKS__"

# ── OCR ────────────────────────────────────────────────────────────────────────
import pytesseract
from PIL import Image
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── Persistance RAG principal ──────────────────────────────────────────────────
def save_rag(embeddings, index, texts):
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    print("💾 Embeddings et index sauvegardés !")


def load_rag():
    if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH)
            and os.path.exists(TEXTS_PATH)):
        return None, None, None
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode='r')
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    print("⚡ Embeddings et index chargés depuis le disque !")
    return embeddings, index, texts


SYMPTOM_INDEX_VERSION = 3   # à incrémenter si le format change

# ── Persistance index symptômes ────────────────────────────────────────────────
def save_symptom_index(med_index):
    payload = {"version": SYMPTOM_INDEX_VERSION, "index": med_index}
    with open(SYMPTOM_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(payload, f)
    print("💾 Index symptômes sauvegardé !")


def load_symptom_index():
    if not os.path.exists(SYMPTOM_EMBEDDINGS_PATH):
        return None
    try:
        with open(SYMPTOM_EMBEDDINGS_PATH, "rb") as f:
            payload = pickle.load(f)

        # ancien format = liste directe (pas de dict avec version)
        if not isinstance(payload, dict) or payload.get("version") != SYMPTOM_INDEX_VERSION:
            print("⚠️  Index symptômes obsolète — reconstruction en cours...")
            os.remove(SYMPTOM_EMBEDDINGS_PATH)
            return None

        med_index = payload["index"]
        # vérification rapide du format des tuples
        if med_index and len(med_index[0]) != 4:
            print("⚠️  Format de l'index incorrect — reconstruction en cours...")
            os.remove(SYMPTOM_EMBEDDINGS_PATH)
            return None

        print("⚡ Index symptômes chargé depuis le disque !")
        return med_index

    except Exception as e:
        print(f"⚠️  Erreur lecture index symptômes ({e}) — reconstruction...")
        os.remove(SYMPTOM_EMBEDDINGS_PATH)
        return None


# ── Utilitaires texte ──────────────────────────────────────────────────────────
def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_gibberish(text):
    vowels = set("aeiouy")
    count = sum(1 for c in text.lower() if c in vowels)
    return count == 0


def is_greeting(query: str) -> bool:
    q = normalize_text(query)
    return any(normalize_text(g) == q or normalize_text(g) in q.split()
               for g in GREETINGS)


def is_thanks(query: str) -> bool:
    q = normalize_text(query)
    q_words = set(q.split())
    for t in THANKS:
        nt = normalize_text(t)
        if not nt:          # ignore tokens arabes/non-latins qui deviennent vides
            continue
        nt_words = set(nt.split())
        # correspondance mot-à-mot pour éviter les faux positifs par sous-chaîne
        if nt_words and nt_words.issubset(q_words):
            return True
    return False


# ── Chargement données ─────────────────────────────────────────────────────────
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


# ── Modèles d'embedding ────────────────────────────────────────────────────────
def load_model_offline(model_name: str) -> SentenceTransformer:
    """Charge le modèle depuis le cache local si le réseau est indisponible."""
    try:
        m = SentenceTransformer(model_name)
        print(f"✅ Modèle '{model_name}' chargé (online/cache).")
        return m
    except Exception:
        print(f"⚠️  Réseau indisponible — chargement local forcé pour '{model_name}'...")
        m = SentenceTransformer(model_name, local_files_only=True)
        print(f"✅ Modèle '{model_name}' chargé depuis le cache local.")
        return m

print("🔄 Chargement du modèle principal...")
model = load_model_offline("all-MiniLM-L6-v2")

print("🔄 Chargement du modèle symptômes...")
symptom_model = load_model_offline("all-MiniLM-L6-v2")


# ── Patterns de négation dans les indications ─────────────────────────────────
_NEGATION_PATTERNS = re.compile(
    r"(sans|en l absence de|ne pas utiliser en cas de|"
    r"contre.indiqu|non recommand|interdit en cas de)\s+(\w+)",
    re.IGNORECASE
)

# ── Index symptômes (hybride : fuzzy + sémantique) ─────────────────────────────
def build_symptom_index(data: list) -> list:
    """
    Pour chaque médicament avec des indications, stocke :
      - le dict médicament
      - le vecteur sémantique des indications
      - les mots-clés positifs normalisés (pour fuzzy)
      - les mots-clés négatifs (symptômes explicitement exclus)
    """
    med_index = []
    for med in data:
        indications = med.get("indications") or []
        if not indications:
            continue
        full_text = " ".join(indications)
        norm_text = normalize_text(full_text)

        # mots négatifs : extraits des segments de négation
        negative_kw = set()
        for m in _NEGATION_PATTERNS.finditer(full_text):
            negative_kw.add(normalize_text(m.group(2)))

        # mots positifs : tous les mots sauf ceux dans les segments négatifs
        all_words = {w for w in norm_text.split() if len(w) >= 4}
        positive_kw = list(all_words - negative_kw)

        vec = symptom_model.encode(full_text, normalize_embeddings=True)
        med_index.append((med, vec, positive_kw, negative_kw))

    print(f"✅ Index symptômes : {len(med_index)} médicaments indexés.")
    return med_index


def _fuzzy_score(query_words: list, keywords: list) -> float:
    """Score fuzzy moyen entre les mots de la requête et une liste de mots-clés."""
    if not keywords or not query_words:
        return 0.0
    return sum(max(fuzz.ratio(qw, kw) for kw in keywords)
               for qw in query_words) / len(query_words)


def search_by_symptom(query: str, top_k: int = 3,
                      sem_threshold: float = SIMILARITY_THRESHOLD,
                      fuzzy_threshold: float = FUZZY_THRESHOLD) -> list:
    """
    Recherche hybride :
      1. Pénalité  si le symptôme apparaît dans les mots négatifs des indications
      2. Score fuzzy  sur les mots-clés POSITIFS des indications  (poids 0.4)
      3. Score cosinus sémantique sur le texte complet             (poids 0.4)
      4. Bonus classe thérapeutique / ATC                          (poids 0.2)
    """
    q_norm  = normalize_text(query)
    q_words = [w for w in q_norm.split() if len(w) >= 3]
    q_vec   = symptom_model.encode(query, normalize_embeddings=True).reshape(1, -1)

    results = []
    for med, vec, positive_kw, negative_kw in med_index:
        if is_invalid_for_reco(med):
            continue

        # ── Pénalité négation ───────────────────────────────────────────────
        # Si un mot de la requête matche fortement un mot négatif → on écarte
        if negative_kw:
            neg_score = _fuzzy_score(q_words, list(negative_kw))
            if neg_score >= 80:
                continue

        sem_score = float(cosine_similarity(q_vec, vec.reshape(1, -1))[0][0])
        fuzzy_sc  = _fuzzy_score(q_words, positive_kw)

        if sem_score < sem_threshold and fuzzy_sc < fuzzy_threshold:
            continue

        # ── Bonus classe thérapeutique ──────────────────────────────────────
        tc_bonus = 0.0
        tc = normalize_text(med.get("therapeutic_class") or "")
        if any(fuzz.partial_ratio(qw, tc) >= 80 for qw in q_words):
            tc_bonus = 0.1

        combined = (sem_score * 0.4 + (fuzzy_sc / 100) * 0.4 + tc_bonus * 0.2)
        results.append((med, combined))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def format_symptom_result(results: list) -> str:
    """
    Formate les résultats de search_by_symptom en texte structuré
    pour les agents suivants (RAG, sécurité, rédacteur).
    """
    if not results:
        return ""

    best_med, best_score = results[0]
    lines = [
        f"Médicament recommandé (similarité={best_score:.2f}) :",
        f"Nom         : {best_med['drug_name']}",
        f"Composition : {', '.join(best_med.get('composition', []))}",
        f"Classe      : {best_med.get('therapeutic_class', 'N/A')}",
        f"Prix PPV    : {best_med.get('price', {}).get('ppv', 'N/A')} MAD",
        f"Indication  : {best_med['indications'][0]}",
    ]

    if len(results) > 1:
        lines.append("\nAlternatives :")
        for med, sc in results[1:]:
            lines.append(f"  - {med['drug_name']} (score={sc:.2f})")

    return "\n".join(lines)


def is_invalid_for_reco(med):
    txt = normalize_text(
        (med.get("presentation") or "") + " " +
        " ".join(med.get("composition") or []) + " " +
        (med.get("therapeutic_class") or "")
    )
    return any(normalize_text(term) in txt for term in BANNED_TERMS)


# ── OCR ────────────────────────────────────────────────────────────────────────
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
    return clahe.apply(denoised)


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

    print(f"🎯 Meilleur score : {best_score} → "
          f"{data[best_match]['drug_name'] if best_match is not None else 'Rien'}")

    if best_score >= 75 and best_match is not None:
        return texts[best_match], extracted_text

    docs = search(extracted_text, data, texts, index)
    return (docs[0] if docs else None), extracted_text


# ── RAG principal ──────────────────────────────────────────────────────────────
def embed(texts):
    return model.encode(texts)


def create_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


def search(query, data, texts, index, k=3):
    # ── 1. Salutations ──────────────────────────────────────────────────────
    if is_greeting(query):
        return [INTENT_GREETING]

    # ── 2. Remerciements ────────────────────────────────────────────────────
    if is_thanks(query):
        return [INTENT_THANKS]

    query_norm = normalize_text(query)

    if len(query_norm) < 3 or looks_like_gibberish(query_norm):
        return []

    # ── 3. Correspondance sur le nom du médicament ─────────────────────────
    # Cherche le 1er mot du nom de marque dans la requête (ex: "doliprane")
    best_match = None
    best_length = 0
    best_fuzzy  = 0

    for i, med in enumerate(data):
        med_name      = normalize_text(med["drug_name"])
        first_word    = med_name.split()[0]          # "doliprane", "paracetamol"…

        # correspondance exacte : nom complet contenu dans la requête
        if med_name in query_norm:
            if len(med_name) > best_length:
                best_match  = i
                best_length = len(med_name)
            continue

        # correspondance fuzzy sur le premier mot (nom de marque)
        if len(first_word) >= 4:
            for qw in query_norm.split():
                score = fuzz.ratio(qw, first_word)
                if score > best_fuzzy and score >= 82:
                    best_fuzzy = score
                    if best_match is None or score > best_fuzzy:
                        best_match = i

    if best_match is not None:
        return [texts[best_match]]

    # ── 4. Recherche par symptôme via index vectoriel ────────────────────────
    symptom_results = search_by_symptom(query)
    if symptom_results:
        formatted = format_symptom_result(symptom_results)
        if formatted:
            return [formatted]

    # ── 5. Recherche sémantique FAISS (fallback) ─────────────────────────────
    q_emb = model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)

    if distances[0][0] > 1.2:
        return []

    return [texts[indices[0][0]]]


# ── Génération de la réponse (Ollama / Mistral) ────────────────────────────────
def generate_answer(query, docs):
    if not docs:
        return "Je ne sais pas."

    context = docs[0]

    # ── Salutations : délégué à Mistral ────────────────────────────────────
    if context == INTENT_GREETING:
        prompt = (
            "Tu es PharmaBot, un assistant pharmaceutique professionnel. "
            "L'utilisateur vient de te saluer. Réponds chaleureusement en français, "
            "présente-toi brièvement et invite-le à poser sa question "
            "(médicament, symptôme, ordonnance…). Sois court et naturel."
        )
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt,
                      "stream": False, "options": {"temperature": 0.7}}
            )
            return response.json()["response"]
        except Exception as e:
            return f"Erreur Ollama : {str(e)}"

    # ── Remerciements : délégué à Mistral ──────────────────────────────────
    if context == INTENT_THANKS:
        prompt = (
            "Tu es PharmaBot, un assistant pharmaceutique professionnel. "
            "L'utilisateur vient de te remercier. Réponds de façon chaleureuse et bienveillante "
            "en français, encourage-le à revenir si besoin. Sois court et naturel."
        )
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt,
                      "stream": False, "options": {"temperature": 0.7}}
            )
            return response.json()["response"]
        except Exception as e:
            return f"Erreur Ollama : {str(e)}"

    prompt = f"""
Tu es PharmaBot, un assistant pharmaceutique professionnel.

RÈGLES IMPORTANTES :
1. Réponds uniquement avec les informations du contexte fourni.
2. Réponds toujours en français naturel, clair et professionnel.
3. Ne dis jamais que tu n'es pas médecin ou assistant.
4. Si la question est vague, demande une précision.
5. Ne jamais inventer d'information.
6. Si information absente du contexte, dis : Je ne sais pas.
7. Si symptôme inquiétant ou douleur forte : conseille de consulter un professionnel de santé.
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
                "options": {"temperature": 0.2, "top_p": 0.9}
            }
        )
        return response.json()["response"]
    except Exception as e:
        return f"Erreur Ollama : {str(e)}"


# ── Initialisation ─────────────────────────────────────────────────────────────
data = load_data("medicaments.jsonl")
data = sorted(data, key=lambda x: len(x["drug_name"]), reverse=True)
texts = [format_medicament(m) for m in data]

# Index RAG principal
embeddings, index, saved_texts = load_rag()
if embeddings is None:
    print("🔄 Création des embeddings principaux...")
    embeddings = embed(texts)
    index = create_index(embeddings)
    save_rag(embeddings, index, texts)
else:
    texts = saved_texts

# Index symptômes vectoriel
med_index = load_symptom_index()
if med_index is None:
    print("🔄 Construction de l'index symptômes...")
    med_index = build_symptom_index(data)
    save_symptom_index(med_index)


# ── Point d'entrée ─────────────────────────────────────────────────────────────
def init_rag():
    global data, texts, index, med_index

    data = load_data("medicaments.jsonl")
    data = sorted(data, key=lambda x: len(x["drug_name"]), reverse=True)
    texts = [format_medicament(m) for m in data]

    embeddings = embed(texts)
    index = create_index(embeddings)

    med_index = build_symptom_index(data)

    return data, texts, index


if __name__ == "__main__":
    print("✅ Système prêt ! Pose tes questions")
    print("   Tape 'image' pour analyser une photo de médicament\n")

    while True:
        query = input("💬 Question: ").strip()

        if normalize_text(query) in ["exit", "quit"]:
            break

        # Mode image
        if normalize_text(query) == "image":
            image_path = input("📁 Chemin de l'image : ").strip()
            doc, extracted = detect_drug_from_image(image_path, data, texts, index)
            if doc:
                answer = generate_answer(
                    f"Identifie ce médicament et explique son usage : {extracted}",
                    [doc]
                )
                print("\n🤖 Réponse :", answer)
            else:
                print("❌ Médicament non identifié depuis l'image.")
            continue

        docs = search(query, data, texts, index)
        answer = generate_answer(query, docs)
        print("\n🤖", answer)