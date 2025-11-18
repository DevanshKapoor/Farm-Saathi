# ==============================================================================
# config.py
# ==============================================================================
# This file stores all configurations, model IDs, and paths.

# --- Paths ---
# Path to the folder containing your PDF knowledge base
DATABASE_PATH = "./database" 

# --- Model IDs ---
RETRIEVER_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_ID = "google/gemma-2b-it"

# --- UI Options ---
LANGUAGE_OPTIONS = {
    "Hindi (हिन्दी)": "hi",
    "English": "en",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn"
}