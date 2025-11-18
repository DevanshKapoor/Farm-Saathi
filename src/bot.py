# ==============================================================================
# bot.py
# ==============================================================================
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import warnings
from src.config import RETRIEVER_MODEL, RERANKER_MODEL, LLM_ID

warnings.filterwarnings("ignore")

class AgriSwarBot:
    def __init__(self, knowledge_base):
        """
        Initializes the bot, loads models, and builds the vector database.
        
        Args:
            knowledge_base (list): A list of text chunks from the data loader.
        """
        if not knowledge_base:
            raise ValueError("❌ CRITICAL: Knowledge Base is empty. Bot cannot be initialized.")
            
        self.knowledge_base = knowledge_base
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Bot initializing on device: {self.device}")

        self._load_models()
        self._build_vector_db()
        print("✅ All models loaded and vector DB built successfully!")

    def _load_models(self):
        """Loads the Retriever, Reranker, and LLM models."""
        print("    > Loading Retriever, Reranker, LLM...")
        self.retriever_model = SentenceTransformer(RETRIEVER_MODEL, device=self.device)
        self.reranker_model = CrossEncoder(RERANKER_MODEL, max_length=512, device=self.device)

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_ID, 
            quantization_config=quant_config, 
            device_map=self.device
        )

    def _build_vector_db(self):
        """Builds the FAISS vector database from the knowledge base."""
        print("    > Building vector database...")
        embeddings = self.retriever_model.encode(
            self.knowledge_base, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        self.vector_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.vector_index.add(embeddings.cpu().numpy())
        print("    ✅ Vector database built")

    def retrieve_and_rerank(self, query, top_k=5, rerank_top_n=3):
        """Retrieves and reranks relevant documents from the vector DB."""
        # 1. Fast Retrieval
        query_embedding = self.retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()
        _, indices = self.vector_index.search(query_embedding, top_k)
        initial_docs = [self.knowledge_base[i] for i in indices[0]]
        
        # 2. Accurate Reranking
        rerank_pairs = [[query, doc] for doc in initial_docs]
        scores = self.reranker_model.predict(rerank_pairs)
        
        # Combine docs with scores and sort
        doc_scores = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return the top N documents
        reranked_docs = [doc for doc, score in doc_scores[:rerank_top_n]]
        return reranked_docs, initial_docs

    def generate(self, query, context, language_name):
        """Generates a text answer using the LLM based on the context."""
        prompt = f"""<start_of_turn>user
You are an Indian agricultural expert. Answer the user's question concisely based on the context. Provide the answer in the {language_name} language only.

CONTEXT:
{" ".join(context)}

QUESTION:
{query}<end_of_turn>
<start_of_turn>model
"""
        # Tokenize and generate
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_length = inputs.input_ids.shape[1]
        
        outputs = self.llm_model.generate(
            **inputs, 
            max_new_tokens=150, 
            eos_token_id=self.llm_tokenizer.eos_token_id
        )
        generated_token_ids = outputs[0, input_token_length:]
        final_answer = self.llm_tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        return final_answer.strip()

    def run_pipeline(self, query_text, language_name):
        """Runs the complete text-to-text pipeline."""
        print(f"1. User Query: '{query_text}'")
        
        # 2. Retrieve & Rerank
        print("\n2. Retrieving and reranking context...")
        reranked_docs, _ = self.retrieve_and_rerank(query_text)
        if not reranked_docs:
            print("❌ No relevant context found.")
            return "मुझे इस सवाल का जवाब मेरे ज्ञानकोष में नहीं मिला।"

        print(f"    🔝 Top Reranked Context: {reranked_docs[0][:100]}...")
        
        # 3. Generate
        print("\n3. Generating text answer...")
        text_answer = self.generate(query_text, reranked_docs, language_name)
        print(f"    💬 Answer: '{text_answer}'")
        
        return text_answer

    def get_model_info(self):
        """Returns a dictionary of the bot's configuration."""
        info = {
            "device": self.device,
            "knowledge_base_size": len(self.knowledge_base),
            "retriever_model": RETRIEVER_MODEL,
            "reranker_model": RERANKER_MODEL,
            "llm_model": LLM_ID
        }
        return info