# AgriSwar 4.0: PDF-Based RAG Bot

This is a multilingual, text-only RAG (Retrieval-Augmented Generation) bot designed to answer agricultural questions. It dynamically builds its knowledge base by reading all PDF documents (e.g., government schemes, pest advisories, university reports) from a `/database` folder.

It uses an advanced Retriever-Reranker-Generator (R-RAG) pipeline to provide accurate, context-aware answers in multiple languages.

## Features

* **Dynamic Knowledge Base:** Automatically ingests, chunks, and indexes all PDF files from the `./database` directory on startup.
* **Multilingual Support:** Uses `paraphrase-multilingual-mpnet-base-v2` to understand queries in multiple languages (Hindi, English, Punjabi, etc.).
* **Accurate R-RAG Pipeline:** Employs a fast **Retriever** (FAISS) and a more accurate **Reranker** (Cross-Encoder) to find the *most relevant* context before answering.
* **Efficient LLM:** Powered by a 4-bit quantized version of `google/gemma-2b-it` for high performance with low memory usage.
* **Interactive UI:** A simple, user-friendly interface built with `ipywidgets` for easy use within any Jupyter environment.

## How It Works: The R-RAG Pipeline

The bot's intelligence comes from its three-stage process to answer a question:



1.  **Retrieve:** When you ask a question (e.g., "What is the subsidy for SMAM?"), the bot first finds the top 5-10 *potentially* relevant text chunks from the entire PDF database using a fast vector search (FAISS).
2.  **Rerank:** This is the crucial step. A more powerful (but slower) Cross-Encoder model re-reads all 5-10 chunks and scores them for *actual relevance* to the query. This filters out irrelevant documents that just happened to have similar keywords.
3.  **Generate:** The top 3 *best* chunks are combined into a context. This context and your original question are sent to the Gemma LLM with the prompt: "You are an expert. Answer the QUESTION based *only* on the CONTEXT." This ensures the answer is factual and grounded in your documents.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/agriswar-rag-bot.git](https://github.com/your-username/agriswar-rag-bot.git)
cd agriswar-rag-bot


### 2. Install Requirements
All necessary libraries are listed in requirements.txt.

pip install -r requirements.txt


### 4. Log in to Hugging Face
The gemma-2b-it model is gated. You must be logged into a Hugging Face account that has been granted access.

huggingface-cli login
# Enter your HF_TOKEN when prompted

## Repo Structure

agriswar-rag-bot/
│
├── .gitignore          # Ignores system files
├── requirements.txt    # All Python dependencies
├── README.md           # This file
├── main.py             # The complete Python code for the bot and UI
│
└── database/
    ├── .gitkeep        # Ensures the folder is tracked by Git
    ├── doc1.pdf        # Add your first knowledge PDF here
    ├── doc2.pdf        # Add your second knowledge PDF here
    └── ...             # Add as many PDFs as you need
