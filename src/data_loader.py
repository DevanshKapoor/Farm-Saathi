# ==============================================================================
# data_loader.py
# ==============================================================================
import os
import glob
from pypdf import PdfReader
import warnings

def load_pdfs_and_chunk(database_path):
    """
    Loads all PDFs from a folder, extracts text, and chunks it.
    
    Args:
        database_path (str): The path to the folder containing PDF files.

    Returns:
        list: A list of text chunks (documents).
    """
    print(f"Scanning '{database_path}' for PDF files...")
    documents = []
    pdf_files = glob.glob(os.path.join(database_path, "*.pdf"))

    if not pdf_files:
        warnings.warn(f"⚠️ No PDF files found in '{database_path}'. The knowledge base will be empty.")
        return []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                if page.extract_text():
                    pdf_text += page.extract_text() + "\n"
            
            # Simple chunking by paragraph. Keep chunks longer than 50 chars.
            chunks = [para.strip() for para in pdf_text.split('\n\n') if len(para.strip()) > 50]
            documents.extend(chunks)
            print(f"    > Loaded & chunked '{os.path.basename(pdf_path)}' into {len(chunks)} chunks.")
        except Exception as e:
            print(f"❌ Error reading {pdf_path}: {e}")
            
    print(f"✅ Knowledge base created with {len(documents)} multilingual documents/chunks.")
    return documents