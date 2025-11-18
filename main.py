# ==============================================================================
# main.py
# ==============================================================================
# This is the main runnable file for the AgriSwar RAG Bot.
# It provides a command-line interface (CLI) to chat with the bot.

import os
import warnings
from huggingface_hub import login
import getpass

# Import our custom modules
from src.config import DATABASE_PATH, LANGUAGE_OPTIONS
from src.data_loader import load_pdfs_and_chunk
from src.bot import AgriSwarBot

warnings.filterwarnings("ignore")

def hf_login():
    """
    Handles Hugging Face login.
    Tries to log in using an environment variable, falling back to interactive.
    """
    print("🔐 Logging into Hugging Face Hub...")
    hf_token = os.environ.get("HF_TOKEN")
    
    if hf_token:
        try:
            login(token=hf_token)
            print("✅ Successfully logged in using HF_TOKEN environment variable!")
            return True
        except Exception as e:
            print(f"⚠️ Token login failed: {e}. Falling back to interactive login.")

    # Fallback to interactive login (getpass for secure token input)
    try:
        print("Please enter your Hugging Face Hub token (will be hidden):")
        token = getpass.getpass()
        login(token=token)
        print("✅ Successfully logged in interactively!")
        return True
    except Exception as e:
        print(f"❌ Login failed. Please set the HF_TOKEN environment variable or run `huggingface-cli login` in your terminal. Error: {e}")
        return False

def select_language():
    """
    Lets the user select a language from a numbered list.
    Returns the simple language name (e.g., "English").
    """
    print("\nPlease select a language for the bot's answers:")
    lang_map = list(LANGUAGE_OPTIONS.items())
    
    for i, (name, _) in enumerate(lang_map):
        print(f"  {i+1}. {name}")
    
    while True:
        try:
            choice = int(input(f"Enter number (1-{len(lang_map)}): "))
            if 1 <= choice <= len(lang_map):
                selected_display_name = lang_map[choice-1][0]
                selected_lang_name = selected_display_name.split(" ")[0]
                print(f"✅ Language set to {selected_display_name}.")
                return selected_lang_name
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(lang_map)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def print_model_info(bot):
    """Prints the bot's configuration info."""
    info = bot.get_model_info()
    print("\n📋 Current Model Configuration:")
    print("=" * 40)
    for key, value in info.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print("=" * 40)

def main_cli(bot: AgriSwarBot):
    """
    Runs the main command-line interface chat loop.
    """
    print("\n\n🚀🚀🚀 AgriSwar Bot (Command-Line) 🚀🚀🚀")
    print("Welcome! Type your question to get an answer.")
    print("\nAvailable commands:")
    print("  /lang   - Change the answer language")
    print("  /info   - Show bot's model information")
    print("  /exit   - Quit the application")
    
    current_lang = select_language()

    while True:
        query = input(f"\nYou ({current_lang}): ")
        
        if not query:
            continue
            
        # Handle commands
        if query.lower() == '/exit':
            print("👋 Goodbye!")
            break
            
        if query.lower() == '/info':
            print_model_info(bot)
            continue
        
        if query.lower() == '/lang':
            current_lang = select_language()
            continue
            
        # Run the bot's pipeline
        print("Bot is thinking...")
        final_answer = bot.run_pipeline(query, current_lang)
        
        print("\nBot:")
        print("="*50)
        print(final_answer)
        print("="*50)

# --- This is the main entry point of the script ---
if __name__ == "__main__":
    
    # 1. Login to Hugging Face
    if not hf_login():
        exit() # Exit if login fails

    # 2. Load Knowledge Base
    print("\n📚 Loading knowledge base from PDF documents...")
    KNOWLEDGE_BASE = load_pdfs_and_chunk(DATABASE_PATH)
    
    if not KNOWLEDGE_BASE:
        print(f"❌ CRITICAL: No documents found in '{DATABASE_PATH}'.")
        print("Please add PDF files to the 'database' folder and restart the application.")
    else:
        # 3. Initialize the Bot
        print("\n🚀 Initializing AgriSwarBot (this may take a moment)...")
        try:
            bot = AgriSwarBot(knowledge_base=KNOWLEDGE_BASE)
            print_model_info(bot)
            
            # 4. Start the CLI
            main_cli(bot)
            
        except Exception as e:
            print(f"❌ An error occurred during bot initialization: {e}")