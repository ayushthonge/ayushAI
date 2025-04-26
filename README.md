# WhatsApp RAG AI

A Retrieval-Augmented Generation (RAG) system that analyzes your WhatsApp chat history and generates highly personalized, context-aware responses in your own communication style.

---

## Features

- **WhatsApp Chat Ingestion:** Parses exported WhatsApp `.txt` chats and processes them into structured data.
- **Vector Database:** Embeds and stores chat chunks for fast semantic retrieval (via ChromaDB).
- **Style Mimicry:** Extracts your unique communication style from your messages for response generation.
- **Contact-Aware Replies:** Generates responses tailored to specific contacts using their conversation history.
- **LLM Integration:** Uses Gemini (or any compatible LLM) for advanced language generation.
- **Quote Avoidance:** Prevents verbatim quoting of your past messages, ensuring originality.
- **Interactive Chat Interface:** Test your AI in a command-line chat loop.

---

## Project Structure

ayushai/
├── src/
│ ├── chat_interface.py
│ ├── rag_model.py
│ ├── style_analyzer.py
│ ├── content_planner.py
│ ├── response_generator.py
│ ├── quote_detector.py
│ ├── data_prep.py
│ ├── clean_data.py
│ └── ...
├── data/
│ ├── raw/ # Place your WhatsApp .txt exports here
│ └── processed/ # Processed CSVs (auto-generated)
├── vector_db/ # Vector databases (auto-generated, DO NOT COMMIT)
├── .env # Your API keys (DO NOT COMMIT)
├── config.json # Configuration (DO NOT COMMIT if it has secrets)
├── requirements.txt
└── README.md


---

## Setup Instructions

### 1. **Clone the Repository**

git clone https://github.com/yourusername/your-repo.git
cd your-repo


### 2. **Create and Activate a Virtual Environment**

python -m venv venv

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate


### 3. **Install Dependencies**

pip install -r requirements.txt


### 4. **Add Your API Key**

- Create a `.env` file in the project root:

GEMINI_API_KEY=your-gemini-api-key

- **Never commit your `.env` file.**

### 5. **Prepare Your Data**

- Export your WhatsApp chats as `.txt` files and place them in `data/raw/`.

### 6. **Process and Vectorize Data**

python -m src.data_prep --input data/raw --output data/processed --name "Your Name"
python -m src.clean_data
python -m src.vectorize_chats


### 7. **Run the Chat Interface**

python -m src.chat_interface


---

## Usage

- Type messages to see how the AI would reply in your style.
- Use commands like `contacts`, `contact [name]`, and `exit` in the chat interface.

---

## Security & Privacy

- **Never commit your `.env`, `config.json` (if it has secrets), or any raw/processed chat data.**
- All sensitive files are listed in `.gitignore`.

---

## License

MIT License

---

## Disclaimer

This project is for educational and personal productivity use. Do **not** use it to impersonate others or automate messaging without consent.

---

