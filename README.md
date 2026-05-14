# 🤖 Nova by Lumio — AI Sales Agent

A fully local, RAG-powered conversational sales agent with a WhatsApp-style UI. Built with Python and Streamlit — zero GPT API costs, zero cloud dependency.

---

## ✨ Features

- 💬 **WhatsApp-style chat UI** — custom CSS bubbles, tails, timestamps, read ticks
- 🧠 **RAG pipeline** — answers questions from your own knowledge base
- 🎯 **Intent detection** — routes greeting / pricing / high-intent / unknown queries
- 📋 **Lead capture** — 3-step conversational flow (name → email → platform)
- 📧 **Auto welcome email** — fires instantly via SendGrid after lead capture
- 💾 **Local storage** — leads saved to CSV, vectors stored in ChromaDB
- 🔒 **Fully local LLM** — runs on phi3 via Ollama, no OpenAI key needed

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| UI | Streamlit + custom CSS |
| Intent Detection | Keyword matching (Python) |
| RAG | LangChain + HuggingFace MiniLM-L6-v2 + ChromaDB |
| LLM | Ollama (phi3) |
| Email | SendGrid API |
| Storage | CSV + ChromaDB (local) |

---

## 📁 Project Structure

```
nova-by-lumio/
├── streamlit_app.py       # Main app
├── knowledge_base.txt     # Your product knowledge (edit this)
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nova-by-lumio.git
cd nova-by-lumio
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and fill in your values:
```
SENDGRID_API_KEY=your_sendgrid_api_key_here
SENDER_EMAIL=your_verified_sender@email.com
```

### 5. Install and run Ollama
- Download Ollama from [ollama.ai](https://ollama.ai)
- Pull the phi3 model:
```bash
ollama pull phi3
```

### 6. Add your knowledge base
Edit `knowledge_base.txt` with your product info — pricing, features, FAQs etc.

### 7. Run the app
```bash
streamlit run streamlit_app.py
```

---

## 📧 SendGrid Setup

1. Sign up at [sendgrid.com](https://sendgrid.com)
2. Go to **Settings → Sender Authentication → Single Sender Verification**
3. Verify your sender email
4. Go to **Settings → API Keys → Create API Key**
5. Copy the key into your `.env` file

---

## 🔒 Environment Variables

| Variable | Description |
|---|---|
| `SENDGRID_API_KEY` | Your SendGrid API key |
| `SENDER_EMAIL` | Your verified sender email address |

> ⚠️ Never commit your `.env` file. It is already in `.gitignore`.

---

## 📌 Roadmap

- [ ] Memory — remember returning users across sessions
- [ ] Calendly integration — book demos directly in chat
- [ ] Sentiment detection — escalate tone for frustrated users
- [ ] Analytics sidebar — live lead count, intent breakdown
- [ ] WhatsApp Business API deployment

---

## 🙌 Built By

Made with ☕ and Python. Feel free to fork, star, and build on top of this!
