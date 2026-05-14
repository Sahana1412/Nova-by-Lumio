import streamlit as st
import os
import csv
from datetime import datetime

from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL     = os.getenv("SENDER_EMAIL")      # your verified sender

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Lumio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================
# WHATSAPP-STYLE CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600&display=swap');

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #111b21 !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e9edef;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebarNav"],
footer { display: none !important; }

/* ── Wallpaper layer ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Cpath d='M30 5 L55 20 L55 40 L30 55 L5 40 L5 20 Z' fill='none' stroke='%23ffffff08' stroke-width='1'/%3E%3C/svg%3E");
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Main container ── */
.main .block-container {
    max-width: 860px !important;
    margin: 0 auto;
    padding: 0 !important;
    position: relative;
    z-index: 1;
}

/* ── Chat header ── */
.wa-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: #202c33;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 18px;
    border-bottom: 1px solid #2a373f;
    box-shadow: 0 1px 4px #0004;
}
.wa-avatar {
    width: 40px; height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00a884, #128c7e);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.wa-header-info h3 {
    margin: 0; font-size: 15px; font-weight: 600; color: #e9edef;
}
.wa-header-info p {
    margin: 0; font-size: 12px; color: #8696a0;
}
.wa-header-icons {
    margin-left: auto;
    display: flex; gap: 22px; color: #aebac1; font-size: 20px;
}

/* ── Date pill ── */
.wa-date-pill {
    text-align: center;
    margin: 14px 0 8px;
}
.wa-date-pill span {
    background: #182229;
    border: 1px solid #2a373f;
    color: #8696a0;
    font-size: 11.5px;
    padding: 4px 14px;
    border-radius: 8px;
}

/* ── Message bubbles ── */
.wa-bubble-row {
    display: flex;
    padding: 2px 18px;
    margin-bottom: 3px;
}
.wa-bubble-row.user  { justify-content: flex-end; }
.wa-bubble-row.bot   { justify-content: flex-start; }

.wa-bubble {
    max-width: 65%;
    padding: 7px 14px 5px;
    font-size: 14.2px;
    line-height: 1.5;
    position: relative;
    word-wrap: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,0.25);
}
.wa-bubble.user {
    background: #005c4b;
    border-radius: 18px 18px 4px 18px;
    color: #e9edef;
}
.wa-bubble.bot {
    background: #202c33;
    border-radius: 18px 18px 18px 4px;
    color: #e9edef;
}

/* Tail — small curved notch at the sharp corner */
.wa-bubble.user::after {
    content: "";
    position: absolute;
    bottom: 0; right: -6px;
    width: 12px; height: 12px;
    background: #005c4b;
    clip-path: polygon(0 0, 0% 100%, 100% 100%);
    border-radius: 0 0 4px 0;
}
.wa-bubble.bot::before {
    content: "";
    position: absolute;
    bottom: 0; left: -6px;
    width: 12px; height: 12px;
    background: #202c33;
    clip-path: polygon(100% 0, 0% 100%, 100% 100%);
    border-radius: 0 0 0 4px;
}

.wa-bubble-meta {
    text-align: right;
    font-size: 11px;
    color: #8696a0;
    margin-top: 4px;
    display: flex; justify-content: flex-end; align-items: center; gap: 4px;
}
/* Double tick (read receipt) */
.wa-tick { color: #53bdeb; font-size: 13px; }

/* ── Typing indicator ── */
.wa-typing {
    display: flex; align-items: center; gap: 6px;
    padding: 4px 18px 8px;
}
.wa-typing-dots {
    background: #202c33;
    border-radius: 0 8px 8px 8px;
    padding: 10px 16px;
    display: flex; gap: 5px; align-items: center;
}
.wa-typing-dots span {
    width: 7px; height: 7px;
    background: #8696a0;
    border-radius: 50%;
    animation: wa-bounce 1.2s ease-in-out infinite;
}
.wa-typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.wa-typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes wa-bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30%            { transform: translateY(-5px); }
}

/* ── Input bar ── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 860px !important;
    max-width: 100vw !important;
    background: #202c33 !important;
    border-top: 1px solid #2a373f !important;
    padding: 8px 12px !important;
    z-index: 200 !important;
}
[data-testid="stChatInput"] textarea {
    background: #2a373f !important;
    color: #e9edef !important;
    border-radius: 24px !important;
    border: none !important;
    padding: 10px 18px !important;
    font-size: 14px !important;
    outline: none !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8696a0 !important; }
[data-testid="stChatInput"] button {
    background: #00a884 !important;
    border-radius: 50% !important;
    color: white !important;
    border: none !important;
}

/* ── Scrollable message area ── */
.wa-messages {
    padding: 12px 0 90px;
    min-height: calc(100vh - 62px);
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("""
<div class="wa-header">
  <div class="wa-avatar">🤖</div>
  <div class="wa-header-info">
    <h3>Nova <span style="font-weight:400; color:#8696a0; font-size:13px;">by Lumio</span></h3>
    <p>Lumio's AI Assistant · online</p>
  </div>
  <div class="wa-header-icons">
    <span title="Search">🔍</span>
    <span title="Menu">⋮</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================
# SAVE LEAD
# =============================
def save_lead(name, email, platform):
    file_exists = os.path.isfile("leads.csv")
    with open("leads.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Email", "Platform", "Timestamp"])
        writer.writerow([name, email, platform, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# =============================
# SEND WELCOME EMAIL
# =============================
def send_welcome_email(name, email, platform):
    if not SENDGRID_API_KEY or not SENDER_EMAIL:
        print("⚠️  SendGrid credentials missing — skipping email.")
        return

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {{ margin:0; padding:0; background:#f4f6f8; font-family:'Segoe UI',sans-serif; }}
        .wrapper {{ max-width:580px; margin:40px auto; background:#ffffff; border-radius:16px;
                    overflow:hidden; box-shadow:0 4px 24px rgba(0,0,0,0.08); }}
        .header {{ background:linear-gradient(135deg,#00a884,#128c7e); padding:36px 40px; text-align:center; }}
        .header h1 {{ margin:0; color:#fff; font-size:28px; letter-spacing:-0.5px; }}
        .header p  {{ margin:6px 0 0; color:#d4f5ec; font-size:14px; }}
        .body {{ padding:36px 40px; color:#1a1a2e; }}
        .body h2 {{ font-size:22px; margin:0 0 12px; color:#111b21; }}
        .body p  {{ font-size:15px; line-height:1.7; color:#444; margin:0 0 16px; }}
        .card {{ background:#f0faf7; border-left:4px solid #00a884; border-radius:8px;
                 padding:16px 20px; margin:20px 0; }}
        .card p {{ margin:4px 0; font-size:14px; color:#333; }}
        .card span {{ font-weight:600; color:#00a884; }}
        .btn {{ display:inline-block; margin:24px 0 8px; padding:14px 32px;
                background:#00a884; color:#fff !important; border-radius:30px;
                text-decoration:none; font-size:15px; font-weight:600;
                letter-spacing:0.3px; }}
        .footer {{ background:#f4f6f8; text-align:center; padding:20px; font-size:12px; color:#999; }}
        .footer a {{ color:#00a884; text-decoration:none; }}
      </style>
    </head>
    <body>
      <div class="wrapper">
        <div class="header">
          <h1>✨ Welcome to Lumio</h1>
          <p>Powered by Nova · Your AI Sales Assistant</p>
        </div>
        <div class="body">
          <h2>Hey {name}! 👋</h2>
          <p>Thanks for reaching out — we're thrilled to have you on board.
             Our team has received your details and will be in touch with you shortly.</p>
          <div class="card">
            <p>👤 <span>Name:</span> {name}</p>
            <p>📧 <span>Email:</span> {email}</p>
            <p>💻 <span>Platform:</span> {platform}</p>
            <p>🕐 <span>Submitted:</span> {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
          </div>
          <p>While you wait, feel free to explore what Lumio can do for your workflow.
             We're excited to show you how Nova can supercharge your sales process.</p>
          <a class="btn" href="https://lumio.ai">Explore Lumio →</a>
          <p style="font-size:13px;color:#999;">
            If you didn't submit this form, you can safely ignore this email.
          </p>
        </div>
        <div class="footer">
          © {datetime.now().year} Lumio. All rights reserved.<br>
          <a href="#">Unsubscribe</a> · <a href="#">Privacy Policy</a>
        </div>
      </div>
    </body>
    </html>
    """

    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=email,
        subject=f"👋 Welcome to Lumio, {name}!",
        html_content=html_body,
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        print(f"✅ Welcome email sent to {email}")
    except Exception as e:
        print(f"❌ Email send failed: {e}")

# =============================
# LOAD LLM
# =============================
llm = ChatOllama(model="phi3", temperature=0.3)

# =============================
# LOAD DATA (RUN ONCE)
# =============================
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = load_vectorstore()

# =============================
# INTENT
# =============================
def detect_intent(text):
    text = text.lower()
    if any(w in text for w in ["buy", "purchase", "subscribe", "sign up"]):
        return "high_intent"
    if any(w in text for w in ["price", "cost", "plan", "pricing"]):
        return "pricing"
    if any(w in text for w in ["hi", "hello", "hey"]):
        return "greeting"
    return "unknown"

# =============================
# RAG
# =============================
def get_rag_response(query):
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""Answer ONLY using this context:\n\n{context}\n\nQuestion: {query}"""
    return llm.invoke(prompt).content

# =============================
# SESSION STATE
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lead_step" not in st.session_state:
    st.session_state.lead_step = None
if "lead_data" not in st.session_state:
    st.session_state.lead_data = {"name": "", "email": "", "platform": ""}

# =============================
# RENDER MESSAGES
# =============================
def bubble(role, content):
    side = "user" if role == "user" else "bot"
    now  = datetime.now().strftime("%H:%M")
    tick = '<span class="wa-tick">✓✓</span>' if role == "user" else ""
    st.markdown(f"""
    <div class="wa-bubble-row {side}">
      <div class="wa-bubble {side}">
        {content}
        <div class="wa-bubble-meta">{now} {tick}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Date separator
today = datetime.now().strftime("%B %d, %Y")
st.markdown(f"""
<div class="wa-messages">
<div class="wa-date-pill"><span>Today, {today}</span></div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    bubble(msg["role"], msg["content"])

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# INPUT
# =============================
user_input = st.chat_input("Message Nova…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── Lead flow ──
    if st.session_state.lead_step:
        step = st.session_state.lead_step
        if step == "name":
            st.session_state.lead_data["name"] = user_input
            st.session_state.lead_step = "email"
            bot_reply = "Got it! 📧 What's your email address?"
        elif step == "email":
            st.session_state.lead_data["email"] = user_input
            st.session_state.lead_step = "platform"
            bot_reply = "Almost there! Which platform do you currently use?"
        elif step == "platform":
            st.session_state.lead_data["platform"] = user_input
            save_lead(
                st.session_state.lead_data["name"],
                st.session_state.lead_data["email"],
                st.session_state.lead_data["platform"],
            )
            send_welcome_email(
                st.session_state.lead_data["name"],
                st.session_state.lead_data["email"],
                st.session_state.lead_data["platform"],
            )
            bot_reply = "✅ All done! Our team will reach out to you very soon. 🎉"
            st.session_state.lead_step = None
            st.session_state.lead_data = {"name": "", "email": "", "platform": ""}
    else:
        intent = detect_intent(user_input)
        if intent == "greeting":
            bot_reply = "Hey there! 👋 How can I help you today?"
        elif intent == "pricing":
            bot_reply = get_rag_response(user_input)
        elif intent == "high_intent":
            st.session_state.lead_step = "name"
            bot_reply = "Awesome! 🚀 Let's get you set up. What's your name?"
        else:
            bot_reply = get_rag_response(user_input)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()
