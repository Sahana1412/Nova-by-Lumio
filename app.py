import os
import csv
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# SAVE LEAD (UPDATED WITH TIMESTAMP)
# ==========================================

def save_lead(name, email, platform):
    file_exists = os.path.isfile("leads.csv")

    with open("leads.csv", "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Email", "Platform", "Timestamp"])

        writer.writerow([name, email, platform, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# ==========================================
# LOAD LOCAL LLM (OLLAMA)
# ==========================================

llm = ChatOllama(
    model="phi3",
    temperature=0.3
)

# ==========================================
# LOAD KNOWLEDGE BASE
# ==========================================

loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# ==========================================
# SPLIT TEXT
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# ==========================================
# EMBEDDINGS
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# VECTOR DB
# ==========================================

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ==========================================
# MEMORY
# ==========================================

lead_data = {"name": None, "email": None, "platform": None}
high_intent = False

# ==========================================
# DISPLAY FUNCTION
# ==========================================

def show_lead(name, email, platform):
    print("\n==============================")
    print("✅ LEAD CAPTURED")
    print("==============================")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Platform: {platform}")
    print("==============================\n")

# ==========================================
# INTENT DETECTION
# ==========================================

def detect_intent(user_input):
    text = user_input.lower()

    if any(word in text for word in ["buy", "purchase", "subscribe", "sign up"]):
        return "high_intent"

    if any(word in text for word in ["price", "cost", "plan", "pricing"]):
        return "pricing"

    if any(word in text for word in ["hi", "hello", "hey"]):
        return "greeting"

    # fallback LLM
    prompt = f"""
You are an intent classifier.

ONLY return ONE word:
greeting
pricing
high_intent
unknown

User: {user_input}
"""

    response = llm.invoke(prompt)
    intent = response.content.strip().lower().split()[0]

    return intent

# ==========================================
# RAG RESPONSE
# ==========================================

def get_rag_response(user_input):
    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are AutoStream assistant.

Answer ONLY from this context:

{context}

Question:
{user_input}

If not found, say:
"I could not find that information."
"""

    response = llm.invoke(prompt)
    return response.content

# ==========================================
# CHAT LOOP
# ==========================================

print("\n==============================")
print("🤖 AutoStream AI Agent Started")
print("Type 'exit' to stop")
print("==============================\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nBot: Goodbye!\n")
        break

    # ===== LEAD FLOW =====
    if high_intent:

        if not lead_data["name"]:
            lead_data["name"] = user_input
            print("Bot: Enter your email.")
            continue

        elif not lead_data["email"]:
            lead_data["email"] = user_input
            print("Bot: Which platform do you use?")
            continue

        elif not lead_data["platform"]:
            lead_data["platform"] = user_input

            # ✅ SHOW LEAD
            show_lead(
                lead_data["name"],
                lead_data["email"],
                lead_data["platform"]
            )

            # ✅ SAVE LEAD (THIS WAS MISSING)
            save_lead(
                lead_data["name"],
                lead_data["email"],
                lead_data["platform"]
            )

            print("Bot: Thanks! We'll contact you.\n")

            # reset
            high_intent = False
            lead_data = {"name": None, "email": None, "platform": None}
            continue

    # ===== INTENT =====
    intent = detect_intent(user_input)
    print(f"[Intent: {intent}]")

    if intent == "greeting":
        print("Bot: Hello! How can I help you?\n")

    elif intent == "pricing":
        print(f"Bot: {get_rag_response(user_input)}\n")

    elif intent == "high_intent":
        high_intent = True
        print(f"Bot: {get_rag_response(user_input)}")
        print("Bot: Let's get started. What's your name?\n")

    else:
        print(f"Bot: {get_rag_response(user_input)}\n")