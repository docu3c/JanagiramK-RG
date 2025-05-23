import time
import logging
import streamlit as st
from openai import AzureOpenAI
from PyPDF2 import PdfReader
import docx
import re
import string

def clean_and_filter_legal_text(text):
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\n\d+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[^\S\r\n]{2,}", " ", text)
    lines = text.splitlines()
    legal_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 60:
            continue
        if not line.endswith(('.', ';', ':')):
            continue
        if any(keyword in line.lower() for keyword in ["agreement", "breach", "obligated", "rights", "remedies", "liable", "hereby", "pursuant"]):
            legal_lines.append(line)
    return " ".join(legal_lines)

st.set_page_config(page_title="Legal Writing Evaluator", layout="centered")
logging.basicConfig(level=logging.INFO)

api_key = "1aiqkqAdhcHzNk1NUAOV34Y838BKv6LFeYfLGxkHqtOji2lvn6JEJQQJ99BAACfhMk5XJ3w3AAABACOGQKhC"
endpoint = "https://gptinswedencentral.openai.azure.com/"

st.sidebar.title("🔧 Debug Info")
st.sidebar.write("✅ API Key Loaded:", bool(api_key))
st.sidebar.write("✅ Endpoint Loaded:", bool(endpoint))

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-02-15-preview",
)

@st.cache_resource
def create_assistant():
    return client.beta.assistants.create(
        name="Legal Writing Assistant",
        instructions="You are a legal writing coach for law firm associates.",
        tools=[], 
        model="gpt-4o"
    )

def run_assistant(prompt, instructions):
    try:
        assistant = create_assistant()
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions=instructions
        )
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status in ["completed", "failed", "cancelled", "expired"]:
                break
            time.sleep(1)
        if run_status.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for msg in reversed(list(messages)):
                if msg.role == "assistant":
                    return msg.content[0].text.value
            return "⚠️ No assistant response found."
        else:
            return f"❌ Run failed with status: {run_status.status}"
    except Exception as e:
        logging.error("Assistant error", exc_info=True)
        return f"🚨 Error: {str(e)}"

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        return "❌ Unsupported file type."

st.title("📚 Legal Writing Assistant")
st.markdown("Upload a legal document (.docx or .pdf) and choose what you want to do:")

uploaded_file = st.file_uploader("📁 Upload DOCX or PDF", type=["pdf", "docx"])

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    text = clean_and_filter_legal_text(raw_text)
    if not text.strip():
        st.error("❌ Could not extract any text from the uploaded file.")
    else:
        col1, col2 = st.columns(2)

        if col1.button("✍️ Improve Legal Writing"):
            with st.spinner("Improving your document..."):
                instructions = (
                    "You are a legal writing assistant. Please rewrite and improve the following legal text to enhance clarity, "
                    "logical structure, formality, and overall readability while maintaining legal precision. "
                    "Use professional legal language suitable for law firm associates."
                )
                result = run_assistant(text, instructions)
                st.subheader("📄 Improved Legal Text")
                st.write(result)

        if col2.button("🔍 Evaluate Legal Writing"):
            with st.spinner("Evaluating your document..."):
                instructions = (
                    "You are a legal writing evaluator. Analyze the provided legal text and deliver a detailed, professional critique "
                    "focusing on clarity, organization, tone, grammar, and persuasiveness. "
                    "Offer specific, constructive recommendations to improve the effectiveness and impact of the writing."
                )
                result = run_assistant(text, instructions)
                st.subheader("📝 Evaluation Feedback")
                st.write(result)
