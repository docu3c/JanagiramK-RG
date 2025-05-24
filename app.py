import time
import logging
import streamlit as st
from openai import AzureOpenAI
from PyPDF2 import PdfReader
import docx
import re

# Configure Streamlit page
st.set_page_config(page_title="Legal Writing Assistant", layout="centered")
logging.basicConfig(level=logging.INFO)

# Azure OpenAI credentials
api_key = "1aiqkqAdhcHzNk1NUAOV34Y838BKv6LFeYfLGxkHqtOji2lvn6JEJQQJ99BAACfhMk5XJ3w3AAABACOGQKhC"
endpoint = "https://gptinswedencentral.openai.azure.com/"

# Sidebar debug information
st.sidebar.title("🔧 Debug Info")
st.sidebar.write("✅ API Key Loaded:", bool(api_key))
st.sidebar.write("✅ Endpoint Loaded:", bool(endpoint))

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-02-15-preview",
)

# Legal Writing Guidelines
LEGAL_WRITING_GUIDELINES = """
**Legal Writing Guidelines**

Apply the following principles to improve or evaluate legal writing:

1. **Be Concise**: Express arguments succinctly without losing meaning.
   - *Not concise*: "The argument made by opposing counsel is one that fails to succeed for reasons including, inter alia, the fact that the legislature clearly did not evince an intent to restrict the business activities of the defendant."
   - *More concise*: "Opposing counsel’s argument fails because the legislature did not intend to limit the defendant’s business activities."

2. **Use Active Voice**: Clearly identify the action and the actor.
   - *Passive*: "The penalty was called by the referee."
   - *Active*: "The referee called the penalty."

3. **Simplify Legalese**: Replace complex legal terms with simpler language unless necessary.
   - *Examples*: "inter alia" → "among other things"; "utilize" → "use"

4. **Limit Nominalizations**: Avoid converting verbs into nouns.
   - *Nominalization*: "There was committee agreement."
   - *Fix*: "The committee agreed."

5. **Omit Unnecessary Words and Phrases**: Use simpler words instead of compound constructions.
   - *Examples*: "at the point in time" → "then"; "by means of" → "by"

6. **Avoid Run-On Sentences**: Focus on one main point per sentence; aim for about 20 words per sentence.

7. **Break Apart Overly Long Paragraphs**: Stick to one idea per paragraph.

8. **Avoid Redundancy**: Use a single word instead of listing synonyms.
   - *Redundant*: "Cease and desist."
   - *Concise*: "Stop."

9. **Avoid Meaningless Adverbs and Weasel Words**: Do not use adverbs or words that weaken your position.
   - *Meaningless*: "Chester v. Morris clearly held..."
   - *Concise*: "Chester v. Morris held..."

10. **Avoid Double Negatives**: Use single positives instead.
    - *Double Negative*: "Not uncommon."
    - *Positive*: "Common."

11. **Omit Phrases with No Meaning**: Remove meaningless phrases to enhance clarity.
    - *Meaningless*: "I would like to point out that..."
    - *Concise*: "Chester v. Morris was overruled."

**Additional Tips**

- **Generate Alternatives**: Present different perspectives fairly or favorably, depending on the context.
- **Marshal Relevant Information**: Gather facts, legal authority, and social authority to support your argument.
- **Examine Information Critically**: Assess the strengths and weaknesses of supporting information.
- **Reach a Conclusion**: Choose the better alternative based on analysis, even if it's not perfect.
"""

# Function to extract text from uploaded files
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        all_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)
        return "\n".join(all_text)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        return "❌ Unsupported file type."

# Cache the assistant creation to avoid redundant API calls
@st.cache_resource
def create_assistant():
    return client.beta.assistants.create(
        name="Legal Writing Assistant",
        instructions="You are a legal writing coach for law firm associates.",
        tools=[], 
        model="gpt-4o"
    )

# Function to run the assistant with given prompt and instructions
def run_assistant(prompt, task_instructions):
    try:
        assistant = create_assistant()
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions=task_instructions + "\n\n" + LEGAL_WRITING_GUIDELINES
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

# Main Streamlit application
st.title("📚 Legal Writing Assistant")
st.markdown("Upload a legal document (.docx or .pdf) and choose what you want to do:")

uploaded_file = st.file_uploader("📁 Upload DOCX or PDF", type=["pdf", "docx"])

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    if not raw_text.strip():
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
                result = run_assistant(raw_text, instructions)
                st.subheader("📄 Improved Legal Text")
                st.write(result)

        if col2.button("🔍 Evaluate Legal Writing"):
            with st.spinner("Evaluating your document..."):
                instructions = (
                    "You are a legal writing evaluator. Analyze the provided legal text and deliver a detailed, professional critique "
                    "focusing on clarity, organization, tone, grammar, and persuasiveness. "
                    "Offer specific, constructive recommendations to improve the effectiveness and impact of the writing."
                )
                result = run_assistant(raw_text, instructions)
                st.subheader("📝 Evaluation Feedback")
                st.write(result)
