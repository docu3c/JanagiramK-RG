import time
import logging
import streamlit as st
from openai import AzureOpenAI
from PyPDF2 import PdfReader
import docx
import difflib
import os
from dotenv import load_dotenv

load_dotenv()  

api_key = os.getenv("API_KEY")
endpoint = os.getenv("ENDPOINT")

print("API_KEY:", api_key)
print("ENDPOINT:", endpoint)

st.set_page_config(page_title="Legal Writing Assistant", layout="centered")
logging.basicConfig(level=logging.INFO)




st.sidebar.title("🔧 Debug Info")
st.sidebar.write("✅ API Key Loaded:", bool(api_key))
st.sidebar.write("✅ Endpoint Loaded:", bool(endpoint))

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-02-15-preview",
)

LEGAL_WRITING_GUIDELINES = """
LEGAL WRITING GUIDELINES

These 15 principles are the foundation of clear, persuasive legal writing. Apply them rigorously. Use silent, structured reasoning to guide each revision, but show only the final product—not your process.

1. BE CONCISE  
Ask: Does this word or phrase add legal value? Could this be shorter without loss of meaning?  
✘ "The argument made by opposing counsel is one that fails to succeed for reasons including, inter alia, the fact that..."  
✔ "Opposing counsel’s argument fails because..."

2. USE ACTIVE VOICE  
Ask: Who is acting here? Is the subject direct and forceful?  
✘ "The penalty was called by the referee."  
✔ "The referee called the penalty."

3. REPLACE LEGALESE WITH PLAIN ENGLISH  
Ask: Can this be simplified without altering legal meaning?  
✘ "inter alia" → ✔ "among other things"  
✘ "pursuant to" → ✔ "under"  
✘ "utilize" → ✔ "use"

4. LIMIT NOMINALIZATIONS  
Ask: Are verbs being buried as nouns? Revive them.  
✘ "There was committee agreement."  
✔ "The committee agreed."

5. OMIT UNNECESSARY WORDS  
Ask: Can this be said more directly?  
✘ "At the point in time" → ✔ "Then"  
✘ "By means of" → ✔ "By"  
✘ "In light of the fact that" → ✔ "Because"

6. AVOID RUN-ON SENTENCES  
Ask: Does this sentence contain more than one core idea or exceed 20–25 words? Break it.  
✘ Overloaded sentence → ✔ One idea per sentence.

7. BREAK LONG PARAGRAPHS  
Ask: Does this paragraph cover more than one concept? Could breaking it help clarity?  
✔ Each paragraph should focus on a single theme or step in the argument.

8. ELIMINATE REDUNDANCY  
Ask: Are there repeated ideas or synonyms? Cut them.  
✘ "Null and void"  
✔ "Void"  
✘ "Cease and desist"  
✔ "Stop"

9. REMOVE WEASEL WORDS  
Ask: Does this word add vagueness or false confidence?  
✘ "Clearly," "arguably," "somewhat"  
✔ Omit or replace with precise reasoning.

10. AVOID DOUBLE NEGATIVES  
Ask: Is this clearer in a positive form?  
✘ "Not uncommon" → ✔ "Common"  
✘ "Not without merit" → ✔ "Has merit"

11. STRIP EMPTY PHRASES  
Ask: Is this introductory phrase adding anything?  
✘ "I would like to point out that..."  
✔ "Chester v. Morris was overruled."

12. CONSIDER LEGALLY RELEVANT ALTERNATIVES  
Ask: Are multiple interpretations possible? Should they be addressed?  
✔ Present plausible alternatives where relevant, not to hedge but to strengthen analysis.

13. MARSHAL RELEVANT FACTS, LAW, AND POLICY  
Ask: Have you gathered and integrated all key materials?  
✔ Use facts, legal rules, and policy rationales to support—not just decorate—your argument.

14. THINK CRITICALLY ABOUT INFORMATION  
Ask: Are your sources valid, and have you weighed their strengths and weaknesses?  
✔ Don't just cite—analyze. Distinguish, analogize, or critique as needed.

15. REACH A DEFINITIVE CONCLUSION  
Ask: What’s the clearest, most supportable legal position?  
✔ End with a firm conclusion, even if it acknowledges limitations.

—

Apply these principles with precision and discipline. The reader should see only clear, confident legal writing. Your thoughtful process should be invisible—but your mastery, unmistakable.
"""


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
        return ""

@st.cache_resource
def create_assistant():
    return client.beta.assistants.create(
        name="Legal Writing Assistant",
        instructions="You are a legal writing coach for law firm associates.",
        tools=[],
        model="gpt-4o"
    )

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

def parse_score_from_evaluation(evaluation_text):
    import re
    match = re.search(r"score\s*[:\-]?\s*(\d{1,3})", evaluation_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score
    return None

def highlight_differences(original, improved):
    """
    Returns HTML highlighting the exact differences between original and improved texts.
    Add <span style="background-color:yellow;"> for added/changed text,
    and <del style="background-color:#faa;"> for removed text.
    """
    differ = difflib.Differ()
    diff = list(differ.compare(original.splitlines(), improved.splitlines()))
    highlighted_lines = []

    for line in diff:
        if line.startswith("  "):
      
            highlighted_lines.append(line[2:])
        elif line.startswith("- "):
          
            highlighted_lines.append(f'<del style="background-color:#faa;">{line[2:]}</del>')
        elif line.startswith("+ "):
       
            highlighted_lines.append(f'<span style="background-color: #fffb91;">{line[2:]}</span>')
        elif line.startswith("? "):
            pass

    return "<br>".join(highlighted_lines)


st.title("📚 Legal Writing Assistant")
st.markdown("Upload a legal document (.docx or .pdf) and the system will evaluate and improve your legal writing, highlighting exactly where changes were made.")

uploaded_file = st.file_uploader("📁 Upload DOCX or PDF", type=["pdf", "docx"])

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    if not raw_text.strip():
        st.error("❌ Could not extract any text from the uploaded file.")
    else:
        if st.button("🧮 Evaluate and Improve Automatically"):
            with st.spinner("Evaluating your document..."):
                eval_instructions = (
                 "You are a legal writing evaluator. Assess the following text using the 15 Legal Writing Guidelines as your sole standard. "
                "Assign a final score from 0 to 100 reflecting the overall quality of legal writing (100 = flawless adherence). "
                "Then provide concise, bullet-point feedback organized by:\n"
                "- Strengths\n"
                "- Weaknesses\n"
                "- Actionable Suggestions\n"
                "Do not explain your reasoning or include any internal thought process.\n"
                "Use the following format strictly:\n"
                "Score: XX\n"
                "- Strength: ...\n"
                "- Weakness: ...\n"
                "- Suggestion: ...\n"
                )

                evaluation = run_assistant(raw_text, eval_instructions)
                st.subheader("📝 Evaluation Feedback")
                st.text(evaluation)

                score = parse_score_from_evaluation(evaluation)
                if score is None:
                    st.warning("⚠️ Could not parse an evaluation score. Improvement will not proceed.")
                else:
                    st.info(f"📊 Evaluation Score: {score}/100")
                    threshold = 85
                    if score >= threshold:
                        st.success("✅ Your legal writing score is good. No improvements are required.")
                    else:
                        st.warning(f"⚠️ Score below threshold ({threshold}). Generating improvements...")
                        improve_instructions = (
                        "You are a meticulous legal writing assistant. Silently apply the 15 Legal Writing Guidelines to evaluate and improve the following text. "
                        "Use internal reasoning, but do not display your thought process.\n\n"
                        "Focus on:\n"
                        "• Concise, direct language (remove redundancies, simplify phrases, eliminate legalese)\n"
                        "• Active voice and clear verb use\n"
                        "• Shorter, well-structured sentences and logically focused paragraphs\n"
                        "• A professional, precise, and confident tone throughout\n"
                        "• Legal integrity—preserve the original legal meaning at all times\n\n"
                        "Output only:\n"
                        "1. The final, revised version of the text—polished and professional\n"
                        "2. A brief summary of the key improvements applied"
                        )

                        with st.spinner("Improving your document..."):
                            improved_text = run_assistant(raw_text, improve_instructions)

                        if "Summary of key changes" in improved_text:
                            parts = improved_text.strip().split("Summary of key changes", 1)
                            improved_main = parts[0].strip()
                            summary = "Summary of key changes" + parts[1].strip()
                        else:
                            improved_main = improved_text.strip()
                            summary = "⚠️ No summary detected."

                        st.subheader("✍️ Improved Legal Text")
                        st.write(improved_main)

                    