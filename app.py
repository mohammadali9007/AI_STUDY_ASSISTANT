import os
import re
import json
import random

import streamlit as st
import numpy as np
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

h1 {
    font-size: 2.5rem !important;
}

.subtitle {
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 25px;
}

.stat-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06);
    text-align: center;
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
}

.stat-label {
    color: #6b7280;
    font-size: 0.9rem;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.05);
}

.topic-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 10px;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# GEMINI API
# =========================================================

def get_api_key():

    try:
        return st.secrets["GEMINI_API_KEY"]

    except Exception:
        return os.getenv("GEMINI_API_KEY")


api_key = get_api_key()


if not api_key:

    st.error(
        "⚠️ Gemini API key not found.\n\n"
        "Please add GEMINI_API_KEY inside "
        ".streamlit/secrets.toml"
    )

    st.stop()


client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"


# =========================================================
# SESSION STATE
# =========================================================

defaults = {

    "document_text": "",

    "chunks": [],

    "vectorizer": None,

    "matrix": None,

    "file_name": "",

    "summary": "",

    "topics": [],

    "mcqs": [],

    "quiz": [],

    "quiz_index": 0,

    "quiz_score": 0,

    "quiz_finished": False,

    "quiz_started": False,

    "last_answer": None,

    "last_quiz_answer": None,

    "quiz_submitted": False,

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:

                    pages.append(text)

            except Exception:

                continue

        final_text = "\n".join(pages)

        return clean_text(final_text)

    except Exception as e:

        return ""


# =========================================================
# CREATE TEXT CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=300,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():

            chunks.append(chunk.strip())

        start = end - overlap

        if start < 0:

            start = 0

    return chunks


# =========================================================
# BUILD TF-IDF SEARCH INDEX
# =========================================================

def build_search_index(chunks):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    matrix = vectorizer.fit_transform(chunks)

    return vectorizer, matrix


# =========================================================
# RETRIEVE RELEVANT CONTEXT
# =========================================================

def retrieve_context(question, top_k=6):

    if (

        st.session_state.vectorizer is None

        or st.session_state.matrix is None

        or not st.session_state.chunks

    ):

        return ""


    try:

        query_vector = (
            st.session_state.vectorizer
            .transform([question])
        )

        scores = cosine_similarity(
            query_vector,
            st.session_state.matrix
        )[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        selected = []

        for index in top_indices:

            if scores[index] > 0:

                selected.append(
                    st.session_state.chunks[index]
                )

        return "\n\n---\n\n".join(selected)

    except Exception:

        return ""


# =========================================================
# GEMINI CALL
# =========================================================

def call_gemini(prompt):

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=prompt

        )

        if response and response.text:

            return response.text.strip()

        return "No answer was generated."

    except Exception as e:

        return f"ERROR: {str(e)}"


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(uploaded_file):

    text = extract_pdf_text(uploaded_file)

    if not text:

        return False, "Could not extract text from this PDF."


    chunks = create_chunks(text)


    if not chunks:

        return False, "No usable text found in this PDF."


    try:

        vectorizer, matrix = (
            build_search_index(chunks)
        )

    except Exception as e:

        return False, f"Could not build search index: {e}"


    st.session_state.document_text = text

    st.session_state.chunks = chunks

    st.session_state.vectorizer = vectorizer

    st.session_state.matrix = matrix

    st.session_state.file_name = uploaded_file.name


    # Reset old results

    st.session_state.summary = ""

    st.session_state.topics = []

    st.session_state.mcqs = []

    st.session_state.quiz = []

    st.session_state.quiz_index = 0

    st.session_state.quiz_score = 0

    st.session_state.quiz_finished = False

    st.session_state.quiz_started = False

    st.session_state.last_answer = None

    st.session_state.last_quiz_answer = None

    st.session_state.quiz_submitted = False


    return True, "PDF processed successfully."


# =========================================================
# ANSWER STUDENT QUESTION
# =========================================================

def answer_question(question):

    context = retrieve_context(
        question,
        top_k=6
    )


    if not context:

        return (
            "I couldn't find relevant information "
            "in the uploaded document."
        )


    prompt = f"""

You are an AI Study Assistant.

You must answer the student's question using
ONLY the information available in the document context.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say clearly:

   "I couldn't find this information in the uploaded document."

4. Explain in simple student-friendly language.
5. If useful, use bullet points.
6. Give a direct answer first.
7. Keep the answer reasonably concise.

DOCUMENT CONTEXT:

{context}


STUDENT QUESTION:

{question}


ANSWER:

"""


    result = call_gemini(prompt)


    if result.startswith("ERROR:"):

        return result


    return result


# =========================================================
# GENERATE SUMMARY
# =========================================================

def generate_summary():

    document = st.session_state.document_text

    # Limit very large documents

    document = document[:50000]


    prompt = f"""

You are an AI Study Assistant.

Create a useful study summary from the document.

Requirements:

1. Give a short overview.
2. Explain the major concepts.
3. Use headings.
4. Use bullet points.
5. Include important definitions.
6. Include important examples when available.
7. Keep language simple.
8. Make it useful for exam preparation.
9. Do not add information that is not in the document.

DOCUMENT:

{document}


SUMMARY:

"""


    return call_gemini(prompt)


# =========================================================
# GENERATE IMPORTANT TOPICS
# =========================================================

def generate_topics():

    document = st.session_state.document_text[:50000]


    prompt = f"""

Analyze the following study document.

Find the most important topics that
a student should study for an exam.

Return ONLY valid JSON.

Format:

[
    {{
        "topic": "Topic name",
        "reason": "Why this topic is important"
    }}
]

Give around 8 to 12 topics.

DOCUMENT:

{document}

"""


    result = call_gemini(prompt)


    if result.startswith("ERROR:"):

        return []


    try:

        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        ).strip()


        data = json.loads(result)


        if isinstance(data, list):

            return data


    except Exception:

        pass


    return []


# =========================================================
# GENERATE MCQ
# =========================================================

def generate_mcqs(number=10):

    document = st.session_state.document_text[:50000]


    prompt = f"""

You are an AI educational question generator.

Create {number} multiple-choice questions
from the provided document.

Questions must be based ONLY on the document.

Return ONLY valid JSON.

Format:

[
    {{
        "question": "Question text",

        "options": [
            "Option A",
            "Option B",
            "Option C",
            "Option D"
        ],

        "answer": "Option A",

        "explanation": "Short explanation"
    }}
]

Rules:

- Exactly 4 options.
- Only one correct answer.
- Questions must come from the document.
- Do not use outside information.
- Make questions useful for exam preparation.
- Mix easy, medium and difficult questions.

DOCUMENT:

{document}

"""


    result = call_gemini(prompt)


    if result.startswith("ERROR:"):

        return []


    try:

        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        ).strip()


        data = json.loads(result)


        if isinstance(data, list):

            return data


    except Exception:

        pass


    return []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 AI Study Assistant")

    st.caption(
        "Upload your study material and "
        "learn with AI."
    )


    st.divider()


    uploaded_file = st.file_uploader(

        "Upload PDF",

        type=["pdf"],

        help="Upload your textbook, notes or study material."

    )


    if uploaded_file:

        if (
            st.session_state.file_name
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                success, message = (
                    process_pdf(uploaded_file)
                )


            if success:

                st.success(message)

            else:

                st.error(message)


    st.divider()


    if st.session_state.document_text:

        word_count = len(
            st.session_state.document_text.split()
        )

        chunk_count = len(
            st.session_state.chunks
        )


        st.subheader("📊 Document Info")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Words",
                f"{word_count:,}"
            )


        with col2:

            st.metric(
                "Chunks",
                chunk_count
            )


        st.caption(
            f"📄 {st.session_state.file_name}"
        )


    st.divider()


    st.subheader("💡 How to use")

    st.markdown(
        """
        1. Upload a PDF
        2. Ask questions
        3. Generate summary
        4. Find important topics
        5. Generate MCQs
        6. Take a quiz
        """
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("📚 AI Study Assistant")

st.markdown(
    """
    <div class="subtitle">
    Your personal AI-powered study partner for
    questions, summaries, MCQs, important topics
    and interactive quizzes.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DOCUMENT NOT UPLOADED
# =========================================================

if not st.session_state.document_text:

    st.info(
        "👈 Please upload a PDF from the sidebar "
        "to start studying."
    )

    st.markdown(
        """
        ### ✨ Features

        | Feature | Description |
        |---|---|
        | 💬 Ask Questions | Ask questions from your PDF |
        | 📝 Summary | Generate a study summary |
        | 🎯 Important Topics | Find important exam topics |
        | ❓ MCQ Generator | Generate MCQs from your PDF |
        | 🧠 Quiz | Take an interactive quiz |
        """
    )

    st.stop()


# =========================================================
# STAT CARDS
# =========================================================

word_count = len(
    st.session_state.document_text.split()
)

chunk_count = len(
    st.session_state.chunks
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-number">
                📄
            </div>
            <div class="stat-label">
                Document
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-number">
                {word_count:,}
            </div>
            <div class="stat-label">
                Words
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-number">
                {chunk_count}
            </div>
            <div class="stat-label">
                Search Chunks
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="stat-box">
            <div class="stat-number">
                🤖
            </div>
            <div class="stat-label">
                Gemini AI
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💬 Ask Questions",
        "📝 Summary",
        "❓ MCQ Generator",
        "🎯 Important Topics",
        "🧠 Quiz"
    ]
)


# =========================================================
# TAB 1 — ASK QUESTIONS
# =========================================================

with tab1:

    st.subheader("💬 Ask Questions")

    st.write(
        "Ask anything related to the uploaded PDF."
    )


    question = st.text_area(

        "Your Question",

        placeholder=(
            "Example: What is Natural Language Processing?"
        ),

        height=120

    )


    if st.button(
        "🤖 Ask AI",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Thinking..."
            ):

                answer = answer_question(
                    question
                )


            st.session_state.last_answer = answer


    if st.session_state.last_answer:

        st.markdown(
            "### 🤖 Answer"
        )


        st.markdown(
            f"""
            <div class="answer-box">
            {st.session_state.last_answer}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# TAB 2 — SUMMARY
# =========================================================

with tab2:

    st.subheader("📝 Document Summary")

    st.write(
        "Generate a simple study-friendly summary."
    )


    if st.button(
        "📝 Generate Summary",
        type="primary"
    ):

        with st.spinner(
            "Generating summary..."
        ):

            summary = generate_summary()


        st.session_state.summary = summary


    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


# =========================================================
# TAB 3 — MCQ GENERATOR
# =========================================================

with tab3:

    st.subheader("❓ MCQ Generator")

    st.write(
        "Generate multiple-choice questions "
        "from your document."
    )


    number_of_mcqs = st.slider(

        "Number of Questions",

        min_value=5,

        max_value=20,

        value=10

    )


    if st.button(
        "✨ Generate MCQs",
        type="primary"
    ):

        with st.spinner(
            "Generating MCQs..."
        ):

            mcqs = generate_mcqs(
                number_of_mcqs
            )


        st.session_state.mcqs = mcqs


    if st.session_state.mcqs:

        st.success(
            f"{len(st.session_state.mcqs)} "
            "questions generated."
        )


        for i, mcq in enumerate(
            st.session_state.mcqs,
            start=1
        ):

            st.markdown(
                f"### Q{i}. {mcq.get('question', '')}"
            )


            options = mcq.get(
                "options",
                []
            )


            for option in options:

                st.write(
                    f"○ {option}"
                )


            with st.expander(
                "View Answer & Explanation"
            ):

                st.success(
                    f"Correct Answer: "
                    f"{mcq.get('answer', '')}"
                )


                st.info(
                    mcq.get(
                        "explanation",
                        ""
                    )
                )


            st.divider()


# =========================================================
# TAB 4 — IMPORTANT TOPICS
# =========================================================

with tab4:

    st.subheader(
        "🎯 Important Topics"
    )

    st.write(
        "Find the topics that are most "
        "important for study and exam preparation."
    )


    if st.button(
        "🎯 Find Important Topics",
        type="primary"
    ):

        with st.spinner(
            "Analyzing document..."
        ):

            topics = generate_topics()


        st.session_state.topics = topics


    if st.session_state.topics:

        for i, item in enumerate(
            st.session_state.topics,
            start=1
        ):

            topic = item.get(
                "topic",
                "Unknown Topic"
            )

            reason = item.get(
                "reason",
                ""
            )


            st.markdown(
                f"""
                <div class="topic-box">

                <strong>
                {i}. {topic}
                </strong>

                <br><br>

                {reason}

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# TAB 5 — QUIZ
# =========================================================

with tab5:

    st.subheader("🧠 Interactive Quiz")

    st.write(
        "Test your knowledge using questions "
        "generated from the uploaded document."
    )


    # -----------------------------------------------------
    # START QUIZ
    # -----------------------------------------------------

    if not st.session_state.quiz_started:

        if st.button(
            "🚀 Start Quiz",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Preparing your quiz..."
            ):

                quiz = generate_mcqs(10)


            if quiz:

                random.shuffle(quiz)


                st.session_state.quiz = quiz

                st.session_state.quiz_index = 0

                st.session_state.quiz_score = 0

                st.session_state.quiz_finished = False

                st.session_state.quiz_started = True

                st.session_state.last_quiz_answer = None

                st.session_state.quiz_submitted = False

                st.rerun()

            else:

                st.error(
                    "Could not generate quiz."
                )


    # -----------------------------------------------------
    # QUIZ ACTIVE
    # -----------------------------------------------------

    if (
        st.session_state.quiz_started
        and not st.session_state.quiz_finished
    ):

        quiz = st.session_state.quiz

        current_index = (
            st.session_state.quiz_index
        )


        if current_index < len(quiz):

            current_question = quiz[
                current_index
            ]


            total_questions = len(quiz)


            st.progress(
                current_index / total_questions
            )


            st.caption(
                f"Question "
                f"{current_index + 1} "
                f"of {total_questions}"
            )


            st.markdown(
                f"""
                ## Q{current_index + 1}. 
                {current_question.get('question', '')}
                """
            )


            options = current_question.get(
                "options",
                []
            )


            selected_option = st.radio(

                "Select your answer:",

                options,

                key=f"quiz_option_{current_index}"

            )


            # ------------------------------------------------
            # SUBMIT ANSWER
            # ------------------------------------------------

            if not st.session_state.quiz_submitted:

                if st.button(
                    "✅ Submit Answer",
                    type="primary"
                ):

                    correct_answer = (
                        current_question.get(
                            "answer",
                            ""
                        )
                    )


                    if selected_option == correct_answer:

                        st.session_state.quiz_score += 1

                        st.session_state.last_quiz_answer = (
                            "correct"
                        )

                    else:

                        st.session_state.last_quiz_answer = (
                            "wrong"
                        )


                    st.session_state.quiz_submitted = True

                    st.rerun()


            # ------------------------------------------------
            # SHOW FEEDBACK
            # ------------------------------------------------

            if st.session_state.quiz_submitted:

                correct_answer = (
                    current_question.get(
                        "answer",
                        ""
                    )
                )


                if (
                    st.session_state.last_quiz_answer
                    == "correct"
                ):

                    st.success(
                        "🎉 Correct Answer!"
                    )

                else:

                    st.error(
                        "❌ Incorrect Answer"
                    )


                    st.info(
                        f"Correct answer: "
                        f"{correct_answer}"
                    )


                explanation = (
                    current_question.get(
                        "explanation",
                        ""
                    )
                )


                if explanation:

                    st.write(
                        f"💡 {explanation}"
                    )


                st.write("")


                if current_index + 1 < total_questions:

                    if st.button(
                        "➡️ Next Question",
                        use_container_width=True
                    ):

                        st.session_state.quiz_index += 1

                        st.session_state.quiz_submitted = False

                        st.session_state.last_quiz_answer = None

                        st.rerun()

                else:

                    if st.button(
                        "🏁 Finish Quiz",
                        use_container_width=True
                    ):

                        st.session_state.quiz_finished = True

                        st.rerun()


    # -----------------------------------------------------
    # QUIZ RESULT
    # -----------------------------------------------------

    if (
        st.session_state.quiz_started
        and st.session_state.quiz_finished
    ):

        total = len(
            st.session_state.quiz
        )


        score = st.session_state.quiz_score


        if total > 0:

            accuracy = (
                score / total
            ) * 100

        else:

            accuracy = 0


        st.success(
            "🎉 Quiz Completed!"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Score",
                f"{score}/{total}"
            )


        with col2:

            st.metric(
                "Accuracy",
                f"{accuracy:.1f}%"
            )


        with col3:

            st.metric(
                "Questions",
                total
            )


        st.divider()


        st.subheader(
            "📖 Quiz Review"
        )


        for i, question in enumerate(
            st.session_state.quiz,
            start=1
        ):

            st.markdown(
                f"**Q{i}. {question.get('question', '')}**"
            )


            st.write(
                f"✅ Correct Answer: "
                f"{question.get('answer', '')}"
            )


        st.write("")


        if st.button(
            "🔄 Take Another Quiz",
            type="primary"
        ):

            st.session_state.quiz = []

            st.session_state.quiz_index = 0

            st.session_state.quiz_score = 0

            st.session_state.quiz_finished = False

            st.session_state.quiz_started = False

            st.session_state.last_quiz_answer = None

            st.session_state.quiz_submitted = False

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    📚 AI Study Assistant  
    <br>
    Powered by Python • Streamlit • Gemini AI

    </div>
    """,
    unsafe_allow_html=True
)
