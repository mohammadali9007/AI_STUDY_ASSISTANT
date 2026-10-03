import re
import os
from io import BytesIO
from collections import Counter

import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PaperIQ | Research Paper Analyzer",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SKILLS / RESEARCH KEYWORDS
# =========================================================

RESEARCH_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "computer vision",
    "neural network",
    "convolutional neural network",
    "cnn",
    "transformer",
    "bert",
    "llm",
    "large language model",
    "generative ai",
    "classification",
    "regression",
    "clustering",
    "sentiment analysis",
    "text classification",
    "image classification",
    "object detection",
    "semantic analysis",
    "feature extraction",
    "tf-idf",
    "word embedding",
    "dataset",
    "accuracy",
    "precision",
    "recall",
    "f1 score",
    "f1-score",
    "experiment",
    "evaluation",
    "methodology",
    "research",
    "algorithm",
    "prediction",
    "optimization",
    "data analysis",
    "python",
    "tensorflow",
    "pytorch",
    "scikit-learn"
]


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "document_text": "",
    "file_name": "",
    "summary": "",
    "keywords": [],
    "sections": {},
    "analysis_df": None,
    "similarity": None,
    "question_answer": "",
    "document_stats": {},
    "comparison_text": ""
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text(file):

    name = file.name.lower()

    if name.endswith(".pdf"):

        reader = PdfReader(file)

        pages = []

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:

                    pages.append(text)

            except Exception:

                continue

        return "\n".join(pages)

    elif name.endswith(".docx"):

        doc = Document(file)

        return "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
        )

    elif name.endswith(".txt"):

        return file.read().decode(
            "utf-8",
            errors="ignore"
        )

    return ""


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# BASIC STATISTICS
# =========================================================

def calculate_statistics(text):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    sentences = re.split(
        r"[.!?]+",
        text
    )

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    return {
        "Words": len(words),
        "Characters": len(text),
        "Sentences": len(
            [s for s in sentences if s.strip()]
        ),
        "Paragraphs": len(paragraphs)
    }


# =========================================================
# EXTRACT RESEARCH SECTIONS
# =========================================================

def extract_sections(text):

    sections = {}

    patterns = {
        "Abstract": r"\babstract\b",
        "Introduction": r"\bintroduction\b",
        "Methodology": r"\b(methodology|methods|method)\b",
        "Results": r"\b(results|findings)\b",
        "Discussion": r"\bdiscussion\b",
        "Conclusion": r"\b(conclusion|conclusions)\b",
        "References": r"\b(references|bibliography)\b"
    }

    lines = text.splitlines()

    current_section = None

    for line in lines:

        clean = line.strip()

        if not clean:
            continue

        detected = None

        for section, pattern in patterns.items():

            if re.fullmatch(
                pattern,
                clean,
                re.I
            ):

                detected = section
                break

        if detected:

            current_section = detected

            sections[current_section] = []

            continue

        if current_section:

            sections[current_section].append(
                clean
            )

    final_sections = {}

    for section, content in sections.items():

        final_sections[section] = "\n".join(
            content[:30]
        )

    return final_sections


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text, top_n=15):

    text_lower = text.lower()

    found = []

    for keyword in RESEARCH_KEYWORDS:

        if keyword in text_lower:

            count = text_lower.count(keyword)

            found.append(
                {
                    "Keyword": keyword,
                    "Frequency": count
                }
            )

    found = sorted(
        found,
        key=lambda x: x["Frequency"],
        reverse=True
    )

    return found[:top_n]


# =========================================================
# TF-IDF TOP WORDS
# =========================================================

def tfidf_keywords(text, top_n=20):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1000
        )

        matrix = vectorizer.fit_transform(
            [text]
        )

        features = vectorizer.get_feature_names_out()

        scores = matrix.toarray()[0]

        data = []

        for word, score in zip(
            features,
            scores
        ):

            if score > 0:

                data.append(
                    {
                        "Term": word,
                        "TF-IDF Score": round(
                            float(score),
                            4
                        )
                    }
                )

        df = pd.DataFrame(data)

        if not df.empty:

            df = df.sort_values(
                "TF-IDF Score",
                ascending=False
            ).head(top_n)

        return df

    except Exception:

        return pd.DataFrame(
            columns=[
                "Term",
                "TF-IDF Score"
            ]
        )


# =========================================================
# TEXT SIMILARITY
# =========================================================

def calculate_similarity(
    text1,
    text2
):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            [
                text1,
                text2
            ]
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return similarity * 100

    except Exception:

        return 0


# =========================================================
# CREATE PDF REPORT
# =========================================================

def create_pdf_report(
    filename,
    stats,
    sections,
    keywords,
    tfidf_df,
    similarity=None
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    styles["Title"].alignment = TA_CENTER

    story = []

    story.append(
        Paragraph(
            "Research Paper Analysis Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Document:</b> {filename}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # DOCUMENT STATISTICS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Document Statistics",
            styles["Heading2"]
        )
    )

    stats_rows = [
        ["Metric", "Value"]
    ]

    for key, value in stats.items():

        stats_rows.append(
            [
                key,
                str(value)
            ]
        )

    table = Table(
        stats_rows,
        colWidths=[
            200,
            300
        ]
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                )
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # SIMILARITY
    # -----------------------------------------------------

    if similarity is not None:

        story.append(
            Paragraph(
                "Document Similarity",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                f"{similarity:.2f}%",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 12)
        )

    # -----------------------------------------------------
    # KEYWORDS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Important Keywords",
            styles["Heading2"]
        )
    )

    keyword_text = ", ".join(
        item["Keyword"]
        for item in keywords
    )

    story.append(
        Paragraph(
            keyword_text or "No keywords detected.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # -----------------------------------------------------
    # SECTIONS
    # -----------------------------------------------------

    for title, content in sections.items():

        story.append(
            Paragraph(
                title,
                styles["Heading2"]
            )
        )

        safe_content = (
            content
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        story.append(
            Paragraph(
                safe_content[:5000]
                if safe_content
                else "Not Found",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 10)
        )

    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Top TF-IDF Terms",
            styles["Heading2"]
        )
    )

    if not tfidf_df.empty:

        rows = [
            [
                "Term",
                "TF-IDF Score"
            ]
        ]

        for _, row in tfidf_df.iterrows():

            rows.append(
                [
                    row["Term"],
                    str(row["TF-IDF Score"])
                ]
            )

        table = Table(
            rows,
            colWidths=[
                350,
                150
            ]
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    )
                ]
            )
        )

        story.append(table)

    doc.build(story)

    buffer.seek(0)

    return buffer


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.08);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 16px;
        color: #777;
    }

    .kpi {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 15px;
        padding: 18px;
        text-align: center;
        background: rgba(128,128,128,0.05);
    }

    .kpi-label {
        font-size: 12px;
        opacity: .7;
        text-transform: uppercase;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
    }

    .keyword-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 PaperIQ")

    st.caption(
        "Research Paper Analyzer"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📤 Analyze Paper",
            "📝 Summary",
            "🔍 Keywords",
            "📊 Analytics",
            "❓ Ask Paper",
            "🔗 Compare Papers",
            "📋 Paper Sections",
            "ℹ️ About"
        ]
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">

        <h1>📚 PaperIQ</h1>

        <p>
        Research Paper Analysis & NLP Assistant
        </p>

        <p>
        Upload academic papers, analyze their content,
        extract keywords, study important sections,
        compare documents and understand research papers.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.document_text:

        st.info(
            "Go to **Analyze Paper** and upload "
            "a research paper to get started."
        )

        st.markdown(
            """
            ### ✨ Features

            | Feature | Description |
            |---|---|
            | 📄 Paper Parsing | Extract text from PDF/DOCX/TXT |
            | 📝 Summary | Generate paper summary |
            | 🔍 Keywords | Find important research keywords |
            | 📊 TF-IDF | Analyze important terms |
            | ❓ Ask Paper | Ask questions from paper |
            | 🔗 Compare | Compare two documents |
            | 📋 Sections | Detect research sections |
            | 📥 PDF Report | Download analysis report |
            """
        )

    else:

        stats = st.session_state.document_stats

        cols = st.columns(4)

        metrics = [
            (
                "Words",
                f"{stats.get('Words', 0):,}"
            ),
            (
                "Sentences",
                stats.get(
                    "Sentences",
                    0
                )
            ),
            (
                "Keywords",
                len(
                    st.session_state.keywords
                )
            ),
            (
                "Sections",
                len(
                    st.session_state.sections
                )
            )
        ]

        for col, (label, value) in zip(
            cols,
            metrics
        ):

            col.markdown(
                f"""
                <div class="kpi">

                <div class="kpi-label">
                {label}
                </div>

                <div class="kpi-value">
                {value}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ANALYZE PAPER
# =========================================================

elif page == "📤 Analyze Paper":

    st.title(
        "📤 Analyze Research Paper"
    )

    uploaded_file = st.file_uploader(
        "Upload Research Paper",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    if st.button(
        "🚀 Analyze Paper",
        type="primary",
        use_container_width=True
    ):

        if not uploaded_file:

            st.error(
                "Please upload a research paper."
            )

            st.stop()

        with st.spinner(
            "Reading and analyzing document..."
        ):

            raw_text = extract_text(
                uploaded_file
            )

            text = clean_text(
                raw_text
            )

            if not text:

                st.error(
                    "Could not extract readable text."
                )

                st.stop()

            sections = extract_sections(
                raw_text
            )

            keywords = extract_keywords(
                text
            )

            stats = calculate_statistics(
                text
            )

            tfidf_df = tfidf_keywords(
                text
            )

            st.session_state.document_text = text

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.sections = sections

            st.session_state.keywords = keywords

            st.session_state.document_stats = stats

            st.session_state.analysis_df = (
                tfidf_df
            )

            st.session_state.summary = ""

            st.session_state.question_answer = ""

        st.success(
            "Research paper analyzed successfully!"
        )

        st.write("")

        cols = st.columns(4)

        for col, (key, value) in zip(
            cols,
            stats.items()
        ):

            col.metric(
                key,
                f"{value:,}"
                if isinstance(value, int)
                else value
            )


# =========================================================
# SUMMARY
# =========================================================

elif page == "📝 Summary":

    st.title(
        "📝 Research Paper Summary"
    )

    if not st.session_state.document_text:

        st.info(
            "Analyze a paper first."
        )

    else:

        st.write(
            f"📄 {st.session_state.file_name}"
        )

        if st.button(
            "✨ Generate Summary",
            type="primary"
        ):

            # Simple extractive summary
            text = st.session_state.document_text

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text
            )

            sentences = [
                s.strip()
                for s in sentences
                if len(s.strip()) > 40
            ]

            if sentences:

                vectorizer = TfidfVectorizer(
                    stop_words="english"
                )

                matrix = vectorizer.fit_transform(
                    sentences
                )

                scores = matrix.sum(
                    axis=1
                ).A1

                count = min(
                    8,
                    len(sentences)
                )

                indices = scores.argsort()[
                    -count:
                ][::-1]

                selected = [
                    sentences[i]
                    for i in sorted(indices)
                ]

                summary = "\n\n".join(
                    selected
                )

                st.session_state.summary = (
                    summary
                )

            else:

                st.session_state.summary = (
                    "Not enough text available "
                    "for summary."
                )

        if st.session_state.summary:

            st.markdown(
                "### 📖 Summary"
            )

            st.write(
                st.session_state.summary
            )


# =========================================================
# KEYWORDS
# =========================================================

elif page == "🔍 Keywords":

    st.title(
        "🔍 Important Research Keywords"
    )

    if not st.session_state.document_text:

        st.info(
            "Analyze a paper first."
        )

    else:

        keywords = (
            st.session_state.keywords
        )

        if keywords:

            df = pd.DataFrame(
                keywords
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "📊 Keyword Frequency"
            )

            st.bar_chart(
                df.set_index(
                    "Keyword"
                )["Frequency"]
            )

        else:

            st.warning(
                "No predefined research keywords found."
            )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.title(
        "📊 Paper Analytics"
    )

    if not st.session_state.document_text:

        st.info(
            "Analyze a paper first."
        )

    else:

        stats = (
            st.session_state.document_stats
        )

        st.subheader(
            "📈 Document Statistics"
        )

        df_stats = pd.DataFrame(
            {
                "Metric": list(
                    stats.keys()
                ),
                "Value": list(
                    stats.values()
                )
            }
        )

        st.dataframe(
            df_stats,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🔤 Top TF-IDF Terms"
        )

        tfidf_df = (
            st.session_state.analysis_df
        )

        if tfidf_df is not None:

            st.dataframe(
                tfidf_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                tfidf_df.set_index(
                    "Term"
                )["TF-IDF Score"]
            )


# =========================================================
# ASK PAPER
# =========================================================

elif page == "❓ Ask Paper":

    st.title(
        "❓ Ask Questions About Paper"
    )

    if not st.session_state.document_text:

        st.info(
            "Analyze a paper first."
        )

    else:

        question = st.text_area(
            "Your Question",
            placeholder=(
                "Example: What is the main objective "
                "of this research?"
            ),
            height=130
        )

        if st.button(
            "🔎 Find Answer",
            type="primary"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                text = (
                    st.session_state.document_text
                )

                paragraphs = re.split(
                    r"(?<=[.!?])\s+",
                    text
                )

                paragraphs = [
                    p.strip()
                    for p in paragraphs
                    if p.strip()
                ]

                if paragraphs:

                    try:

                        vectorizer = TfidfVectorizer(
                            stop_words="english",
                            ngram_range=(1, 2)
                        )

                        matrix = vectorizer.fit_transform(
                            paragraphs
                        )

                        query_vector = (
                            vectorizer.transform(
                                [question]
                            )
                        )

                        scores = cosine_similarity(
                            query_vector,
                            matrix
                        )[0]

                        top_indices = scores.argsort()[
                            -5:
                        ][::-1]

                        answers = []

                        for index in top_indices:

                            if scores[index] > 0:

                                answers.append(
                                    paragraphs[index]
                                )

                        if answers:

                            st.session_state.question_answer = (
                                "\n\n".join(
                                    answers[:3]
                                )
                            )

                        else:

                            st.session_state.question_answer = (
                                "No relevant information "
                                "was found in the paper."
                            )

                    except Exception:

                        st.session_state.question_answer = (
                            "Could not analyze the question."
                        )

        if st.session_state.question_answer:

            st.subheader(
                "📖 Relevant Information"
            )

            st.write(
                st.session_state.question_answer
            )


# =========================================================
# COMPARE PAPERS
# =========================================================

elif page == "🔗 Compare Papers":

    st.title(
        "🔗 Compare Research Papers"
    )

    if not st.session_state.document_text:

        st.info(
            "Analyze your first paper before comparison."
        )

    else:

        st.write(
            f"Current Paper: "
            f"**{st.session_state.file_name}**"
        )

        second_file = st.file_uploader(
            "Upload second paper",
            type=[
                "pdf",
                "docx",
                "txt"
            ],
            key="second_paper"
        )

        if st.button(
            "🔗 Compare Documents",
            type="primary"
        ):

            if not second_file:

                st.error(
                    "Please upload a second paper."
                )

            else:

                second_text = clean_text(
                    extract_text(
                        second_file
                    )
                )

                if not second_text:

                    st.error(
                        "Could not read second paper."
                    )

                else:

                    similarity = calculate_similarity(
                        st.session_state.document_text,
                        second_text
                    )

                    st.session_state.similarity = (
                        similarity
                    )

                    st.session_state.comparison_text = (
                        second_text
                    )

        if st.session_state.similarity is not None:

            similarity = (
                st.session_state.similarity
            )

            st.metric(
                "Text Similarity",
                f"{similarity:.2f}%"
            )

            st.progress(
                min(
                    similarity / 100,
                    1.0
                )
            )

            if similarity >= 70:

                st.warning(
                    "The two documents contain "
                    "a high level of textual similarity."
                )

            elif similarity >= 40:

                st.info(
                    "The documents have moderate "
                    "textual similarity."
                )

            else:

                st.success(
                    "The documents have relatively "
                    "low textual similarity."
                )


# =========================================================
# PAPER SECTIONS
# =========================================================

elif page == "📋 Paper Sections":

    st.title(
        "📋 Research Paper Sections"
    )

    if not st.session_state.document_text:

        st.info(
            "Analyze a paper first."
        )

    else:

        sections = (
            st.session_state.sections
        )

        if not sections:

            st.warning(
                "Standard research sections "
                "could not be detected."
            )

        else:

            for section, content in sections.items():

                with st.expander(
                    f"📌 {section}",
                    expanded=False
                ):

                    if content:

                        st.write(
                            content
                        )

                    else:

                        st.caption(
                            "No content found."
                        )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About PaperIQ"
    )

    st.markdown(
        """
        **PaperIQ** is an NLP-based Research Paper
        Analysis application.

        It helps students and researchers understand
        academic papers using traditional NLP and
        machine learning techniques.

        ### 🚀 Features

        - PDF / DOCX / TXT processing
        - Research section detection
        - Important keyword extraction
        - TF-IDF analysis
        - Text similarity
        - Extractive summarization
        - Question-based document search
        - Paper comparison
        - PDF report generation

        ### 🛠️ Technology

        **Python**

        **Streamlit**

        **Pandas**

        **Scikit-learn**

        **PyPDF**

        **python-docx**

        **ReportLab**

        ### 🧠 NLP Techniques

        - Text preprocessing
        - TF-IDF
        - Cosine Similarity
        - Keyword extraction
        - Extractive summarization
        - Information retrieval

        ### ⚠️ Note

        The application provides automated text analysis
        and should be used as a research/study assistant,
        not as a replacement for human academic judgment.
        """
    )


# =========================================================
# DOWNLOAD REPORT
# =========================================================

if (
    st.session_state.document_text
    and page in [
        "🏠 Dashboard",
        "📊 Analytics",
        "📋 Paper Sections"
    ]
):

    st.divider()

    st.subheader(
        "📥 Download Analysis Report"
    )

    pdf_report = create_pdf_report(
        st.session_state.file_name,
        st.session_state.document_stats,
        st.session_state.sections,
        st.session_state.keywords,
        st.session_state.analysis_df
        if st.session_state.analysis_df is not None
        else pd.DataFrame(),
        st.session_state.similarity
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        os.path.splitext(
            st.session_state.file_name
        )[0]
    )

    st.download_button(
        "📄 Download PDF Report",
        pdf_report,
        file_name=f"{safe_name}_analysis_report.pdf",
        mime="application/pdf"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        padding:25px;
        color:#777;
    ">

    📚 PaperIQ  
    <br>
    Research Paper Analyzer • NLP + Machine Learning

    </div>
    """,
    unsafe_allow_html=True
)
