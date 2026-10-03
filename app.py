import re
from io import BytesIO

import streamlit as st
import pandas as pd

from pypdf import PdfReader

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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ResearchGap Finder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Header */

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #111827,
        #1e293b
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

/* Cards */

.card {
    padding: 20px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 15px;
}

.card-title {
    font-size: 15px;
    color: #64748b;
    margin-bottom: 5px;
}

.card-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}

/* Section */

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Gap */

.gap-box {
    padding: 18px;
    border-left: 5px solid #ef4444;
    background: #fff7f7;
    border-radius: 10px;
    margin-bottom: 10px;
}

.future-box {
    padding: 18px;
    border-left: 5px solid #3b82f6;
    background: #f5f9ff;
    border-radius: 10px;
    margin-bottom: 10px;
}

/* Keyword */

.keyword {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    background: #eef2ff;
    color: #3730a3;
    font-size: 14px;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""

if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# =========================================================
# RESEARCH KEYWORDS
# =========================================================

RESEARCH_KEYWORDS = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "computer vision",
    "neural network",
    "convolutional neural network",
    "cnn",
    "transformer",
    "bert",
    "llm",
    "large language model",
    "classification",
    "regression",
    "clustering",
    "sentiment analysis",
    "image classification",
    "object detection",
    "feature extraction",
    "transfer learning",
    "reinforcement learning",
    "data mining",
    "data science",
    "recommendation system",
    "generative ai",
    "computer science",
    "algorithm",
    "optimization",
    "prediction",
    "dataset"
]


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^A-Za-z0-9.,;:!?()\-% ]",
        " ",
        text
    )

    return text.strip()


# =========================================================
# STATISTICS
# =========================================================

def calculate_statistics(text):

    words = re.findall(r"\b\w+\b", text)

    sentences = re.split(r"[.!?]+", text)

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    return {
        "words": len(words),
        "sentences": len(sentences),
        "characters": len(text),
        "paragraphs": len(paragraphs)
    }


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text):

    lower_text = text.lower()

    found = []

    for keyword in RESEARCH_KEYWORDS:

        pattern = r"\b" + re.escape(keyword) + r"\b"

        matches = re.findall(
            pattern,
            lower_text
        )

        if matches:
            found.append(
                (keyword, len(matches))
            )

    found.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return found[:15]


# =========================================================
# TF-IDF KEYWORDS
# =========================================================

def tfidf_keywords(text, top_n=15):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=100
        )

        matrix = vectorizer.fit_transform([text])

        scores = matrix.toarray()[0]

        words = vectorizer.get_feature_names_out()

        data = list(
            zip(words, scores)
        )

        data.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return data[:top_n]

    except:

        return []


# =========================================================
# SECTION DETECTION
# =========================================================

def extract_sections(text):

    sections = {}

    patterns = {
        "Abstract": r"abstract(.*?)(?=introduction|keywords|1\.|background|$)",
        "Introduction": r"(?:introduction|1\.\s*introduction)(.*?)(?=methodology|methods|2\.|literature review|$)",
        "Methodology": r"(?:methodology|methods|2\.\s*methods)(.*?)(?=results|3\.|experiments|$)",
        "Results": r"(?:results|3\.\s*results)(.*?)(?=discussion|conclusion|4\.|$)",
        "Discussion": r"(?:discussion|4\.\s*discussion)(.*?)(?=conclusion|5\.|$)",
        "Conclusion": r"(?:conclusion|5\.\s*conclusion)(.*?)(?=references|$)"
    }

    lower = text.lower()

    for name, pattern in patterns.items():

        match = re.search(
            pattern,
            lower,
            re.S
        )

        if match:

            content = match.group(1).strip()

            if len(content) > 50:

                sections[name] = content[:5000]

    return sections


# =========================================================
# GAP DETECTION
# =========================================================

def detect_research_gaps(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    gap_words = [
        "limitation",
        "limitations",
        "challenge",
        "challenges",
        "however",
        "future research",
        "future work",
        "lack",
        "limited",
        "shortcoming",
        "drawback",
        "problem",
        "remain",
        "remains",
        "not addressed",
        "further research"
    ]

    gaps = []

    for sentence in sentences:

        sentence_clean = sentence.strip()

        if len(sentence_clean) < 40:
            continue

        lower = sentence_clean.lower()

        if any(
            word in lower
            for word in gap_words
        ):

            gaps.append(
                sentence_clean
            )

    # remove duplicates

    unique = []

    for item in gaps:

        if item not in unique:

            unique.append(item)

    return unique[:10]


# =========================================================
# FUTURE WORK
# =========================================================

def detect_future_work(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    future_words = [
        "future work",
        "future research",
        "in future",
        "future studies",
        "further research",
        "further studies",
        "will investigate",
        "should investigate",
        "can be extended",
        "could be extended",
        "future direction"
    ]

    future = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 40:
            continue

        lower = sentence.lower()

        if any(
            word in lower
            for word in future_words
        ):

            future.append(sentence)

    return future[:8]


# =========================================================
# TOPIC DETECTION
# =========================================================

def detect_topic(keywords):

    if not keywords:

        return "General Research"

    top = keywords[0][0]

    topic_map = {

        "machine learning": "Machine Learning",

        "deep learning": "Deep Learning",

        "artificial intelligence": "Artificial Intelligence",

        "natural language processing":
            "Natural Language Processing",

        "computer vision":
            "Computer Vision",

        "transformer":
            "Transformers / NLP",

        "bert":
            "NLP / BERT",

        "llm":
            "Large Language Models",

        "large language model":
            "Large Language Models",

        "classification":
            "Machine Learning Classification",

        "recommendation system":
            "Recommendation Systems",

        "generative ai":
            "Generative AI"
    }

    return topic_map.get(
        top,
        "Artificial Intelligence / Computing"
    )


# =========================================================
# READABILITY
# =========================================================

def readability_score(text):

    words = re.findall(
        r"\b\w+\b",
        text
    )

    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        s for s in sentences
        if s.strip()
    ]

    if not words or not sentences:

        return 0

    avg_words = len(words) / len(sentences)

    score = max(
        0,
        min(
            100,
            100 - (avg_words * 2)
        )
    )

    return round(score, 1)


# =========================================================
# SIMILARITY SEARCH
# =========================================================

def search_paper(text, query):

    paragraphs = re.split(
        r"\n+",
        text
    )

    paragraphs = [
        p.strip()
        for p in paragraphs
        if len(p.strip()) > 50
    ]

    if not paragraphs:

        return []

    try:

        documents = paragraphs + [query]

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        similarity = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        )[0]

        results = []

        for i, score in enumerate(similarity):

            results.append(
                (
                    paragraphs[i],
                    score
                )
            )

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results[:5]

    except:

        return []


# =========================================================
# PDF REPORT
# =========================================================

def create_report(
    paper_name,
    statistics,
    topic,
    keywords,
    gaps,
    future_work
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "ResearchGap Finder Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Paper:</b> {paper_name}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Detected Research Area:</b> {topic}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    data = [
        ["Metric", "Value"],
        ["Words", statistics["words"]],
        ["Sentences", statistics["sentences"]],
        ["Characters", statistics["characters"]],
        ["Paragraphs", statistics["paragraphs"]]
    ]

    table = Table(data)

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Important Keywords",
            styles["Heading2"]
        )
    )

    keyword_text = ", ".join(
        [k[0] for k in keywords]
    )

    story.append(
        Paragraph(
            keyword_text or "No keywords found.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Potential Research Gaps",
            styles["Heading2"]
        )
    )

    for gap in gaps:

        story.append(
            Paragraph(
                "• " + gap,
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 5)
        )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Future Work",
            styles["Heading2"]
        )
    )

    for item in future_work:

        story.append(
            Paragraph(
                "• " + item,
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 5)
        )

    doc.build(story)

    buffer.seek(0)

    return buffer


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🔬 ResearchGap Finder")

    st.caption(
        "NLP-powered research paper analysis"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📄 Analyze Paper",
            "⚠️ Research Gaps",
            "🔑 Keywords",
            "🔍 Search Paper",
            "📊 Analytics",
            "📥 Report",
            "ℹ️ About"
        ]
    )

    st.divider()

    if st.session_state.paper_name:

        st.success(
            f"Paper loaded:\n\n"
            f"{st.session_state.paper_name}"
        )

    else:

        st.info(
            "Upload a research paper to begin."
        )


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>🔬 ResearchGap Finder</h1>

<p>
Analyze research papers, discover important topics,
extract keywords and identify potential research gaps.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Research Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.paper_text:

        st.info(
            "👈 Go to **Analyze Paper** and upload a PDF research paper."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("""
            <div class="card">

            <div class="card-title">
            📄 Document Analysis
            </div>

            <div class="card-value">
            PDF
            </div>

            <p>
            Extract and analyze research paper text.
            </p>

            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown("""
            <div class="card">

            <div class="card-title">
            ⚠️ Gap Detection
            </div>

            <div class="card-value">
            NLP
            </div>

            <p>
            Detect limitations and future research clues.
            </p>

            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown("""
            <div class="card">

            <div class="card-title">
            🔑 Keywords
            </div>

            <div class="card-value">
            TF-IDF
            </div>

            <p>
            Extract important research terms.
            </p>

            </div>
            """, unsafe_allow_html=True)

    else:

        text = st.session_state.paper_text

        stats = calculate_statistics(text)

        keywords = extract_keywords(text)

        topic = detect_topic(keywords)

        gaps = detect_research_gaps(text)

        future = detect_future_work(text)

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">Words</div>
                <div class="card-value">{stats["words"]:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">Sentences</div>
                <div class="card-value">{stats["sentences"]:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">Keywords</div>
                <div class="card-value">{len(keywords)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:

            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">Potential Gaps</div>
                <div class="card-value">{len(gaps)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"### 🎯 Detected Research Area: **{topic}**"
        )

        st.progress(
            min(
                len(gaps) / 10,
                1.0
            )
        )

        st.caption(
            "The gap count represents sentences containing "
            "common limitation/future-research indicators."
        )


# =========================================================
# ANALYZE PAPER
# =========================================================

elif page == "📄 Analyze Paper":

    st.markdown(
        '<div class="section-title">Upload Research Paper</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF research paper",
        type=["pdf"]
    )

    if uploaded_file:

        st.info(
            f"📄 Selected: {uploaded_file.name}"
        )

        if st.button(
            "🚀 Analyze Research Paper",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting and analyzing paper..."
            ):

                text = extract_pdf_text(
                    uploaded_file
                )

                text = clean_text(text)

                if len(text) < 100:

                    st.error(
                        "Could not extract enough text from this PDF."
                    )

                else:

                    st.session_state.paper_text = text

                    st.session_state.paper_name = (
                        uploaded_file.name
                    )

                    st.session_state.analysis_done = True

                    st.success(
                        "✅ Research paper analyzed successfully!"
                    )

                    st.rerun()


    if st.session_state.paper_text:

        st.divider()

        text = st.session_state.paper_text

        stats = calculate_statistics(text)

        keywords = extract_keywords(text)

        topic = detect_topic(keywords)

        sections = extract_sections(text)

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Words",
                f"{stats['words']:,}"
            )

        with col2:

            st.metric(
                "Sentences",
                f"{stats['sentences']:,}"
            )

        with col3:

            st.metric(
                "Characters",
                f"{stats['characters']:,}"
            )

        with col4:

            st.metric(
                "Detected Topic",
                topic
            )

        st.markdown("### 📑 Paper Sections")

        for section, content in sections.items():

            with st.expander(section):

                st.write(
                    content[:2000]
                )


# =========================================================
# RESEARCH GAPS
# =========================================================

elif page == "⚠️ Research Gaps":

    st.markdown(
        '<div class="section-title">Potential Research Gaps</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.paper_text:

        st.warning(
            "Please upload a research paper first."
        )

    else:

        gaps = detect_research_gaps(
            st.session_state.paper_text
        )

        future = detect_future_work(
            st.session_state.paper_text
        )

        st.subheader(
            f"⚠️ Potential Gaps Found: {len(gaps)}"
        )

        if gaps:

            for gap in gaps:

                st.markdown(
                    f"""
                    <div class="gap-box">
                    <b>Potential Gap</b><br><br>
                    {gap}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No obvious limitation-related sentences were detected."
            )

        st.markdown(
            "### 🔮 Future Research Directions"
        )

        if future:

            for item in future:

                st.markdown(
                    f"""
                    <div class="future-box">
                    {item}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No explicit future-work sentences were detected."
            )

        st.caption(
            "Important: these are NLP-based indicators, "
            "not verified scientific research gaps."
        )


# =========================================================
# KEYWORDS
# =========================================================

elif page == "🔑 Keywords":

    st.markdown(
        '<div class="section-title">Research Keywords</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.paper_text:

        st.warning(
            "Upload a paper first."
        )

    else:

        text = st.session_state.paper_text

        keywords = extract_keywords(text)

        tfidf = tfidf_keywords(text)

        st.subheader("🎯 Research Keywords")

        if keywords:

            for word, count in keywords:

                st.markdown(
                    f"""
                    <span class="keyword">
                    {word} · {count}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No predefined research keywords found."
            )

        st.divider()

        st.subheader(
            "📊 TF-IDF Important Terms"
        )

        if tfidf:

            df = pd.DataFrame(
                tfidf,
                columns=[
                    "Term",
                    "TF-IDF Score"
                ]
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# SEARCH PAPER
# =========================================================

elif page == "🔍 Search Paper":

    st.markdown(
        '<div class="section-title">Search Inside Paper</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.paper_text:

        st.warning(
            "Upload a paper first."
        )

    else:

        query = st.text_input(
            "What do you want to find?",
            placeholder="Example: What dataset was used?"
        )

        if query:

            results = search_paper(
                st.session_state.paper_text,
                query
            )

            if results:

                st.subheader(
                    "🔎 Relevant Sections"
                )

                for i, (paragraph, score) in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### Result {i}"
                    )

                    st.progress(
                        min(
                            float(score),
                            1.0
                        )
                    )

                    st.write(
                        f"Similarity: **{score * 100:.1f}%**"
                    )

                    st.info(
                        paragraph[:1500]
                    )

            else:

                st.info(
                    "No relevant content found."
                )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-title">Paper Analytics</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.paper_text:

        st.warning(
            "Upload a paper first."
        )

    else:

        text = st.session_state.paper_text

        stats = calculate_statistics(text)

        keywords = extract_keywords(text)

        readability = readability_score(text)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Average Words / Sentence",
                round(
                    stats["words"] /
                    max(stats["sentences"], 1),
                    2
                )
            )

        with col2:

            st.metric(
                "Readability Indicator",
                f"{readability}/100"
            )

        with col3:

            st.metric(
                "Research Keywords",
                len(keywords)
            )

        st.markdown("### 📈 Keyword Frequency")

        if keywords:

            df = pd.DataFrame(
                keywords,
                columns=[
                    "Keyword",
                    "Frequency"
                ]
            )

            st.bar_chart(
                df.set_index("Keyword")
            )

        st.markdown("### 📊 Document Statistics")

        stat_df = pd.DataFrame({
            "Metric": [
                "Words",
                "Sentences",
                "Characters",
                "Paragraphs"
            ],
            "Value": [
                stats["words"],
                stats["sentences"],
                stats["characters"],
                stats["paragraphs"]
            ]
        })

        st.dataframe(
            stat_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# REPORT
# =========================================================

elif page == "📥 Report":

    st.markdown(
        '<div class="section-title">Research Analysis Report</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.paper_text:

        st.warning(
            "Upload a paper first."
        )

    else:

        text = st.session_state.paper_text

        stats = calculate_statistics(text)

        keywords = extract_keywords(text)

        topic = detect_topic(keywords)

        gaps = detect_research_gaps(text)

        future = detect_future_work(text)

        report = create_report(
            st.session_state.paper_name,
            stats,
            topic,
            keywords,
            gaps,
            future
        )

        st.success(
            "Your research analysis report is ready."
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=report,
            file_name="research_gap_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">About ResearchGap Finder</div>',
        unsafe_allow_html=True
    )

    st.write("""
    **ResearchGap Finder** is an NLP-based research paper
    analysis application.

    It helps students and researchers quickly inspect
    research papers and identify sentences that may indicate
    limitations, challenges and future research directions.
    """)

    st.markdown("### 🛠️ Technologies")

    st.write("""
    - Python
    - Streamlit
    - PyPDF
    - Scikit-learn
    - TF-IDF
    - Cosine Similarity
    - Regular Expressions
    - ReportLab
    """)

    st.markdown("### 🧠 NLP Techniques")

    st.write("""
    1. Text preprocessing
    2. Keyword extraction
    3. TF-IDF
    4. Text similarity
    5. Sentence matching
    6. Pattern-based research-gap detection
    7. Document statistics
    """)

    st.info(
        "This system provides potential research-gap indicators. "
        "A researcher should manually verify whether a detected "
        "point is actually a research gap."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🔬 <b>ResearchGap Finder</b><br>
NLP-Based Research Paper Intelligence System

</div>
""", unsafe_allow_html=True)
