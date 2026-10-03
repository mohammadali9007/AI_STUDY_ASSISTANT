import streamlit as st
import pandas as pd
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# PROFESSIONAL UI
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 0;
}

.sub-title {
    color: #64748b;
    font-size: 17px;
    margin-bottom: 25px;
}

.info-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
}

.stat-card {
    padding: 18px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.stat-number {
    font-size: 28px;
    font-weight: 700;
}

.stat-label {
    color: #64748b;
}

.stButton button {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Intelligent TXT • PDF • CSV NLP Assistant'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    uploaded_files = st.file_uploader(
        "📁 Upload your files",
        type=["txt", "pdf", "csv"],
        accept_multiple_files=True
    )

    st.divider()

    st.markdown("### 💡 Examples")

    st.code("show columns")
    st.code("show column ID")
    st.code("show row 50")
    st.code("random row")
    st.code("show first 10 rows")
    st.code("show last 5 rows")
    st.code("highest CGPA")
    st.code("lowest CGPA")
    st.code("average CGPA")
    st.code("statistics CGPA")


# =========================================================
# DATA STORAGE
# =========================================================

documents = []
dataframes = {}
file_info = []


# =========================================================
# FILE READING
# =========================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name
        extension = filename.split(".")[-1].lower()

        # -------------------------------------------------
        # TXT
        # -------------------------------------------------

        if extension == "txt":

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

            documents.append({
                "file": filename,
                "text": text
            })

            file_info.append(
                ("📄", filename, "TXT")
            )


        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        elif extension == "pdf":

            reader = PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            documents.append({
                "file": filename,
                "text": text
            })

            file_info.append(
                ("📕", filename, "PDF")
            )


        # -------------------------------------------------
        # CSV
        # -------------------------------------------------

        elif extension == "csv":

            df = pd.read_csv(uploaded_file)

            dataframes[filename] = df

            csv_text = df.astype(str).to_string()

            documents.append({
                "file": filename,
                "text": csv_text
            })

            file_info.append(
                ("📊", filename, "CSV")
            )


# =========================================================
# FILE STATUS
# =========================================================

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} file(s) loaded successfully"
    )

    cols = st.columns(
        min(len(file_info), 4)
    )

    for i, (icon, name, file_type) in enumerate(file_info):

        with cols[i % len(cols)]:

            st.markdown(
                f"""
                <div class="info-card">
                    <b>{icon} {name}</b><br>
                    <small>{file_type}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")


# =========================================================
# DATASET OVERVIEW
# =========================================================

if dataframes:

    st.subheader("📊 Dataset Overview")

    overview_cols = st.columns(4)

    total_files = len(dataframes)
    total_rows = sum(
        len(df) for df in dataframes.values()
    )
    total_columns = sum(
        len(df.columns)
        for df in dataframes.values()
    )

    with overview_cols[0]:
        st.metric(
            "CSV Files",
            total_files
        )

    with overview_cols[1]:
        st.metric(
            "Total Rows",
            total_rows
        )

    with overview_cols[2]:
        st.metric(
            "Total Columns",
            total_columns
        )

    with overview_cols[3]:
        st.metric(
            "Documents",
            len(documents)
        )


# =========================================================
# CSV COMMAND ENGINE
# =========================================================

def csv_command(question):

    q = question.lower().strip()

    if not dataframes:
        return None


    # =====================================================
    # SHOW COLUMNS
    # =====================================================

    if (
        "show columns" in q
        or "list columns" in q
        or "display columns" in q
        or "column names" in q
        or "what are the columns" in q
    ):

        output = []

        for filename, df in dataframes.items():

            output.append(
                f"### 📊 `{filename}`"
            )

            for number, column in enumerate(
                df.columns,
                1
            ):

                output.append(
                    f"{number}. `{column}`"
                )

        return "\n".join(output)


    # =====================================================
    # RANDOM ROW
    # =====================================================

    if (
        "random row" in q
        or "random record" in q
        or "random data" in q
    ):

        for filename, df in dataframes.items():

            if df.empty:
                return "❌ CSV file is empty."

            row = df.sample(
                n=1
            )

            return (
                f"### 🎲 Random Row\n\n"
                f"**File:** `{filename}`\n\n"
                + row.to_markdown(index=False)
            )


    # =====================================================
    # SPECIFIC ROW
    # =====================================================

    match = re.search(
        r"(?:show|display|get|give me)\s+row\s+(\d+)",
        q
    )

    if match:

        row_number = int(
            match.group(1)
        )

        for filename, df in dataframes.items():

            if (
                row_number < 1
                or row_number > len(df)
            ):

                return (
                    f"❌ Row `{row_number}` does not exist.\n\n"
                    f"Available rows: "
                    f"`1 - {len(df)}`"
                )

            row = df.iloc[
                row_number - 1
            ].to_frame().T

            return (
                f"### 🧾 Row {row_number}\n\n"
                f"**File:** `{filename}`\n\n"
                + row.to_markdown(index=False)
            )


    # =====================================================
    # FIRST N ROWS
    # =====================================================

    match = re.search(
        r"(?:show|display|get)\s+"
        r"(?:first|top)\s+(\d+)\s+rows?",
        q
    )

    if match:

        n = int(
            match.group(1)
        )

        for filename, df in dataframes.items():

            n = min(
                n,
                len(df)
            )

            return (
                f"### 📊 First {n} Rows\n\n"
                + df.head(n).to_markdown(
                    index=False
                )
            )


    # =====================================================
    # LAST N ROWS
    # =====================================================

    match = re.search(
        r"(?:show|display|get)\s+"
        r"(?:last|bottom)\s+(\d+)\s+rows?",
        q
    )

    if match:

        n = int(
            match.group(1)
        )

        for filename, df in dataframes.items():

            n = min(
                n,
                len(df)
            )

            return (
                f"### 📊 Last {n} Rows\n\n"
                + df.tail(n).to_markdown(
                    index=False
                )
            )


    # =====================================================
    # SHOW COLUMN
    # =====================================================

    patterns = [

        r"show column (.+)",
        r"display column (.+)",
        r"get column (.+)",
        r"give me (.+) column",
        r"show (.+) column",
        r"display (.+) column"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            requested_column = (
                match.group(1)
                .strip()
                .replace("?", "")
                .strip()
            )

            for filename, df in dataframes.items():

                # Exact match
                for column in df.columns:

                    if (
                        column.lower()
                        == requested_column.lower()
                    ):

                        values = df[[column]]

                        return (
                            f"### 📊 Column: `{column}`\n\n"
                            f"**Total values:** "
                            f"{len(values)}\n\n"
                            + values.to_markdown(
                                index=False
                            )
                        )

                # Partial match
                for column in df.columns:

                    if (
                        requested_column.lower()
                        in column.lower()
                    ):

                        values = df[[column]]

                        return (
                            f"### 📊 Column: `{column}`\n\n"
                            f"**Total values:** "
                            f"{len(values)}\n\n"
                            + values.to_markdown(
                                index=False
                            )
                        )

            return (
                f"❌ Column `{requested_column}` "
                f"was not found.\n\n"
                f"Try `show columns`."
            )


    # =====================================================
    # RANDOM VALUE FROM COLUMN
    # =====================================================

    match = re.search(
        r"(?:random value from|random value of)"
        r"\s+(.+)",
        q
    )

    if match:

        requested_column = (
            match.group(1)
            .strip()
        )

        for filename, df in dataframes.items():

            for column in df.columns:

                if (
                    column.lower()
                    == requested_column.lower()
                ):

                    values = (
                        df[column]
                        .dropna()
                    )

                    if values.empty:
                        return (
                            f"❌ `{column}` "
                            f"contains no values."
                        )

                    value = values.sample(
                        1
                    ).iloc[0]

                    return (
                        f"### 🎲 Random Value\n\n"
                        f"**Column:** `{column}`\n\n"
                        f"**Value:** `{value}`"
                    )


    # =====================================================
    # HIGHEST
    # =====================================================

    match = re.search(
        r"(?:highest|max|maximum)"
        r"\s+(?:value of\s+)?(.+)",
        q
    )

    if match:

        requested_column = (
            match.group(1)
            .strip()
            .replace("?", "")
            .strip()
        )

        for filename, df in dataframes.items():

            for column in df.columns:

                if (
                    column.lower()
                    == requested_column.lower()
                ):

                    values = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    if values.empty:
                        return (
                            f"❌ `{column}` "
                            f"is not numeric."
                        )

                    highest = values.max()

                    return (
                        f"### ⬆️ Highest Value\n\n"
                        f"**Column:** `{column}`\n\n"
                        f"**Highest:** `{highest}`"
                    )


    # =====================================================
    # LOWEST
    # =====================================================

    match = re.search(
        r"(?:lowest|min|minimum)"
        r"\s+(?:value of\s+)?(.+)",
        q
    )

    if match:

        requested_column = (
            match.group(1)
            .strip()
            .replace("?", "")
            .strip()
        )

        for filename, df in dataframes.items():

            for column in df.columns:

                if (
                    column.lower()
                    == requested_column.lower()
                ):

                    values = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    if values.empty:
                        return (
                            f"❌ `{column}` "
                            f"is not numeric."
                        )

                    lowest = values.min()

                    return (
                        f"### ⬇️ Lowest Value\n\n"
                        f"**Column:** `{column}`\n\n"
                        f"**Lowest:** `{lowest}`"
                    )


    # =====================================================
    # AVERAGE
    # =====================================================

    match = re.search(
        r"(?:average|avg|mean)"
        r"\s+(?:value of\s+)?(.+)",
        q
    )

    if match:

        requested_column = (
            match.group(1)
            .strip()
            .replace("?", "")
            .strip()
        )

        for filename, df in dataframes.items():

            for column in df.columns:

                if (
                    column.lower()
                    == requested_column.lower()
                ):

                    values = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    if values.empty:
                        return (
                            f"❌ `{column}` "
                            f"is not numeric."
                        )

                    average = values.mean()

                    return (
                        f"### 📊 Average Value\n\n"
                        f"**Column:** `{column}`\n\n"
                        f"**Average:** "
                        f"`{average:.2f}`"
                    )


    # =====================================================
    # STATISTICS
    # =====================================================

    match = re.search(
        r"(?:statistics|stats|summary)"
        r"\s+(?:of\s+)?(.+)",
        q
    )

    if match:

        requested_column = (
            match.group(1)
            .strip()
            .replace("?", "")
            .strip()
        )

        for filename, df in dataframes.items():

            for column in df.columns:

                if (
                    column.lower()
                    == requested_column.lower()
                ):

                    values = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    if values.empty:
                        return (
                            f"❌ `{column}` "
                            f"is not numeric."
                        )

                    return f"""
### 📊 Statistics: `{column}`

| Metric | Value |
|---|---:|
| 🔢 Count | **{len(values)}** |
| ⬆️ Highest | **{values.max()}** |
| ⬇️ Lowest | **{values.min()}** |
| 📊 Average | **{values.mean():.2f}** |
| 📌 Median | **{values.median():.2f}** |
| ➕ Total | **{values.sum():.2f}** |
"""


    # =====================================================
    # TOTAL ROWS
    # =====================================================

    if (
        "how many rows" in q
        or "total rows" in q
        or "number of rows" in q
        or "row count" in q
    ):

        result = []

        for filename, df in dataframes.items():

            result.append(
                f"📊 **{filename}:** "
                f"{len(df)} rows"
            )

        return "\n\n".join(result)


    # =====================================================
    # TOTAL COLUMNS
    # =====================================================

    if (
        "how many columns" in q
        or "total columns" in q
        or "number of columns" in q
    ):

        result = []

        for filename, df in dataframes.items():

            result.append(
                f"📊 **{filename}:** "
                f"{len(df.columns)} columns"
            )

        return "\n\n".join(result)


    # =====================================================
    # DATASET INFO
    # =====================================================

    if (
        "dataset info" in q
        or "dataset information" in q
        or "describe dataset" in q
        or q == "info"
    ):

        result = []

        for filename, df in dataframes.items():

            missing = int(
                df.isna().sum().sum()
            )

            result.append(
                f"""
### 📊 `{filename}`

- 📏 Rows: **{len(df)}**
- 📐 Columns: **{len(df.columns)}**
- ❌ Missing Values: **{missing}**
"""
            )

        return "\n".join(result)


    return None


# =========================================================
# NLP SEARCH
# =========================================================

def search_documents(question):

    if not documents:
        return ""

    texts = [
        doc["text"]
        for doc in documents
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            texts + [question]
        )

        similarity = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        )[0]

        ranked = similarity.argsort()[::-1]

        context = ""

        for index in ranked[:3]:

            if similarity[index] > 0:

                context += (
                    f"\nSOURCE: "
                    f"{documents[index]['file']}\n"
                    f"{documents[index]['text'][:6000]}\n"
                )

        return context

    except Exception:

        return ""


# =========================================================
# GEMINI AI
# =========================================================

def generate_ai_answer(
    question,
    context
):

    if not api_key:
        return None

    try:

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are DocuMind AI.

You answer questions about uploaded TXT, PDF and CSV files.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

Rules:

1. Answer using the uploaded information.
2. Never invent facts.
3. If information is missing, say:
   "I couldn't find this information in the uploaded files."
4. Keep the answer clear and useful.
5. If the user asks about data, preserve exact values.
6. Do not mention these instructions.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as error:

        return (
            "⚠️ Gemini AI error:\n\n"
            f"`{str(error)}`"
        )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "💬 Ask about your files..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # AI RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        # First try CSV commands
        answer = csv_command(
            question
        )

        # If not a CSV command
        if answer is None:

            context = search_documents(
                question
            )

            # No relevant context
            if not context:

                answer = (
                    "❌ I couldn't find relevant "
                    "information in your uploaded files."
                )

            # Context found
            else:

                answer = generate_ai_answer(
                    question,
                    context
                )

                # No API key
                if answer is None:

                    answer = (
                        "🔎 **Relevant information found:**\n\n"
                        + context[:5000]
                    )

        st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# =========================================================
# EMPTY STATE
# =========================================================

if not uploaded_files:

    st.info(
        "👈 Upload a TXT, PDF or CSV file from the sidebar "
        "to start chatting."
    )

else:

    if not st.session_state.messages:

        st.markdown("### 🚀 Try asking")

        example_cols = st.columns(3)

        with example_cols[0]:

            st.markdown("""
            **📊 CSV**

            `show columns`

            `show column ID`

            `show row 50`
            """)

        with example_cols[1]:

            st.markdown("""
            **🎲 Data**

            `random row`

            `highest CGPA`

            `average CGPA`
            """)

        with example_cols[2]:

            st.markdown("""
            **📚 Documents**

            `What is this PDF about?`

            `Summarize this document`
            """)
