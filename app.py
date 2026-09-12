import streamlit as st
import requests
import pymupdf
import faiss
import numpy as np
import json
import ast
import operator
import math


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

CHAT_MODEL = "gemma3:1b"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4


# ============================================================
# OLLAMA CHAT
# ============================================================

def ask_ollama(messages, json_mode=False):

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False
    }

    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# ============================================================
# OLLAMA EMBEDDINGS
# ============================================================

def get_embeddings(texts):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["embeddings"]


# ============================================================
# AGENT ROUTER
# ============================================================

def route_request(user_input):

    router_prompt = f"""
You are an AI agent router.

Your job is to decide which route should handle
the user's request.

Available routes:

CHAT
Use CHAT for:
- normal conversation
- explanations
- general knowledge
- greetings
- programming questions

RAG
Use RAG for:
- questions about uploaded PDF documents
- questions referring to a document
- questions asking "according to the PDF"
- questions about a CV, paper, report, article,
  or uploaded file

CALCULATOR
Use CALCULATOR for:
- arithmetic calculations
- mathematical expressions
- numerical calculations

Return ONLY valid JSON.

Allowed outputs:

{{"route": "CHAT"}}

{{"route": "RAG"}}

{{"route": "CALCULATOR"}}

User message:

{user_input}
"""

    try:

        result = ask_ollama(
            [
                {
                    "role": "user",
                    "content": router_prompt
                }
            ],
            json_mode=True
        )

        data = json.loads(result)

        route = data.get(
            "route",
            "CHAT"
        ).upper()

        if route not in [
            "CHAT",
            "RAG",
            "CALCULATOR"
        ]:
            return "CHAT"

        return route

    except Exception:
        return "CHAT"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text(
            "text"
        ).strip()

        if text:

            pages.append({
                "filename": uploaded_file.name,
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(pages):

    chunks = []

    for page_data in pages:

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "filename": page_data[
                        "filename"
                    ],
                    "page": page_data[
                        "page"
                    ]
                })

            start += (
                CHUNK_SIZE
                - CHUNK_OVERLAP
            )

    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = get_embeddings(
        texts
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # Normalize vectors
    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    # Inner Product after normalization
    # becomes cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# RETRIEVE DOCUMENT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    k=TOP_K
):

    question_embedding = (
        get_embeddings(question)[0]
    )

    question_vector = np.array(
        [question_embedding],
        dtype="float32"
    )

    faiss.normalize_L2(
        question_vector
    )

    scores, indices = index.search(
        question_vector,
        min(k, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = chunks[idx].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(
    retrieved_chunks
):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        source = (
            f"[Source {i}: "
            f"{chunk['filename']}, "
            f"page {chunk['page']}]"
        )

        context_parts.append(
            f"{source}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# CALCULATOR
# ============================================================

BINARY_OPERATORS = {

    ast.Add:
        operator.add,

    ast.Sub:
        operator.sub,

    ast.Mult:
        operator.mul,

    ast.Div:
        operator.truediv,

    ast.Mod:
        operator.mod,

    ast.Pow:
        operator.pow,

    ast.FloorDiv:
        operator.floordiv
}


UNARY_OPERATORS = {

    ast.USub:
        operator.neg,

    ast.UAdd:
        operator.pos
}


ALLOWED_FUNCTIONS = {

    "sqrt":
        math.sqrt,

    "sin":
        math.sin,

    "cos":
        math.cos,

    "tan":
        math.tan,

    "log":
        math.log,

    "log10":
        math.log10,

    "abs":
        abs,

    "round":
        round
}


def safe_calculate(
    expression
):

    if len(expression) > 200:
        raise ValueError(
            "Expression is too long."
        )

    tree = ast.parse(
        expression,
        mode="eval"
    )

    def evaluate(node):

        if isinstance(
            node,
            ast.Constant
        ):

            if isinstance(
                node.value,
                (int, float)
            ):
                return node.value

            raise ValueError(
                "Invalid constant."
            )

        if isinstance(
            node,
            ast.BinOp
        ):

            left = evaluate(
                node.left
            )

            right = evaluate(
                node.right
            )

            op_type = type(
                node.op
            )

            if (
                op_type
                not in
                BINARY_OPERATORS
            ):
                raise ValueError(
                    "Unsupported operator."
                )

            return (
                BINARY_OPERATORS[
                    op_type
                ](
                    left,
                    right
                )
            )

        if isinstance(
            node,
            ast.UnaryOp
        ):

            operand = evaluate(
                node.operand
            )

            op_type = type(
                node.op
            )

            if (
                op_type
                not in
                UNARY_OPERATORS
            ):
                raise ValueError(
                    "Unsupported unary operator."
                )

            return (
                UNARY_OPERATORS[
                    op_type
                ](
                    operand
                )
            )

        if isinstance(
            node,
            ast.Call
        ):

            if not isinstance(
                node.func,
                ast.Name
            ):

                raise ValueError(
                    "Invalid function."
                )

            function_name = (
                node.func.id
            )

            if (
                function_name
                not in
                ALLOWED_FUNCTIONS
            ):

                raise ValueError(
                    f"Function "
                    f"'{function_name}' "
                    f"is not allowed."
                )

            arguments = [
                evaluate(arg)
                for arg
                in node.args
            ]

            return (
                ALLOWED_FUNCTIONS[
                    function_name
                ](
                    *arguments
                )
            )

        raise ValueError(
            "Invalid mathematical expression."
        )

    return evaluate(
        tree.body
    )


# ============================================================
# EXTRACT MATHEMATICAL EXPRESSION
# ============================================================

def extract_math_expression(
    user_input
):

    prompt = f"""
Extract the mathematical expression from the user's message.

Return ONLY valid JSON.

Example:

User:
Calculate 200 * 3 + 15

Output:

{{
    "expression":
    "200 * 3 + 15"
}}

Example:

User:
What is the square root of 144?

Output:

{{
    "expression":
    "sqrt(144)"
}}

User message:

{user_input}
"""

    result = ask_ollama(
        [
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ],
        json_mode=True
    )

    data = json.loads(
        result
    )

    return data[
        "expression"
    ]


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Local AI Agent",
    page_icon="🤖",
    layout="wide"
)


st.title(
    "🤖 Local AI Agent"
)


st.caption(
    "Gemma 3 1B + Ollama + "
    "FAISS + RAG + Tool Calling"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "chunks" not in st.session_state:

    st.session_state.chunks = []


if (
    "faiss_index"
    not in
    st.session_state
):

    st.session_state.faiss_index = None


if (
    "processed_files"
    not in
    st.session_state
):

    st.session_state.processed_files = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📄 Documents"
    )

    uploaded_files = (
        st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
    )

    if uploaded_files:

        current_files = tuple(

            (
                file.name,
                file.size
            )

            for file
            in uploaded_files
        )

        if (
            current_files
            !=
            st.session_state.processed_files
        ):

            with st.spinner(
                "Creating semantic "
                "embeddings..."
            ):

                try:

                    all_pages = []

                    for uploaded_file \
                            in uploaded_files:

                        pages = (
                            extract_pdf(
                                uploaded_file
                            )
                        )

                        all_pages.extend(
                            pages
                        )

                    chunks = (
                        create_chunks(
                            all_pages
                        )
                    )

                    if not chunks:

                        st.error(
                            "No readable text "
                            "was found."
                        )

                    else:

                        index = (
                            build_faiss_index(
                                chunks
                            )
                        )

                        st.session_state.chunks = (
                            chunks
                        )

                        (
                            st.session_state
                            .faiss_index
                        ) = index

                        (
                            st.session_state
                            .processed_files
                        ) = current_files

                        st.success(
                            f"{len(uploaded_files)} "
                            f"PDF(s) processed."
                        )

                except Exception as e:

                    st.error(
                        f"Document processing "
                        f"error:\n{e}"
                    )


    # --------------------------------------------------------
    # RAG STATUS
    # --------------------------------------------------------

    if st.session_state.chunks:

        st.divider()

        st.success(
            "RAG system ready"
        )

        st.write(
            f"**Chunks:** "
            f"{len(st.session_state.chunks)}"
        )

        st.write(
            f"**Embedding:** "
            f"`{EMBEDDING_MODEL}`"
        )

        st.write(
            f"**LLM:** "
            f"`{CHAT_MODEL}`"
        )


    st.divider()


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# SHOW PREVIOUS CHAT
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me something..."
)


if user_input:

    # --------------------------------------------------------
    # STORE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # ========================================================
    # AGENT ROUTER
    # ========================================================

    with st.spinner(
        "Agent is choosing a tool..."
    ):

        route = route_request(
            user_input
        )


    st.caption(
        f"🧠 Agent Route: "
        f"**{route}**"
    )


    retrieved_chunks = []


    # ========================================================
    # CALCULATOR ROUTE
    # ========================================================

    if route == "CALCULATOR":

        try:

            expression = (
                extract_math_expression(
                    user_input
                )
            )

            result = (
                safe_calculate(
                    expression
                )
            )

            # Let Gemma formulate
            # the final answer.
            final_prompt = f"""
The user asked:

{user_input}

The calculator tool evaluated:

{expression}

Calculator result:

{result}

Give the user a short and clear final answer.
"""

            answer = ask_ollama(
                [
                    {
                        "role":
                            "system",

                        "content":
                            "You are a helpful "
                            "AI assistant."
                    },
                    {
                        "role":
                            "user",

                        "content":
                            final_prompt
                    }
                ]
            )

        except Exception as e:

            answer = (
                f"Calculator error: "
                f"{e}"
            )


    # ========================================================
    # RAG ROUTE
    # ========================================================

    elif route == "RAG":

        if (
            st.session_state.faiss_index
            is None
            or
            not st.session_state.chunks
        ):

            answer = (
                "I need an uploaded PDF "
                "to answer this document-based "
                "question. Please upload a PDF "
                "from the sidebar."
            )

        else:

            try:

                retrieved_chunks = (
                    retrieve_chunks(
                        user_input,
                        st.session_state
                        .faiss_index,
                        st.session_state
                        .chunks
                    )
                )

                context = (
                    build_context(
                        retrieved_chunks
                    )
                )


                rag_prompt = f"""
You are a document question-answering assistant.

Use ONLY the document context below to answer the user's question.

Rules:

1. Do not invent information.

2. If the answer cannot be found in the retrieved context,
say that the information could not be found in the uploaded documents.

3. Cite relevant sources using labels such as:
[Source 1]
[Source 2]

4. Keep the answer clear and concise.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{user_input}
"""


                answer = ask_ollama(
                    [
                        {
                            "role":
                                "system",

                            "content":
                                "You are a "
                                "document assistant."
                        },
                        {
                            "role":
                                "user",

                            "content":
                                rag_prompt
                        }
                    ]
                )

            except Exception as e:

                answer = (
                    f"RAG error: {e}"
                )


    # ========================================================
    # CHAT ROUTE
    # ========================================================

    else:

        system_prompt = """
You are a helpful local AI assistant.

Answer clearly and concisely.

You are running locally through Ollama.
"""


        messages = [
            {
                "role":
                    "system",

                "content":
                    system_prompt
            }
        ]


        # Keep recent conversation
        messages += (
            st.session_state
            .messages[-8:]
        )


        try:

            answer = ask_ollama(
                messages
            )

        except Exception as e:

            answer = (
                f"Ollama error: {e}"
            )


    # ========================================================
    # SHOW ANSWER
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # RETRIEVED RAG SOURCES
        # ----------------------------------------------------

        if (
            route == "RAG"
            and retrieved_chunks
        ):

            with st.expander(
                "📚 Retrieved Sources"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"### Source {i}"
                    )

                    st.write(
                        f"**File:** "
                        f"{chunk['filename']}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{chunk['page']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{chunk['score']:.3f}"
                    )

                    st.text(
                        chunk["text"][
                            :700
                        ]
                    )

                    st.divider()


    # ========================================================
    # STORE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role":
                "assistant",

            "content":
                answer
        }
    )