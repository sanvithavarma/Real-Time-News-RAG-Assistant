
import os
import requests
import streamlit as st
import numpy as np
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ==========================================
# 2. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Real-Time News RAG Assistant",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Real-Time News RAG Assistant")

st.write(
    "Search recent news and ask questions about the articles."
)


# ==========================================
# 3. CHECK API KEYS
# ==========================================

if not NEWS_API_KEY:
    st.error("NEWS_API_KEY is missing in your .env file.")
    st.stop()

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing in your .env file.")
    st.stop()


# ==========================================
# 4. INITIALIZE GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# 5. LOAD SENTENCE TRANSFORMER MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("Loading embedding model..."):

    model = load_embedding_model()


# ==========================================
# 6. INITIALIZE SESSION STATE
# ==========================================

if "articles" not in st.session_state:
    st.session_state.articles = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "topic" not in st.session_state:
    st.session_state.topic = ""


# ==========================================
# 7. FETCH NEWS ARTICLES
# ==========================================

def fetch_news(topic):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 30,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":

        raise ValueError(
            data.get("message", "NewsAPI request failed")
        )

    articles = []

    for item in data.get("articles", []):

        title = item.get("title") or ""
        description = item.get("description") or ""
        content = item.get("content") or ""

        article_text = f"""
        Title: {title}
        Description: {description}
        Content: {content}
        """

        if len(article_text.strip()) > 30:

            articles.append({
                "text": article_text.strip(),
                "title": title,
                "source": (
                    item.get("source", {}).get("name", "Unknown")
                ),
                "url": item.get("url", ""),
                "published": item.get("publishedAt", "")
            })

    return articles


# ==========================================
# 8. SPLIT ARTICLES INTO CHUNKS
# ==========================================

def create_chunks(articles):

    chunks = []

    chunk_size = 120
    overlap = 30

    for article in articles:

        words = article["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "title": article["title"],
                    "source": article["source"],
                    "url": article["url"],
                    "published": article["published"]
                })

            start += chunk_size - overlap

    return chunks


# ==========================================
# 9. CREATE TEXT EMBEDDINGS
# ==========================================

def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# ==========================================
# 10. CREATE FAISS VECTOR DATABASE
# ==========================================

def create_vector_database(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ==========================================
# 11. RETRIEVE TOP-K DOCUMENTS
# ==========================================

def retrieve_documents(
    question,
    index,
    chunks,
    top_k=5
):

    if index is None or not chunks:

        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    k = min(top_k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        k
    )

    retrieved_chunks = []

    for score, idx in zip(scores[0], indices[0]):

        if idx != -1:

            result = chunks[idx].copy()

            result["score"] = float(score)

            retrieved_chunks.append(result)

    return retrieved_chunks


# ==========================================
# 12. GENERATE ANSWER USING GEMINI
# ==========================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:

        return "No relevant news articles were found."

    context = ""

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context += f"""
        Article {number}

        Title: {chunk['title']}

        Source: {chunk['source']}

        Published: {chunk['published']}

        Content: {chunk['text']}

        URL: {chunk['url']}

        -------------------------
        """

    prompt = f"""
    You are a helpful Real-Time News Research Assistant.

    Answer the user's question using only the news context
    provided below.

    Instructions:

    1. Give a clear and simple answer.
    2. Do not invent facts.
    3. Use only the supplied news context.
    4. If the answer is not available in the context,
       clearly say that the available articles do not provide
       enough information.
    5. Mention relevant article titles.
    6. Organize the answer using bullet points when useful.

    NEWS CONTEXT:

    {context}

    USER QUESTION:

    {question}
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:

        raise ValueError(
            "Gemini returned an empty response. Please try again."
        )

    return response.text


# ==========================================
# 13. SIDEBAR - NEWS SEARCH
# ==========================================

st.sidebar.header("📰 News Search")

topic = st.sidebar.text_input(
    "Enter a news topic",
    value="Artificial Intelligence"
)

if st.sidebar.button("Fetch Latest News"):

    if not topic.strip():

        st.sidebar.warning(
            "Please enter a news topic."
        )

    else:

        try:

            with st.spinner("Fetching latest news..."):

                # Fetch articles
                articles = fetch_news(topic)

                if not articles:

                    st.session_state.articles = []
                    st.session_state.chunks = []
                    st.session_state.index = None
                    st.session_state.topic = ""

                    st.warning(
                        "No articles found for this topic."
                    )

                else:

                    # Create chunks
                    chunks = create_chunks(articles)

                    if not chunks:

                        raise ValueError(
                            "No usable article chunks were created."
                        )

                    # Generate embeddings
                    embeddings = create_embeddings(chunks)

                    # Create FAISS database
                    index = create_vector_database(embeddings)

                    # Store everything in session state
                    st.session_state.articles = articles
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.topic = topic

                    st.success(
                        f"Loaded {len(articles)} articles "
                        f"and created {len(chunks)} chunks."
                    )

        except requests.RequestException as error:

            st.error(f"News API error: {error}")

        except Exception as error:

            st.error(f"News processing error: {error}")


# ==========================================
# 14. DISPLAY FETCHED NEWS
# ==========================================

if st.session_state.articles:

    st.subheader(
        f"Latest News: {st.session_state.topic}"
    )

    for article in st.session_state.articles:

        with st.expander(article["title"]):

            st.write(
                "Source:",
                article["source"]
            )

            st.write(
                "Published:",
                article["published"]
            )

            if article["url"]:

                st.markdown(
                    f"[Read Full Article]({article['url']})"
                )

else:

    st.info(
        "Please fetch news using the sidebar before "
        "asking questions."
    )


# ==========================================
# 15. ASK QUESTIONS ABOUT NEWS
# ==========================================

st.divider()

st.subheader("💬 Ask Questions About the News")

question = st.text_input(
    "Enter your question",
    placeholder="What are the latest AI updates?"
)

if st.button("Generate Answer"):

    if st.session_state.index is None:

        st.warning(
            "Please fetch news first."
        )

    elif not question.strip():

        st.warning(
            "Please enter your question."
        )

    else:

        try:

            with st.spinner(
                "Retrieving news and generating answer..."
            ):

                # Retrieve relevant chunks
                retrieved_chunks = retrieve_documents(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k=5
                )

                # Generate answer using Gemini
                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            # Display answer
            st.subheader("🤖 Gemini AI Answer")

            st.write(answer)

            # Display retrieved sources
            st.subheader("📚 Retrieved News Sources")

            for chunk in retrieved_chunks:

                with st.expander(chunk["title"]):

                    st.write(
                        "Source:",
                        chunk["source"]
                    )

                    st.write(
                        "Published:",
                        chunk["published"]
                    )

                    st.write(
                        f"Similarity Score: {chunk['score']:.3f}"
                    )

                    if chunk["url"]:

                        st.markdown(
                            f"[Read Original Article]({chunk['url']})"
                        )

        except Exception as error:

            st.error(
                f"Answer generation failed: {error}"
            )


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.caption(
    "Built using Python | Streamlit | NewsAPI | "
    "Sentence Transformers | FAISS | Google Gemini"
)
