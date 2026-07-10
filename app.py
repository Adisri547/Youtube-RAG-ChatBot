import streamlit as st
import re
import os

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# ---------------------------------------------------------
# CONFIG & UI STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬", layout="wide")

# Custom CSS for a beautiful, modern UI
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1 {
        font-weight: 800;
        background: linear-gradient(45deg, #FF0000, #FF4B4B, #FF8585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""))

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def extract_video_id(url_or_id: str) -> str:
    """Accepts a raw video ID or a full YouTube URL and returns the video ID."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"^([0-9A-Za-z_-]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id  # fallback


@st.cache_data(show_spinner=False)
def get_transcript(video_id: str, lang: str = "en"):
    ytapi = YouTubeTranscriptApi()
    transcript_list = ytapi.fetch(video_id, languages=[lang])
    # CHANGE chunk['text'] TO chunk.text BELOW:
    return " ".join(chunk.text for chunk in transcript_list)


@st.cache_resource(show_spinner=False)
def build_vectorstore(transcript: str, _video_id: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_model():
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-4-31B-it",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
    )
    return ChatHuggingFace(llm=llm)


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


PROMPT = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)

# ---------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png", width=60)
    st.title("Control Panel")
    st.write("Configure your YouTube RAG Engine here.")
    
    if not HF_TOKEN:
        st.warning("⚠️ No Hugging Face Token found in your secrets.")
        
    video_input = st.text_input(
        "YouTube Video URL or ID", 
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    st.divider()
    st.caption("Built with Streamlit, LangChain & HuggingFace Gemma 4")

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------
st.title("🎬 YouTube RAG Chatbot")
st.markdown("##### Chat seamlessly with any YouTube video context in real time.")

if video_input:
    video_id = extract_video_id(video_input)
    
    # Initialize session state for transcript/vectorstore reset when video changes
    if "current_video" not in st.session_state or st.session_state.current_video != video_id:
        st.session_state.current_video = video_id
        st.session_state.chat_history = []  # Clear previous chat

    # Layout Split: Left for Media, Right for Chat
    col1, col2 = st.columns([1.1, 1.4], gap="large")
    
    with col1:
        st.subheader("📺 Video Context")
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        
        try:
            with st.spinner("Decoding transcript..."):
                transcript = get_transcript(video_id)
            with st.spinner("Indexing into Vector DB..."):
                vector_store = build_vectorstore(transcript, video_id)
                
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            st.success(f"🤖 Context Ready: Loaded {len(transcript.split())} words successfully.")
            
        except TranscriptsDisabled:
            st.error("❌ Captions are disabled/unavailable for this video.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    with col2:
        st.subheader("💬 Chat Interface")
        
        # Display existing chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Handle User Inputs
        if question := st.chat_input("Ask something about the video..."):
            # Append User Question to History & Display
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
                
            # Process AI Response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing transcript..."):
                    model = get_model()
                    parallel_chain = RunnableParallel({
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                    })
                    chain = parallel_chain | PROMPT | model | StrOutputParser()
                    answer = chain.invoke(question)
                    st.write(answer)
                    
            # Append AI Response to History
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    # Warm welcome screen state
    st.info("👈 Please enter a YouTube video URL or ID in the sidebar panel to begin your RAG chat experience.")
    
    # Quick placeholders showing how it works
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. Feed Link")
        st.caption("Paste any YouTube URL into the sidebar menu.")
    with c2:
        st.markdown("### 2. Auto-Index")
        st.caption("The script fetches subtitles and constructs a FAISS semantic index vector map.")
    with c3:
        st.markdown("### 3. Chat Deeply")
        st.caption("Ask questions and get structured responses strictly mapped from video speech context.")