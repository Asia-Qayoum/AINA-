import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM UI STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="AINA | Relational Persona Studio", layout="wide", initial_sidebar_state="expanded")

# Injecting the "Parchment & Linen" Theme & Urdu Typography
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #FDFBF7;
        color: #2A2A2A;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f4f1ea;
        border-right: 1px solid #7A8B7B;
    }
    
    /* Typography & Headers */
    h1, h2, h3 {
        color: #2A2A2A;
        font-family: 'Georgia', serif;
    }
    
    /* Cards with soft drop shadows */
    .aina-card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #e0dfd5;
        border-top: 4px solid #7A8B7B;
        margin-bottom: 20px;
    }
    
    /* Accent Buttons */
    .stButton>button {
        background-color: #7A8B7B;
        color: white;
        border: none;
        border-radius: 6px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #5c6a5d;
        color: white;
    }

    /* Allama Iqbal Poetry Block - RTL Nastaliq Formatting */
    .urdu-poetry {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Nastaleeq', 'Urdu Typesetting', 'Arial', serif;
        font-size: 32px;
        text-align: center;
        color: #5c6a5d;
        direction: rtl;
        line-height: 2.2;
        padding: 20px;
        margin-bottom: 30px;
        border-bottom: 1px solid #e0dfd5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. STATE MANAGEMENT & MODEL LOADING
# ---------------------------------------------------------
if 'baseline_texts' not in st.session_state:
    st.session_state['baseline_texts'] = ""

@st.cache_resource
def load_local_model():
    """Loads the assignment-required local embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------
# 3. HEADER & PHILOSOPHY
# ---------------------------------------------------------
st.markdown("""
<div class="urdu-poetry">
تو بچا بچا کے نہ رکھ اسے، ترا آئینہ ہے وہ آئینہ<br>
کہ شکستہ ہو تو عزیز تر ہے نگاہِ آئینہ ساز میں
</div>
""", unsafe_allow_html=True)

st.title("AINA: Relational Persona Studio")
st.markdown("*A privacy-first, clinical-level behavioral co-pilot.*")

# ---------------------------------------------------------
# 4. DUAL-TRACK SWITCHER (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.title("System Tracks")
mode = st.sidebar.radio(
    "Select Operating Mode:",
    ["Lab Assignment Mode", "Product Mode (Co-Pilot)"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data is processed instantly. No logs are retained on standard servers.")

# =========================================================
# TRACK A: LAB ASSIGNMENT MODE (Strictly Local + Matplotlib)
# =========================================================
if mode == "Lab Assignment Mode":
    st.header("Lab Analysis Dashboard")
    st.markdown("Mathematical vector mapping of raw text using **all-MiniLM-L6-v2**.")
    
    local_model = load_local_model()
    
    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("Text Source A (e.g., Mom)", "I am so worried about you, please call me.")
        text2 = st.text_area("Text Source B (e.g., Boss)", "Please submit the quarterly report by EOD.")
    with col2:
        text3 = st.text_area("Text Source C (e.g., Best Friend)", "Yo what time are we chilling tonight?")
        text4 = st.text_area("Live Text (Target Analysis)", "I can't deal with this right now, leave me alone.")

    if st.button("Run Vector Analysis"):
        texts = [text1, text2, text3, text4]
        labels = ["Source A", "Source B", "Source C", "Live Text"]
        
        # Extract Embeddings
        embeddings = local_model.encode(texts)
        
        st.markdown("### Visualization Suite")
        fig_col1, fig_col2 = st.columns(2)
        
        # 1. Cosine Similarity Heatmap
        with fig_col1:
            st.markdown("<div class='aina-card'>", unsafe_allow_html=True)
            st.subheader("1. Vector Proximity (Heatmap)")
            sim_matrix = cosine_similarity(embeddings)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(sim_matrix, annot=True, cmap="crest", xticklabels=labels, yticklabels=labels)
            fig.patch.set_facecolor('#FDFBF7')
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # 2. PCA 2D Scatter Plot
        with fig_col2:
            st.markdown("<div class='aina-card'>", unsafe_allow_html=True)
            st.subheader("2. Structural Archetypes (PCA Plot)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(pca_result[:, 0], pca_result[:, 1], color='#7A8B7B', s=100)
            for i, txt in enumerate(labels):
                ax2.annotate(txt, (pca_result[i, 0]+0.02, pca_result[i, 1]+0.02))
            fig2.patch.set_facecolor('#FDFBF7')
            ax2.set_facecolor('#FDFBF7')
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Bar Chart of Magnitude (Simplistic representation of vector intensity)
        st.markdown("<div class='aina-card'>", unsafe_allow_html=True)
        st.subheader("3. Linguistic Intensity (Vector Magnitude Bar Chart)")
        magnitudes = np.linalg.norm(embeddings, axis=1)
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        sns.barplot(x=labels, y=magnitudes, palette="dark:salmon_r")
        fig3.patch.set_facecolor('#FDFBF7')
        ax3.set_facecolor('#FDFBF7')
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TRACK B: PRODUCT MODE (WhatsApp GenAI Co-Pilot)
# =========================================================
elif mode == "Product Mode (Co-Pilot)":
    st.header("Relational Co-Pilot")
    
    # Needs API Key for Gemini
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Required for Clinical Mode")
    
    st.markdown("### The Calibration Desk (Input A)")
    with st.expander("Establish your Behavioral Baseline", expanded=True):
        st.write("Paste 2-3 paragraphs of standard conversations (e.g., Mom, Friend, Boss) so the engine can learn your default syntax.")
        baseline_input = st.text_area("Your Historical Chat Data", height=150, 
                                      value=st.session_state['baseline_texts'])
        if st.button("Save Baseline to Desk Memory"):
            st.session_state['baseline_texts'] = baseline_input
            st.success("Snapshot established in local memory.")

    st.markdown("### The Help Desk (Input B)")
    live_chat = st.text_area("Paste the active argument or 'stuck' conversation here:", height=150,
                             placeholder="E.g., [10:45 AM] Them: Why are you always ignoring me?\n[10:46 AM] You: I am just busy...")

    if st.button("Diagnose & Generate Strategy"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        elif not live_chat:
            st.warning("Please paste the live conversation to analyze.")
        else:
            with st.spinner("Analyzing psychological anchors and drafting response protocols..."):
                try:
                    # Initialize modern google-genai client
                    client = genai.Client(api_key=api_key)
                    
                    system_prompt = """
                    You are a clinical-level AI relationship strategist. 
                    Analyze the user's live conversation against their baseline communication style.
                    Provide a response strictly structured with the following markdown headers:
                    
                    ### 1. Psychological Diagnosis
                    (Provide 3 bullet points diagnosing the emotional states in the active chat - Respect, Fear, Love, or Confusion.)
                    
                    ### 2. Textual Evidence
                    (Quote exact words/phrases from the chat that prove the diagnosis.)
                    
                    ### 3. Strategic Script Panel
                    (Provide 3 variations of text messages the user can copy/paste right now to de-escalate or respond optimally.)
                    """
                    
                    prompt = f"""
                    User's Baseline Communication Habits:
                    {st.session_state['baseline_texts']}
                    
                    Active 'Stuck' Conversation to Analyze:
                    {live_chat}
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.4,
                        )
                    )
                    
                    # Output Layout
                    st.markdown("---")
                    st.subheader("Insight Dashboard")
                    
                    # Wrap output in styled card
                    st.markdown("<div class='aina-card'>", unsafe_allow_html=True)
                    st.markdown(response.text)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Engine Error: {str(e)}")
