import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & ORGANIC CSS THEME
# ---------------------------------------------------------
st.set_page_config(page_title="AINA Studio", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

# Injecting the "Warm Organic Editorial" Webflow-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Nunito:wght@300;400;600&display=swap');

    /* Global Organic Theme */
    .stApp {
        background-color: #FAF8F5; /* Soft warm cream */
        color: #3E3B35;
        font-family: 'Nunito', sans-serif;
    }
    
    /* Elegant Serif Typography */
    h1, h2, h3, h4 {
        font-family: 'Lora', serif;
        color: #2C3625; /* Deep earthy green */
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Clean, unobtrusive sidebar */
    [data-testid="stSidebar"] {
        background-color: #F1EFE7;
        border-right: none;
    }

    /* Soft floating cards for inputs and outputs */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background: #FFFFFF;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.03);
        border: 1px solid #EAE6DB;
        margin-bottom: 20px;
    }

    /* Styling the buttons to look pill-shaped and organic */
    .stButton>button {
        background-color: #5B6B4D;
        color: #FAF8F5;
        border: none;
        border-radius: 50px;
        padding: 10px 24px;
        font-family: 'Nunito', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(91, 107, 77, 0.2);
    }
    .stButton>button:hover {
        background-color: #4A573F;
        transform: translateY(-2px);
    }

    /* Custom File Uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #F8F6F0;
        border-radius: 12px;
        border: 1px dashed #D4CCB6;
        padding: 10px;
    }

    /* Allama Iqbal Poetry - Refined Nastaliq */
    .urdu-poetry {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Nastaleeq', 'Urdu Typesetting', 'Arial', serif;
        font-size: 34px;
        text-align: center;
        color: #5B6B4D;
        direction: rtl;
        line-height: 2.4;
        padding: 30px;
        background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(241,239,231,0.5) 100%);
        border-radius: 20px;
        margin-bottom: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. STATE MANAGEMENT & MODEL LOADING
# ---------------------------------------------------------
if 'baseline_memory' not in st.session_state:
    st.session_state['baseline_memory'] = ""

@st.cache_resource
def load_local_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to read file or text
def get_text_input(uploaded_file, text_area_val):
    if uploaded_file is not None:
        return uploaded_file.getvalue().decode("utf-8")
    return text_area_val

# ---------------------------------------------------------
# 3. HEADER & PHILOSOPHY
# ---------------------------------------------------------
st.markdown("""
<div class="urdu-poetry">
تو بچا بچا کے نہ رکھ اسے، ترا آئینہ ہے وہ آئینہ<br>
کہ شکستہ ہو تو عزیز تر ہے نگاہِ آئینہ ساز میں
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>AINA Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #737067; margin-bottom: 40px;'>A clinical-level behavioral co-pilot.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ---------------------------------------------------------
mode = st.sidebar.radio(
    "Navigation",
    ["Lab Assignment Mode (Quiz)", "Relational Co-Pilot (Live)"]
)
st.sidebar.markdown("---")
st.sidebar.caption("🌿 Data is processed in real-time. No logs retained.")

# =========================================================
# TRACK A: LAB ASSIGNMENT MODE (Interactive Plotly Graphs)
# =========================================================
if mode == "Lab Assignment Mode (Quiz)":
    st.markdown("## Linguistic Vector Analysis")
    st.write("Evaluating raw semantic overlap using the local `all-MiniLM-L6-v2` architecture.")
    
    local_model = load_local_model()
    
    # Clean, organic input cards
    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("Text Source A", "I am so worried about you, please call me.")
        text2 = st.text_area("Text Source B", "Please submit the quarterly report by EOD.")
    with col2:
        text3 = st.text_area("Text Source C", "Yo what time are we chilling tonight?")
        text4 = st.text_area("Live Text Target", "I can't deal with this right now, leave me alone.")

    if st.button("Generate Visualizations"):
        texts = [text1, text2, text3, text4]
        labels = ["Source A", "Source B", "Source C", "Live Text"]
        
        with st.spinner("Mapping linguistic vectors..."):
            embeddings = local_model.encode(texts)
            sim_matrix = cosine_similarity(embeddings)
            
            st.markdown("---")
            st.markdown("### Interactive Analysis Dashboard")
            
            fig_col1, fig_col2 = st.columns(2)
            
            # 1. Plotly Heatmap (Much cleaner than Matplotlib)
            with fig_col1:
                st.markdown("#### Vector Proximity (Heatmap)")
                fig1 = px.imshow(sim_matrix, 
                                 x=labels, y=labels, 
                                 color_continuous_scale="Earth", # Warm organic color scale
                                 text_auto=".2f")
                fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig1, use_container_width=True)

            # 2. Plotly PCA Scatter
            with fig_col2:
                st.markdown("#### Structural Archetypes (2D PCA)")
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(embeddings)
                
                fig2 = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], text=labels, size=[10,10,10,10], 
                                  color_discrete_sequence=["#5B6B4D"])
                fig2.update_traces(textposition='top center', marker=dict(line=dict(width=2, color='DarkSlateGrey')))
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                   xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
                st.plotly_chart(fig2, use_container_width=True)

            # 3. Plotly Bar Chart
            st.markdown("#### Linguistic Intensity (Vector Magnitude)")
            magnitudes = np.linalg.norm(embeddings, axis=1)
            fig3 = px.bar(x=labels, y=magnitudes, color_discrete_sequence=["#D4A373"])
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="Source Texts", yaxis_title="Vector Magnitude")
            st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# TRACK B: PRODUCT MODE (WhatsApp Co-Pilot)
# =========================================================
elif mode == "Relational Co-Pilot (Live)":
    st.markdown("## Clinical Chat Strategist")
    
    api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password", help="Required for generative insights.")
    
    # --- INPUT A: BASELINE ---
    st.markdown("### 1. Establish Baseline Personality")
    st.caption("Upload a `.txt` file of historical chats OR paste raw text below to map baseline syntax.")
    
    base_file = st.file_uploader("Upload Baseline (.txt)", type=["txt"], key="base_file")
    base_text = st.text_area("Or Paste Baseline Text:", height=100, placeholder="Paste a few paragraphs from normal conversations...", key="base_text")
    
    if st.button("Save Baseline to Memory"):
        final_baseline = get_text_input(base_file, base_text)
        if final_baseline.strip():
            st.session_state['baseline_memory'] = final_baseline
            st.success("🌿 Baseline snapshot successfully secured in memory.")
        else:
            st.warning("Please provide baseline data via file or text.")

    st.markdown("---")
    
    # --- INPUT B: ACTIVE CONVERSATION ---
    st.markdown("### 2. The Active Situation")
    st.caption("Upload the current chat file or paste the text where communication broke down.")
    
    live_file = st.file_uploader("Upload Live Chat (.txt)", type=["txt"], key="live_file")
    live_text = st.text_area("Or Paste Active Argument:", height=100, placeholder="[10:45 AM] Them: Why are you doing this?...", key="live_text")
    
    # --- INPUT C: USER QUESTION ---
    st.markdown("### 3. Your Specific Query")
    user_query = st.text_input("What do you want to know about this interaction?", 
                               placeholder="e.g., Why are they acting so defensive? How can I de-escalate without apologizing?")

    if st.button("Generate Strategy"):
        final_live_chat = get_text_input(live_file, live_text)
        
        if not api_key:
            st.error("Please configure your API key in the sidebar.")
        elif not final_live_chat.strip():
            st.warning("Please provide the live chat data to analyze.")
        else:
            with st.spinner("Decoding relational dynamics..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    system_prompt = """
                    You are a clinical-level AI relationship strategist. 
                    Analyze the user's live conversation against their baseline communication style.
                    Address the user's specific query.
                    
                    Structure your response beautifully with markdown headers:
                    ### 💡 Psychological Diagnosis
                    (Diagnose emotional states: Respect, Fear, Love, Frustration, or Confusion)
                    
                    ### 🔍 Textual Evidence
                    (Quote exact phrases from the chat that prove the diagnosis)
                    
                    ### 📝 Strategic Script Options
                    (Provide 3 actionable, high-EQ text responses the user can copy/paste right now)
                    """
                    
                    prompt = f"""
                    User's Established Baseline Syntax:
                    {st.session_state.get('baseline_memory', 'No baseline provided. Analyze standalone.')}
                    
                    Active Conversation:
                    {final_live_chat}
                    
                    User's Specific Query:
                    {user_query if user_query else "Provide a general behavioral analysis and response strategy."}
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.4,
                        )
                    )
                    
                    # Output Rendered inside a stylized card layout natively via CSS
                    st.markdown("---")
                    st.markdown("## Strategy Dashboard")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Engine Error: {str(e)}")
