import streamlit as st
import pandas as pd
from Predict_V2 import predict
from Explain_V2 import explain_article

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="News Popularity Intelligence System",
    page_icon="📰",
    layout="wide"
)

# ================= CUSTOM STYLING =================
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight:600;
}
.score-font {
    font-size:32px !important;
    font-weight:700;
    color:#FF4B4B;
}
</style>
""", unsafe_allow_html=True)

st.title("📰 News Popularity Intelligence System")
st.markdown("Transformer-Based AI for Editorial Ranking & Media Intelligence")

# ================= SIDEBAR =================
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "News Intelligence", "Comparison Demo", "Model Reasoning"]
)

# =====================================================
# ===================== HOME PAGE =====================
# =====================================================
if menu == "Home":

    st.header("📌 Problem Overview")

    st.write("""
    Real-world popularity signals such as clicks, shares, and impressions 
    are unavailable at publishing time.

    Therefore, popularity is treated as a latent variable and inferred 
    using Transformer-based semantic intelligence.
    """)

    st.subheader("⚙️ System Architecture")

    st.code("""
    Title + Description
            ↓
    DistilBERT Embeddings
            ↓
    Clustering-Based Popularity Groups
            ↓
    Signal Fusion Scoring Engine
            ↓
    Ranking + Explanation Layer
    """)

# =====================================================
# ================ NEWS INTELLIGENCE ==================
# =====================================================
elif menu == "News Intelligence":

    st.header("🔮 Predict Article Popularity")

    title = st.text_input("Enter News Title")
    description = st.text_area("Enter News Description")

    if st.button("Analyze Article"):

        if not title.strip() or not description.strip():
            st.warning("Please provide both title and description.")
        else:
            text = title + " " + description

            cluster = predict(text)
            explanation = explain_article(text)

            # ===== Improved Intelligent Scoring =====
            base_score = 1 - (cluster / 5)

            signal_boost = (
                explanation["sentiment_score"] * 0.2 +
                explanation["urgency_terms"] * 0.1 +
                (explanation["unique_words"] / 50) * 0.2
            )

            popularity_score = round(min(base_score + signal_boost, 1.0), 2)

            # ===== Display Metrics =====
            col1, col2 = st.columns(2)
            col1.metric("⭐ Popularity Score", popularity_score)
            col2.metric("📊 Predicted Cluster", cluster)

            # ===== Visual Chart =====
            metrics_df = pd.DataFrame({
                "Metric": ["Sentiment", "Urgency Terms", "Lexical Diversity", "Text Length"],
                "Value": [
                    explanation["sentiment_score"],
                    explanation["urgency_terms"],
                    explanation["unique_words"] / 50,
                    explanation["text_length"] / 200
                ]
            })

            st.subheader("📊 Signal Strength Visualization")
            st.bar_chart(metrics_df.set_index("Metric"))

            st.subheader("🧠 Model Explanation")
            st.success(explanation["reasoning"])

# =====================================================
# ================= COMPARISON DEMO ===================
# =====================================================
elif menu == "Comparison Demo":

    st.header("📊 Compare Two Articles")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Article 1")
        title1 = st.text_input("Title 1")
        desc1 = st.text_area("Description 1")

    with colB:
        st.subheader("Article 2")
        title2 = st.text_input("Title 2")
        desc2 = st.text_area("Description 2")

    if st.button("Compare Articles"):

        if not all([title1, desc1, title2, desc2]):
            st.warning("Please fill all fields.")
        else:
            text1 = title1 + " " + desc1
            text2 = title2 + " " + desc2

            cluster1 = predict(text1)
            cluster2 = predict(text2)

            score1 = round(1 - (cluster1 / 5), 2)
            score2 = round(1 - (cluster2 / 5), 2)

            col1, col2 = st.columns(2)
            col1.metric("Article 1 Score", score1)
            col2.metric("Article 2 Score", score2)

            if score1 > score2:
                st.success("Article 1 has higher predicted attention potential.")
            elif score2 > score1:
                st.success("Article 2 has higher predicted attention potential.")
            else:
                st.info("Both articles show similar attention potential.")

# =====================================================
# ================= MODEL REASONING ===================
# =====================================================
elif menu == "Model Reasoning":

    st.header("🧠 How the Model Infers Popularity")

    st.subheader("1️⃣ Latent Popularity Concept")

    st.write("""
    Popularity is treated as a latent variable rather than a supervised label.
    Since real engagement metrics (clicks, shares) are unavailable,
    the system infers attention potential using semantic intelligence.
    """)

    st.subheader("2️⃣ Transformer Representation Learning")

    st.write("""
    • Title and description are concatenated  
    • DistilBERT extracts contextual embeddings  
    • Articles are mapped into high-dimensional semantic space  
    • Clustering identifies attention-driven groups  
    """)

    st.subheader("3️⃣ Popularity Scoring Logic")

    st.write("""
    The final popularity score is computed using a fusion of:
    
    • Embedding-based cluster strength  
    • Emotional intensity  
    • Urgency-related keywords  
    • Lexical diversity  
    """)

    st.code("""
Base Score = 1 - (Cluster_ID / Total_Clusters)

Signal Boost =
    (Sentiment * 0.2)
  + (Urgency_Terms * 0.1)
  + (Lexical_Diversity * 0.2)

Final Popularity Score =
    min(Base Score + Signal Boost, 1.0)
    """)

    st.write("""
    • The base score reflects semantic cluster positioning  
    • Signal boost enhances articles with emotional and urgent tone  
    • Score is normalized between 0 and 1  
    """)

    st.subheader("4️⃣ Why This Works Without Labels")

    st.write("""
    Even without real engagement labels, articles with:
    
    • Urgency
    • Emotional intensity
    • Clear narrative structure
    
    tend to cluster together in embedding space.
    
    These clusters are interpreted as high-attention groups,
    allowing the system to rank articles meaningfully.
    """)

    st.info("""
    If real engagement data becomes available, the Transformer
    can be fine-tuned with a regression head for direct
    supervised popularity prediction.
    """)
