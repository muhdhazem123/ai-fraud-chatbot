import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import google.generativeai as genai

# =====================================================
# 🔐 STEP 1: PASTE YOUR GEMINI API KEY HERE
# =====================================================
API_KEY = "AIzaSyD4BPhCJL-aZBbYzNcCqKpiy74WQyzBW3s"

genai.configure(api_key=API_KEY)

# Auto select working Gemini model
def load_ai():
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                return genai.GenerativeModel(m.name)
    except:
        return None

ai_model = load_ai()

# =====================================================
# 🧠 STEP 2: LOAD ML MODEL
# =====================================================
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =====================================================
# ⚙️ STEP 3: APP CONFIG
# =====================================================
st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# =====================================================
# 🎨 STEP 4: PREMIUM CARD STYLE
# =====================================================
st.markdown("""
<style>
.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 🧠 SESSION STATE (REAL-TIME DATA)
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# 📊 KPI CALCULATIONS (LIVE)
# =====================================================
total_txn = len(st.session_state.history)

fraud_count = sum(1 for h in st.session_state.history if h["Prediction"] == 1)

safe_count = total_txn - fraud_count

accuracy = (safe_count / total_txn * 100) if total_txn > 0 else 0

amount_protected = sum(
    h["Amount"] for h in st.session_state.history if h["Prediction"] == 0
)

# =====================================================
# 🏦 HEADER
# =====================================================
st.title("🏦 AI Fraud Detection Dashboard")
st.caption("Real-time Fraud Monitoring System")

# =====================================================
# 💳 KPI CARDS (REAL TIME)
# =====================================================
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f'<div class="card"><h2>{total_txn}</h2>Total Transactions</div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card"><h2>{fraud_count}</h2>Frauds Detected</div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card"><h2>₹{amount_protected:.2f}</h2>Amount Protected</div>', unsafe_allow_html=True)
col4.markdown(f'<div class="card"><h2>{accuracy:.2f}%</h2>Accuracy</div>', unsafe_allow_html=True)

# =====================================================
# 🔍 TRANSACTION INPUT
# =====================================================
st.markdown("## 🔍 Analyze Transaction")

colA, colB, colC = st.columns(3)

amount = colA.number_input("Amount", 0.0)
location = colB.selectbox("Location Changed?", [0, 1])
frequency = colC.slider("Transactions (24 hrs)", 1, 50)

# =====================================================
# 🚀 ANALYZE BUTTON
# =====================================================
if st.button("Analyze Transaction"):

    input_data = np.array([[amount, location, frequency]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Safe ({(1-prob)*100:.2f}%)")

    # Save transaction (REAL-TIME UPDATE)
    st.session_state.history.append({
        "Amount": amount,
        "Location": location,
        "Frequency": frequency,
        "Prediction": prediction,
        "Probability": prob
    })

# =====================================================
# 📊 PLOTLY ANALYTICS
# =====================================================
st.markdown("## 📊 Analytics")

if st.session_state.history:

    df = pd.DataFrame(st.session_state.history)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Prediction",
                           color="Prediction",
                           title="Fraud vs Safe",
                           template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.line(df, y="Probability",
                       markers=True,
                       title="Fraud Probability Trend",
                       template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df, use_container_width=True)

else:
    st.info("No transactions yet")

# =====================================================
# 🤖 AI ASSISTANT (CONTEXT-AWARE)
# =====================================================
st.markdown("## 🤖 AI Assistant")

query = st.text_input("Ask about fraud trends")

if st.button("Ask AI"):

    if ai_model and st.session_state.history:

        context = f"""
        Transactions Data:
        {st.session_state.history}
        """

        try:
            response = ai_model.generate_content(context + query)
            st.write(response.text)
        except Exception as e:
            st.error(f"AI Error: {e}")

    else:
        st.warning("No data or AI unavailable")