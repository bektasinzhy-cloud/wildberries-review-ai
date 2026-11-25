
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="REVIEW AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "wb_reviews.csv"
MODEL_PATH = BASE_DIR / "best_model.pkl"
METRICS_PATH = BASE_DIR / "model_comparison.csv"


# =========================================
# 2. CUSTOM DESIGN
# =========================================

st.markdown("""
<style>
.stApp {
    background: #090D18;
    color: #E8ECF8;
}

[data-testid="stSidebar"] {
    background: #101729;
    border-right: 1px solid #25314B;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #F2F5FF;
}

p, label {
    color: #B7C3D9;
}

.metric-card {
    background: linear-gradient(135deg, #151F36, #202B49);
    border: 1px solid #34466B;
    padding: 22px;
    border-radius: 18px;
    min-height: 125px;
}

.metric-label {
    color: #9EAFCD;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #FFFFFF;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.hero {
    background: linear-gradient(110deg, #182744, #282052);
    border: 1px solid #47558C;
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 25px;
}

.hero-subtitle {
    color: #B8C8EA;
    font-size: 15px;
}

div.stButton > button {
    background: linear-gradient(90deg, #6257E8, #268DDC);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 45px;
    font-weight: 700;
}

div.stButton > button:hover {
    border: 1px solid #9B9AFF;
    color: white;
}

.result-positive {
    background: #103D35;
    border: 1px solid #20A981;
    padding: 20px;
    border-radius: 16px;
}

.result-negative {
    background: #451F32;
    border: 1px solid #D65B82;
    padding: 20px;
    border-radius: 16px;
}

.small-note {
    color: #8192B2;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# 3. DATA LOADING
# =========================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    required = ["review_text", "rating", "sentiment"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            f"CSV файлында мына бағандар жоқ: {missing}"
        )

    data = data.dropna(subset=["review_text", "sentiment"]).copy()
    data["review_text"] = data["review_text"].astype(str)
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")

    data = data[data["review_text"].str.strip() != ""]
    data = data.dropna(subset=["rating"])

    return data.reset_index(drop=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
except Exception as e:
    st.error("Датасетті жүктеу кезінде қате пайда болды.")
    st.exception(e)
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("best_model.pkl моделін жүктеу мүмкін болмады.")
    st.exception(e)
    st.stop()


# =========================================
# 4. HELPER FUNCTIONS
# =========================================

def normalize_sentiment(value):
    """Әртүрлі sentiment белгілерін бір форматқа келтіреді."""

    value = str(value).strip().lower()

    positive_values = [
        "1", "positive", "pos", "позитивті",
        "положительный", "позитив"
    ]

    negative_values = [
        "0", "negative", "neg", "негативті",
        "отрицательный", "негатив"
    ]

    if value in positive_values:
        return "Positive"

    if value in negative_values:
        return "Negative"

    return value.title()


def prediction_label(prediction):
    """Модель болжамын Positive немесе Negative деп аударады."""

    value = str(prediction).strip().lower()

    if value in ["1", "1.0", "positive", "pos", "позитивті",
                 "положительный", "позитив"]:
        return "Positive"

    if value in ["0", "0.0", "negative", "neg", "негативті",
                 "отрицательный", "негатив"]:
        return "Negative"

    return str(prediction)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def predict_review(text):
    prediction = model.predict([text])[0]
    label = prediction_label(prediction)

    probability = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba([text])[0]
            classes = list(model.classes_)

            positive_index = next(
                (
                    i for i, c in enumerate(classes)
                    if prediction_label(c) == "Positive"
                ),
                None
            )

            if positive_index is not None:
                probability = float(probabilities[positive_index])

        except Exception:
            probability = None

    return label, probability


# =========================================
# 5. PREPARE DATA
# =========================================

df["sentiment_clean"] = df["sentiment"].apply(normalize_sentiment)

total_reviews = len(df)
positive_count = int((df["sentiment_clean"] == "Positive").sum())
negative_count = int((df["sentiment_clean"] == "Negative").sum())

average_rating = df["rating"].mean()
positive_percent = (
    positive_count / total_reviews * 100
    if total_reviews else 0
)


# =========================================
# 6. SIDEBAR NAVIGATION
# =========================================

with st.sidebar:
    st.markdown("# 🤖 REVIEW AI")
    st.caption("INTELLIGENT REVIEW ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📚 Reviews Explorer",
            "🧠 Sentiment Analyzer",
            "⚙️ Model Laboratory"
        ]
    )

    st.divider()

    st.markdown("**PROJECT INFORMATION**")
    st.caption("Wildberries Reviews")
    st.caption("NLP · Machine Learning")
    st.caption("TF-IDF · Sentiment Analysis")

    st.divider()
    st.caption(f"Dataset: {total_reviews:,} reviews")


# =========================================
# 7. HEADER
# =========================================

st.markdown("""
<div class="hero">
    <h1>REVIEW AI</h1>
    <div class="hero-subtitle">
        Intelligent Customer Feedback Analysis Platform
    </div>
    <p>
        Wildberries пікірлерін машиналық оқыту және
        табиғи тілді өңдеу арқылы интеллектуалды талдау.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================
# PAGE 1: DASHBOARD
# =========================================

if page == "🏠 Dashboard":

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("TOTAL REVIEWS", f"{total_reviews:,}")

    with col2:
        metric_card("POSITIVE REVIEWS", f"{positive_count:,}")

    with col3:
        metric_card("NEGATIVE REVIEWS", f"{negative_count:,}")

    with col4:
        metric_card(
            "AVERAGE RATING",
            f"{average_rating:.2f} / 5"
            if not pd.isna(average_rating)
            else "N/A"
        )

    st.markdown("### Sentiment Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [positive_count, negative_count]
        })

        fig = px.pie(
            sentiment_data,
            names="Sentiment",
            values="Count",
            hole=0.55,
            title="Positive vs Negative"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        rating_data = (
            df["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_data.columns = ["Rating", "Count"]

        fig = px.bar(
            rating_data,
            x="Rating",
            y="Count",
            title="Rating Distribution",
            text="Count"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Dataset Reviews")

    recent = df[["review_text", "rating", "sentiment_clean"]].head(10)

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Статистика бастапқы CSV датасетіндегі барлық жарамды пікірлерден есептеледі."
    )


# =========================================
# PAGE 2: REVIEWS EXPLORER
# =========================================

elif page == "📚 Reviews Explorer":

    st.subheader("📚 All Wildberries Reviews")

    search_text = st.text_input(
        "🔎 Search reviews",
        placeholder="Пікір мәтінінен сөз ізде..."
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_sentiment = st.selectbox(
            "Sentiment",
            ["All", "Positive", "Negative"]
        )

    with col2:
        rating_values = sorted(df["rating"].dropna().unique())
        selected_ratings = st.multiselect(
            "Rating",
            rating_values,
            default=rating_values
        )

    filtered = df.copy()

    if search_text:
        filtered = filtered[
            filtered["review_text"].str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        ]

    if selected_sentiment != "All":
        filtered = filtered[
            filtered["sentiment_clean"] == selected_sentiment
        ]

    filtered = filtered[
        filtered["rating"].isin(selected_ratings)
    ]

    st.markdown(f"**Табылған пікір саны: {len(filtered):,}**")

    page_size = 50
    total_pages = max(1, int(np.ceil(len(filtered) / page_size)))

    selected_page = st.number_input(
        "Бет нөмірі",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

    start = (selected_page - 1) * page_size
    end = start + page_size

    display_data = filtered[
        ["review_text", "rating", "sentiment_clean"]
    ].iloc[start:end].copy()

    display_data.columns = ["Review", "Rating", "Sentiment"]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        height=550
    )

    st.caption(
        f"Бет {selected_page} / {total_pages}. Бір бетте {page_size} пікірге дейін."
    )

    csv = filtered.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ Filtered reviews жүктеу",
        data=csv,
        file_name="filtered_reviews.csv",
        mime="text/csv"
    )


# =========================================
# PAGE 3: SENTIMENT ANALYZER
# =========================================

elif page == "🧠 Sentiment Analyzer":

    st.subheader("🧠 AI Sentiment Analyzer")

    st.write(
        "Жаңа пікір енгізіңіз. AI моделі оның тональдігін анықтайды."
    )

    review_text = st.text_area(
        "Review text",
        height=180,
        placeholder="Товар очень хороший, качество отличное..."
    )

    if st.button("🚀 Analyze Review", use_container_width=True):

        if not review_text.strip():
            st.warning("Алдымен пікір мәтінін енгізіңіз.")

        else:
            try:
                label, probability = predict_review(review_text.strip())

                if label == "Positive":
                    st.markdown(
                        """
                        <div class="result-positive">
                            <h2>😊 POSITIVE REVIEW</h2>
                            <p>Модель пікірді позитивті деп анықтады.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif label == "Negative":
                    st.markdown(
                        """
                        <div class="result-negative">
                            <h2>☹️ NEGATIVE REVIEW</h2>
                            <p>Модель пікірді негативті деп анықтады.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.info(f"Модель нәтижесі: {label}")

                if probability is not None:
                    st.metric(
                        "Positive class probability",
                        f"{probability:.1%}"
                    )

                    st.progress(probability)

                else:
                    st.caption(
                        "Бұл модель predict_proba() арқылы ықтималдық бермейді."
                    )

            except Exception as e:
                st.error("Пікірді талдау кезінде қате пайда болды.")
                st.exception(e)


# =========================================
# PAGE 4: MODEL LABORATORY
# =========================================

elif page == "⚙️ Model Laboratory":

    st.subheader("⚙️ Machine Learning Model Laboratory")

    st.markdown("### Current model")

    model_name = type(model).__name__

    if hasattr(model, "named_steps"):
        st.success(
            f"Pipeline model: {model_name}"
        )

        st.write("Pipeline components:")

        for step_name, step_object in model.named_steps.items():
            st.write(
                f"- **{step_name}**: {type(step_object).__name__}"
            )
    else:
        st.info(f"Model: {model_name}")

    st.markdown("### Model comparison")

    if METRICS_PATH.exists():

        try:
            metrics = pd.read_csv(METRICS_PATH)

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True
            )

            if "Model" in metrics.columns:

                metric_columns = [
                    col for col in [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1-score",
                        "ROC-AUC"
                    ]
                    if col in metrics.columns
                ]

                if metric_columns:
                    chart_metrics = metrics.set_index("Model")[
                        metric_columns
                    ]

                    st.bar_chart(chart_metrics)

        except Exception as e:
            st.error("Model comparison файлын оқу мүмкін болмады.")
            st.exception(e)

    else:
        st.warning(
            "model_comparison.csv файлы табылмады."
        )

    st.markdown("### Dataset information")

    st.write(f"Dataset rows: {len(df):,}")
    st.write(f"Dataset columns: {len(df.columns)}")
    st.write(f"Model file: {MODEL_PATH.name}")


# =========================================
# FOOTER
# =========================================

st.divider()

st.markdown(
    """
    <p style="text-align:center;color:#7183A5;font-size:12px">
        REVIEW AI · WILDBERRIES REVIEW INTELLIGENCE · NLP & MACHINE LEARNING
    </p>
    """,
    unsafe_allow_html=True
)
