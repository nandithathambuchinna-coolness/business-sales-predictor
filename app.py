
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# BUSINESS PREDICTOR | SALES INTELLIGENCE DASHBOARD
# Simulated educational dataset - 90 records
# ============================================================

st.set_page_config(
    page_title="Business Predictor | Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- THEME --------------------

BG = "#F3F2EC"
PANEL = "#FFFFFF"
INK = "#102522"
MUTED = "#435650"
PLUM = "#172B2A"
BURGUNDY = "#087E73"
GOLD = "#B8D94A"
GOLD_LIGHT = "#D8EE8B"
EMERALD = "#087E73"
TEAL = "#55B8A9"
LAVENDER = "#687F78"
GRID = "#E1E7E2"

FEATURES = ["Ad_Spend", "Price", "Economy", "Competition"]
REQUIRED = FEATURES + ["Sales"]


# -------------------- CUSTOM CSS --------------------

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: {INK};
    }}

    .stApp {{
        background:
            radial-gradient(ellipse at 90% 0%,
                rgba(8,126,115,0.09), transparent 32%),
            linear-gradient(180deg, #F6F5F0 0%, {BG} 100%);
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}

    /* Dark, readable headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #102522 !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }}

    h1, .serif {{
        font-family: 'Playfair Display', Georgia, serif !important;
    }}

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {{
        color: #102522 !important;
        font-weight: 800 !important;
    }}

    p, li {{
        color: #263B35;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #172B2A 0%, #203735 100%);
        border-right: 1px solid rgba(184,217,74,0.35);
    }}

    [data-testid="stSidebar"] * {{
        color: #F3F2EC;
    }}

    [data-testid="stSidebar"] label {{
        color: #F3F2EC !important;
        font-weight: 600 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #D5E0DA;
    }}

    /* Brand */
    .brand-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.2rem;
    }}

    .brand-mark {{
        width: 48px;
        height: 48px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: {PLUM};
        color: {GOLD};
        border: 1px solid {GOLD};
        font-size: 23px;
        font-weight: 800;
    }}

    .brand-name {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: {INK};
    }}

    .brand-caption {{
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .68rem;
        color: {MUTED};
        margin-top: 3px;
    }}

    /* Hero */
    .hero {{
        background: linear-gradient(120deg, #FFFFFF 0%, #EAF4EF 100%);
        color: {INK};
        padding: 2.2rem;
        border-radius: 24px;
        margin: .4rem 0 1.25rem 0;
        box-shadow: 0 15px 40px rgba(16,37,34,.10);
        border: 1px solid #C9D8CF;
    }}

    .eyebrow {{
        text-transform: uppercase;
        letter-spacing: .19em;
        font-size: .75rem;
        font-weight: 800;
        color: {BURGUNDY};
        margin-bottom: .65rem;
    }}

    .hero h1 {{
        color: #102522 !important;
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.1;
        margin: .2rem 0 .8rem 0;
    }}

    .hero p {{
        color: #435650;
        font-size: 1rem;
        max-width: 780px;
        line-height: 1.7;
    }}

    .hero-tag {{
        display: inline-block;
        margin-top: .8rem;
        padding: 8px 12px;
        border: 1px solid #B7D9CF;
        border-radius: 999px;
        color: {BURGUNDY};
        font-size: .75rem;
        font-weight: 800;
        background: #F3FAF6;
    }}

    /* Sections */
    .section-head {{
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 2rem 0 .35rem 0;
    }}

    .section-number {{
        min-width: 31px;
        height: 31px;
        border-radius: 9px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #DDECE6;
        color: #075E55;
        font-weight: 800;
        font-size: .8rem;
    }}

    .section-subtitle {{
        color: #465852;
        font-size: .9rem;
        margin-bottom: 1rem;
    }}

    .small-note {{
        color: #435650;
        font-size: .8rem;
        line-height: 1.6;
    }}

    /* Metric cards */
    .metric-card {{
        background: {PANEL};
        border: 1px solid #C9D8CF;
        border-radius: 17px;
        padding: 19px;
        min-height: 125px;
        box-shadow: 0 5px 18px rgba(16,37,34,.05);
    }}

    .metric-label {{
        color: #435650;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .10em;
        font-weight: 800;
        margin-bottom: 11px;
    }}

    .metric-value {{
        font-family: 'Playfair Display', Georgia, serif;
        color: #102522;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }}

    .metric-note {{
        color: #435650;
        font-size: .78rem;
        margin-top: 9px;
    }}

    /* Insight cards */
    .insight-card {{
        background: #FFFFFF;
        border: 1px solid #D3DFD8;
        border-left: 5px solid {BURGUNDY};
        border-radius: 13px;
        padding: 15px;
        margin: .45rem 0;
    }}

    .insight-title {{
        color: #102522;
        font-weight: 800;
        margin-bottom: 5px;
    }}

    .insight-body {{
        color: #435650;
        font-size: .88rem;
        line-height: 1.6;
    }}

    /* Prediction */
    .prediction-card {{
        background: linear-gradient(135deg, #F8FAF6, #E8F3EE);
        border: 1px solid #BFDCD1;
        border-radius: 20px;
        padding: 25px;
    }}

    .prediction-label {{
        color: #075E55;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .75rem;
        font-weight: 800;
    }}

    .prediction-value {{
        font-family: 'Playfair Display', Georgia, serif;
        color: #075E55;
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        line-height: 1.2;
        margin: .5rem 0;
    }}

    .prediction-note {{
        color: #435650;
        font-size: .88rem;
        line-height: 1.6;
    }}

    /* Graphs */
    div[data-testid="stPlotlyChart"] {{
        background: #FFFFFF;
        border: 1px solid #C9D8CF;
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 5px 18px rgba(16,37,34,.05);
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid #C2D0C8;
        border-radius: 12px;
        overflow: hidden;
    }}

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 11px;
        border: 1px solid {BURGUNDY};
        background: {BURGUNDY};
        color: #FFFFFF;
        font-weight: 800;
        padding: .65rem 1rem;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: {PLUM};
        background: {PLUM};
        color: #FFFFFF;
    }}

    hr {{
        border-color: #D3DFD8;
    }}

    footer {{
        visibility: hidden;
    }}

    #MainMenu {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- LOAD DATA --------------------

@st.cache_data
def load_data():
    csv_path = Path(__file__).resolve().parent / "data.csv"
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("At least 10 valid records are required.")

    return df


try:
    data = load_data()
except Exception as exc:
    st.error(f"Unable to load data.csv: {exc}")
    st.info(
        "Place data.csv in the same folder as app.py. "
        "Required columns: " + ", ".join(REQUIRED)
    )
    st.stop()


# -------------------- TRAIN MODEL --------------------

X = data[FEATURES]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

correlation = data.corr(numeric_only=True)

# Coefficients in original feature units:
# useful for explaining the model in the same way as project.py.
original_model = LinearRegression()
original_model.fit(X_train, y_train)
original_coefficients = original_model.coef_

standardised_coefficients = (
    model.named_steps["linearregression"].coef_
)


# -------------------- SHARED CHART STYLE --------------------

def style_plot(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            family="DM Sans, sans-serif",
            color=INK,
            size=12,
        ),
        title_font=dict(
            family="Playfair Display, Georgia, serif",
            size=20,
            color="#102522",
        ),
        margin=dict(l=35, r=30, t=65, b=45),
        hoverlabel=dict(
            bgcolor=PLUM,
            font_color="#FFFFFF",
            bordercolor=GOLD,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=0,
            font=dict(color=INK),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor="#AEBDB5",
        title_font=dict(color=INK, size=13),
        tickfont=dict(color="#263B35", size=11),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor="#AEBDB5",
        title_font=dict(color=INK, size=13),
        tickfont=dict(color="#263B35", size=11),
    )

    return fig


def section(number, heading, subtitle):
    st.markdown(
        f"""
        <div class="section-head">
            <span class="section-number">{number}</span>
            <h2 style="margin:0;color:#102522;">
                {heading}
            </h2>
        </div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def relationship_label(value):
    if value > 0.7:
        return "Strong positive relationship"
    elif value > 0.3:
        return "Moderate positive relationship"
    elif value < -0.7:
        return "Strong negative relationship"
    elif value < -0.3:
        return "Moderate negative relationship"
    return "Weak or no linear relationship"


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 18px 0;">
            <div style="font-size:1.45rem;font-weight:800;
                        color:#F3F2EC;">
                ◈ Business Predictor
            </div>
            <div style="font-size:.68rem;letter-spacing:.15em;
                        color:#D8EE8B;text-transform:uppercase;">
                Sales Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Prediction studio")
    st.write(
        "Adjust the business factors below, then generate a sales estimate."
    )

    input_values = {}

    for feature in FEATURES:
        low = float(data[feature].min())
        high = float(data[feature].max())
        midpoint = float(data[feature].median())

        labels = {
            "Ad_Spend": "Advertising spend",
            "Price": "Product price",
            "Economy": "Economic indicator",
            "Competition": "Competition index",
        }

        steps = {
            "Ad_Spend": 1.0,
            "Price": 0.1,
            "Economy": 1.0,
            "Competition": 1.0,
        }

        input_values[feature] = st.slider(
            labels[feature],
            min_value=low,
            max_value=high,
            value=min(max(midpoint, low), high),
            step=steps[feature],
            help=f"Observed data range: {low:g} to {high:g}",
        )

    predict_clicked = st.button(
        "✦ Generate prediction",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Model notes")
    st.write("Linear Regression")
    st.write("Standardised input features")
    st.write("80/20 train-test split")
    st.write("Random seed: 42")

    st.markdown("---")
    st.caption(
        f"Educational demo · {len(data)} simulated records. "
        "Not real business data."
    )


# -------------------- HERO --------------------

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">BP</div>
        <div>
            <div class="brand-name">Business Predictor</div>
            <div class="brand-caption">Sales Intelligence Dashboard</div>
        </div>
    </div>

    <div class="hero">
        <div class="eyebrow">
            Decision intelligence · Edition 01
        </div>
        <h1>Find the signal<br>behind your sales.</h1>
        <p>
            Explore how advertising spend, pricing, economic conditions
            and competition relate to sales. Investigate the data,
            evaluate a machine-learning model and test your own scenario.
        </p>
        <span class="hero-tag">
            LINEAR REGRESSION · INTERACTIVE ANALYTICS
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="small-note">
        Educational research demo. All records are simulated.
        Model outputs are estimates, not guarantees or causal claims.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- 01: AT A GLANCE --------------------

section(
    "01",
    "At a glance",
    "A quick summary of the dataset and model performance.",
)

kpis = [
    ("DATA RECORDS", f"{len(data):,}", "Simulated observations"),
    ("TEST R²", f"{r2:.3f}", "Held-out test performance"),
    ("TEST MAE", f"{mae:.2f}", "Average error in sales units"),
    ("MODEL", "Linear", "Standardised regression"),
]

cols = st.columns(4)

for col, (label, value, note) in zip(cols, kpis):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- 02: PREDICTION STUDIO --------------------

section(
    "02",
    "Prediction studio",
    "Choose values in the sidebar and generate a sales estimate.",
)

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if predict_clicked:
    scenario = pd.DataFrame([input_values], columns=FEATURES)
    prediction = float(model.predict(scenario)[0])

    st.session_state.last_prediction = {
        "values": input_values.copy(),
        "prediction": prediction,
    }

pred_cols = st.columns([1.2, 1])

with pred_cols[0]:
    saved = st.session_state.last_prediction

    prediction_text = (
        "—" if saved is None else f"{saved['prediction']:,.2f}"
    )

    prediction_note = (
        "Adjust the inputs in the sidebar, then select Generate prediction."
        if saved is None
        else
        "Estimated sales for the most recently submitted scenario. "
        "This is a model estimate, not a guarantee."
    )

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Scenario estimate</div>
            <div class="prediction-value">{prediction_text}</div>
            <div class="prediction-note">{prediction_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with pred_cols[1]:
    st.markdown("#### Current scenario")

    current_df = pd.DataFrame([
        {
            "Business factor": key.replace("_", " "),
            "Selected value": value,
        }
        for key, value in input_values.items()
    ])

    st.dataframe(
        current_df,
        hide_index=True,
        use_container_width=True,
        height=205,
    )

    if saved is not None:
        report = [
            "BUSINESS PREDICTOR | SALES PREDICTION REPORT",
            "=" * 45,
            "",
            f"Predicted sales: {saved['prediction']:.2f} units",
            "",
            "Scenario inputs:",
        ]

        for key, value in saved["values"].items():
            report.append(f"- {key}: {value}")

        report.extend([
            "",
            "Model: Standardised Linear Regression",
            f"Dataset size: {len(data)} simulated records",
            f"Test MAE: {mae:.3f}",
            f"Test R²: {r2:.3f}",
            "",
            "Limitations: Simulated data; exploratory estimates only.",
        ])

        st.download_button(
            "↓ Download scenario report",
            data="\n".join(report),
            file_name="Business_Prediction_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )


# -------------------- 03: CORRELATION AND INSIGHTS --------------------

section(
    "03",
    "What moves with sales?",
    "Correlation measures linear association, not causation.",
)

heat_col, insight_col = st.columns([1.15, 1])

with heat_col:
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale=[
            [0.0, "#A83E50"],
            [0.25, "#E9B8A9"],
            [0.5, "#F3F2EC"],
            [0.75, "#9AD5C0"],
            [1.0, "#087E73"],
        ],
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation matrix",
        labels={"color": "Pearson r"},
    )

    fig.update_coloraxes(colorbar_title="Pearson r")
    style_plot(fig, 450)
    st.plotly_chart(fig, use_container_width=True)

with insight_col:
    st.markdown("#### Relationship highlights")

    for feature in FEATURES:
        value = float(correlation.loc[feature, "Sales"])
        description = relationship_label(value)

        accent = (
            EMERALD if value > 0.3
            else BURGUNDY if value < -0.3
            else LAVENDER
        )

        st.markdown(
            f"""
            <div class="insight-card"
                 style="border-left-color:{accent};">
                <div style="display:flex;justify-content:space-between;
                            gap:10px;align-items:center;">
                    <div class="insight-title">
                        {feature.replace('_', ' ')}
                    </div>
                    <div style="font-weight:800;color:{accent};">
                        {value:+.2f}
                    </div>
                </div>
                <div class="insight-body">
                    {description} with sales in this dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="small-note">
            Relationships can reflect how the simulated data was created
            and may be influenced by relationships among input features.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- 04: INTERACTIVE EXPLORATION --------------------

section(
    "04",
    "Explore the data",
    "Choose a factor to explore its relationship with sales.",
)

explore_left, explore_right = st.columns([1, 2.1])

with explore_left:
    chosen_feature = st.selectbox(
        "Choose a business factor",
        FEATURES,
        format_func=lambda x: x.replace("_", " "),
    )

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-title">How to read this chart</div>
            <div class="insight-body">
                Each point represents a simulated observation.
                The trend line summarises a linear pattern, not causation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with explore_right:
    fig = px.scatter(
        data,
        x=chosen_feature,
        y="Sales",
        color="Competition",
        color_continuous_scale=["#D8EE8B", "#55B8A9", "#172B2A"],
        hover_data=FEATURES,
        trendline="ols",
        labels={
            chosen_feature: chosen_feature.replace("_", " "),
            "Sales": "Sales units",
            "Competition": "Competition index",
        },
        title=f"{chosen_feature.replace('_', ' ')} vs Sales",
    )

    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.82,
            line=dict(width=1, color=PANEL),
        )
    )

    style_plot(fig, 430)
    st.plotly_chart(fig, use_container_width=True)


# -------------------- 05: ALL FOUR SCATTER PLOTS --------------------

section(
    "05",
    "Business data visualisations",
    "All four factor-versus-sales graphs, plus sales distribution "
    "and model performance.",
)

st.markdown("### Individual business factor graphs")

for start in range(0, len(FEATURES), 2):
    pair = FEATURES[start:start + 2]
    graph_cols = st.columns(2)

    for col, feature in zip(graph_cols, pair):
        with col:
            fig = px.scatter(
                data,
                x=feature,
                y="Sales",
                trendline="ols",
                hover_data=FEATURES,
                labels={
                    feature: feature.replace("_", " "),
                    "Sales": "Sales units",
                },
                title=f"{feature.replace('_', ' ')} vs Sales",
            )

            fig.update_traces(
                marker=dict(
                    size=9,
                    color=BURGUNDY,
                    opacity=0.78,
                    line=dict(width=1, color=PANEL),
                )
            )

            style_plot(fig, 380)
            st.plotly_chart(fig, use_container_width=True)


# -------------------- SALES DISTRIBUTION --------------------

st.markdown("### Sales distribution")

fig = px.histogram(
    data,
    x="Sales",
    nbins=15,
    title="Distribution of Sales",
    labels={
        "Sales": "Sales units",
        "count": "Number of records",
    },
    color_discrete_sequence=[EMERALD],
)

fig.update_traces(
    marker_line_color="#FFFFFF",
    marker_line_width=1,
)

style_plot(fig, 380)
st.plotly_chart(fig, use_container_width=True)


# -------------------- ACTUAL VS PREDICTED --------------------

st.markdown("### Actual vs predicted sales")

performance_df = pd.DataFrame({
    "Actual Sales": y_test.to_numpy(),
    "Predicted Sales": test_predictions,
})

fig = px.scatter(
    performance_df,
    x="Actual Sales",
    y="Predicted Sales",
    title="Actual vs Predicted Sales (Test Set)",
    labels={
        "Actual Sales": "Actual sales units",
        "Predicted Sales": "Predicted sales units",
    },
    color_discrete_sequence=[BURGUNDY],
)

line_low = min(
    float(performance_df["Actual Sales"].min()),
    float(performance_df["Predicted Sales"].min()),
)

line_high = max(
    float(performance_df["Actual Sales"].max()),
    float(performance_df["Predicted Sales"].max()),
)

fig.add_shape(
    type="line",
    x0=line_low,
    y0=line_low,
    x1=line_high,
    y1=line_high,
    line=dict(color=EMERALD, dash="dash", width=2),
)

style_plot(fig, 420)
st.plotly_chart(fig, use_container_width=True)


# -------------------- RESIDUAL GRAPH --------------------

st.markdown("### Prediction errors")

performance_df["Residual"] = (
    performance_df["Actual Sales"]
    - performance_df["Predicted Sales"]
)

fig = px.scatter(
    performance_df,
    x="Predicted Sales",
    y="Residual",
    title="Prediction Residuals",
    labels={
        "Predicted Sales": "Predicted sales units",
        "Residual": "Actual minus predicted sales",
    },
    color_discrete_sequence=[LAVENDER],
)

fig.add_hline(
    y=0,
    line_dash="dash",
    line_color=BURGUNDY,
)

style_plot(fig, 380)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
    <div class="insight-card">
        <div class="insight-title">Test-set performance</div>
        <div class="insight-body">
            Mean Absolute Error: <b>{mae:.2f}</b> sales units.
            R²: <b>{r2:.3f}</b>.
            These metrics describe the held-out test set and do not
            guarantee future prediction accuracy.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- 06: MODEL COEFFICIENTS --------------------

section(
    "06",
    "Inside the model",
    "Compare the direction and relative magnitude of the "
    "standardised regression coefficients.",
)

coef_df = pd.DataFrame({
    "Business factor": [f.replace("_", " ") for f in FEATURES],
    "Coefficient": standardised_coefficients,
})

coef_df["Direction"] = np.where(
    coef_df["Coefficient"] >= 0,
    "Positive",
    "Negative",
)

coef_df = coef_df.sort_values("Coefficient")

fig = px.bar(
    coef_df,
    x="Coefficient",
    y="Business factor",
    orientation="h",
    color="Direction",
    color_discrete_map={
        "Positive": EMERALD,
        "Negative": BURGUNDY,
    },
    text=coef_df["Coefficient"].map(lambda x: f"{x:+.2f}"),
    title="Standardised Regression Coefficients",
)

fig.update_traces(textposition="outside", cliponaxis=False)
fig.add_vline(x=0, line_width=1, line_color=GOLD)

style_plot(fig, 390)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View coefficients in original units"):
    original_coef_df = pd.DataFrame({
        "Business factor": FEATURES,
        "Original-unit coefficient": original_coefficients,
    })

    st.dataframe(
        original_coef_df.style.format({
            "Original-unit coefficient": "{:.3f}"
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "These coefficients come from a separate ordinary LinearRegression "
        "model trained on the same training records without StandardScaler. "
        "They are comparable to the coefficient units used by project.py."
    )

with st.expander("How to interpret the model and its limitations"):
    st.markdown(
        """
        - **Linear Regression:** estimates a linear relationship between
          the four inputs and sales.
        - **Standardisation:** puts input features on comparable scales.
        - **R²:** measures the share of sales variation explained on this
          particular held-out test split.
        - **MAE:** measures average absolute prediction error in sales units.
        - **Small simulated dataset:** results may change with other data
          or different train/test splits.
        - **No causal proof:** coefficients and correlations do not establish
          that changing a factor causes sales to change.
        - **Range limitation:** predictions can remain uncertain even
          when inputs are inside the observed data range.
        """
    )


# -------------------- 07: COMPLETE BUSINESS ANALYSIS REPORT --------------------

section(
    "07",
    "Business analysis report",
    "Your dataset, correlation matrix, model evaluation, business insights, "
    "coefficients, example prediction and limitations — all in one place.",
)


# 07.1 Complete dataset
with st.expander("01 · Business dataset", expanded=True):
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", len(data))
    c2.metric("Dataset columns", len(data.columns))
    c3.metric("Source", "Simulated")

    st.caption("Simulated sample data for demonstration.")


# 07.2 Correlation matrix
with st.expander("02 · Correlation matrix", expanded=True):
    st.dataframe(
        correlation.style.format("{:.6f}"),
        use_container_width=True,
    )

    st.caption(
        "Positive values indicate positive linear association; negative "
        "values indicate negative linear association. Correlation does "
        "not establish cause and effect."
    )


# 07.3 Model evaluation
with st.expander("03 · Model evaluation", expanded=True):
    c1, c2, c3 = st.columns(3)

    c1.metric("Training records", len(X_train))
    c2.metric("Testing records", len(X_test))
    c3.metric("MAE", f"{mae:.3f} sales units")

    st.metric("R²", f"{r2:.3f}")

    st.write(
        "**MAE:** average absolute difference between actual and predicted "
        "sales. Lower is better."
    )

    st.write(
        "**R²:** measures how well the model explains variation in sales "
        "on this particular test set."
    )


# 07.4 Business insights
with st.expander("04 · Correlation-based business insights", expanded=True):
    for feature in FEATURES:
        value = float(correlation.loc[feature, "Sales"])

        st.markdown(
            f"**{feature.replace('_', ' ')}** → "
            f"{relationship_label(value)} "
            f"(corr = {value:.2f})"
        )


# 07.5 Model coefficients
with st.expander("05 · Model coefficients", expanded=True):
    report_coef_df = pd.DataFrame({
        "Business factor": FEATURES,
        "Standardised coefficient": standardised_coefficients,
        "Original-unit coefficient": original_coefficients,
    })

    st.dataframe(
        report_coef_df.style.format({
            "Standardised coefficient": "{:.3f}",
            "Original-unit coefficient": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "A coefficient estimates the fitted association between an input "
        "and predicted sales while the other inputs are held constant. "
        "Standardised coefficients and original-unit coefficients use "
        "different scales."
    )

    st.caption(
        "These are associations, not proof of cause and effect."
    )


# 07.6 Example prediction
with st.expander("06 · Example sales prediction", expanded=True):
    example_values = {
        "Ad_Spend": 3500,
        "Price": 8,
        "Economy": 70,
        "Competition": 5,
    }

    example_df = pd.DataFrame(
        [example_values],
        columns=FEATURES,
    )

    example_prediction = float(model.predict(example_df)[0])

    st.dataframe(
        example_df,
        use_container_width=True,
        hide_index=True,
    )

    st.metric(
        "Predicted sales",
        f"{example_prediction:,.2f} units",
    )

    st.caption(
        "This uses the example scenario from project.py. Its prediction "
        "may differ because the model's training setup differs. The data "
        "is simulated and this is not a dependable real-world forecast."
    )


# 07.7 Limitations
limitations = [
    "The dataset is simulated and does not represent a real business.",
    "Only 90 records are available, so results may change with other data.",
    "Correlation does not prove that a factor causes sales to change.",
    "Related input features can make individual effects hard to separate.",
    "Test metrics are estimates based on this particular train/test split.",
    "Predictions can be uncertain even within the observed data range.",
]

with st.expander("07 · Limitations", expanded=True):
    for item in limitations:
        st.markdown(f"- {item}")


# -------------------- DOWNLOAD COMPLETE REPORT --------------------

report_lines = [
    "BUSINESS PREDICTOR | COMPLETE BUSINESS ANALYSIS REPORT",
    "=" * 55,
    "",
    "BUSINESS DATASET",
    data.to_string(index=False),
    "",
    f"Dataset rows: {len(data)}",
    f"Dataset columns: {len(data.columns)}",
    "Data source: Simulated sample data for demonstration",
    "",
    "CORRELATION MATRIX",
    correlation.to_string(),
    "",
    "MODEL EVALUATION",
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    f"MAE = {mae:.3f} sales units",
    f"R² = {r2:.3f}",
    "",
    "CORRELATION-BASED BUSINESS INSIGHTS",
]

for feature in FEATURES:
    value = float(correlation.loc[feature, "Sales"])

    report_lines.append(
        f"{feature}: {relationship_label(value)} "
        f"(corr = {value:.3f})"
    )

report_lines.extend([
    "",
    "STANDARDISED MODEL COEFFICIENTS",
])

for feature, coefficient in zip(
    FEATURES,
    standardised_coefficients,
):
    report_lines.append(f"{feature}: {coefficient:.3f}")

report_lines.extend([
    "",
    "ORIGINAL-UNIT MODEL COEFFICIENTS",
])

for feature, coefficient in zip(
    FEATURES,
    original_coefficients,
):
    report_lines.append(f"{feature}: {coefficient:.3f}")

report_lines.extend([
    "",
    "EXAMPLE SALES PREDICTION",
    "Ad_Spend = 3500",
    "Price = 8",
    "Economy = 70",
    "Competition = 5",
    f"Predicted Sales = {example_prediction:.2f}",
    "",
    "LIMITATIONS",
])

report_lines.extend(f"- {item}" for item in limitations)

report_lines.extend([
    "",
    "Note: Simulated educational data. Predictions are exploratory "
    "estimates and do not prove causation or guarantee real-world results.",
])

st.download_button(
    "↓ Download complete business analysis report",
    data="\n".join(report_lines),
    file_name="Business_Analysis_Report.txt",
    mime="text/plain",
    use_container_width=True,
)


# -------------------- 08: SOURCE DATASET AND CSV DOWNLOAD --------------------

section(
    "08",
    "Inspect the source dataset",
    "Review the underlying records or download the CSV.",
)

with st.expander("Open complete source dataset"):
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "↓ Download dataset CSV",
        data=csv_bytes,
        file_name="Business_Predictor_Simulated_Data.csv",
        mime="text/csv",
    )


# -------------------- FOOTER --------------------

st.markdown("---")

st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;
                gap:12px;flex-wrap:wrap;color:{MUTED};
                font-size:.78rem;padding:.25rem 0 1rem 0;">
        <span>
            <b style="color:{BURGUNDY};">BUSINESS PREDICTOR</b>
            &nbsp;·&nbsp; Sales Intelligence Dashboard
        </span>
        <span>
            Educational prototype · Simulated data · Linear Regression
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
