import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="The Concourse — Customer Segment Directory",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DESIGN TOKENS
# ============================================================================
INK        = "#F4F1E6"   # warm ivory text
INK_MUTE   = "#9FB4AC"   # muted sage-grey text
BG_DEEP    = "#071815"   # near-black bottle green
BG_PANEL   = "#0E241F"   # panel green
LINE       = "rgba(201,162,39,0.24)"   # brass hairline
GOLD       = "#C9A227"   # brass
GOLD_BRT   = "#E8C468"   # bright brass
CORAL      = "#E8734A"   # signage coral (spending / accent 2)
TEAL       = "#3E9C82"   # signage teal (accent 3)
PLUM       = "#8C6BAE"   # accent 4
SLATE      = "#5B7A73"   # accent 5

CLUSTER_PALETTE = [GOLD_BRT, CORAL, TEAL, PLUM, SLATE, "#D65D8A", "#4E8FBF"]

FONT_CSS = "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap"

def plot_theme(fig, height=440, showlegend=True):
    """Single helper to apply the app's plotly theme — avoids update_layout()
    duplicate-keyword clashes by always routing through one function."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Space Grotesk, sans-serif", color=INK, size=13),
        title_font=dict(family="Fraunces, serif", color=INK, size=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=LINE,
            borderwidth=1,
            font=dict(size=11, color=INK_MUTE),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(
            bgcolor=BG_PANEL,
            font_family="Space Grotesk, sans-serif",
            font_color=INK,
            bordercolor=GOLD,
        ),
    )
    fig.update_xaxes(gridcolor="rgba(159,180,172,0.10)", zerolinecolor="rgba(159,180,172,0.15)")
    fig.update_yaxes(gridcolor="rgba(159,180,172,0.10)", zerolinecolor="rgba(159,180,172,0.15)")
    return fig

# ============================================================================
# GLOBAL CSS
# ============================================================================
st.markdown(f"""
<link href="{FONT_CSS}" rel="stylesheet">
<style>
:root {{
    --ink: {INK}; --ink-mute: {INK_MUTE}; --gold: {GOLD}; --gold-brt: {GOLD_BRT};
    --coral: {CORAL}; --teal: {TEAL}; --line: {LINE};
}}
html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif; }}

.stApp {{
    background:
        radial-gradient(1200px 600px at 12% -8%, rgba(201,162,39,0.10), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(62,156,130,0.10), transparent 55%),
        linear-gradient(180deg, {BG_DEEP} 0%, #0A1F1B 55%, {BG_DEEP} 100%);
    color: {INK};
}}

/* Hide default chrome */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---- Directory board hero ---- */
.directory-board {{
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 2.4rem 2.6rem 2rem 2.6rem;
    background: linear-gradient(180deg, rgba(232,196,104,0.05), rgba(255,255,255,0.015));
    position: relative;
    margin-bottom: 1.6rem;
    overflow: hidden;
}}
.directory-board::before {{
    content: "";
    position: absolute; inset: 8px;
    border: 1px solid rgba(201,162,39,0.16);
    border-radius: 2px;
    pointer-events: none;
}}
.directory-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.3em;
    font-size: 0.68rem;
    color: {GOLD_BRT};
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.6rem;
}}
.directory-eyebrow::before {{ content: "◆"; font-size: 0.6rem; }}
.directory-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1.08;
    margin: 0 0 0.5rem 0;
    color: {INK};
}}
.directory-title em {{ color: {GOLD_BRT}; font-style: italic; }}
.directory-sub {{
    color: {INK_MUTE};
    font-size: 1.02rem;
    max-width: 640px;
    line-height: 1.55;
}}
.directory-arrow-row {{
    display: flex; gap: 1.1rem; margin-top: 1.6rem; flex-wrap: wrap;
}}
.arrow-chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    color: {INK_MUTE};
    border: 1px solid {LINE};
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
}}

/* ---- Section labels ---- */
.wing-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: {GOLD_BRT};
    border-left: 2px solid {GOLD};
    padding-left: 0.6rem;
    margin: 1.6rem 0 0.2rem 0;
}}
.section-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.7rem;
    color: {INK};
    margin: 0.1rem 0 0.9rem 0;
}}
.section-note {{
    color: {INK_MUTE};
    font-size: 0.93rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}}

/* ---- Glass card ---- */
.glass-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 1.3rem 1.4rem;
    backdrop-filter: blur(6px);
    height: 100%;
}}

/* ---- Metric tiles ---- */
.metric-tile {{
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 1.1rem 1.2rem;
    background: rgba(255,255,255,0.025);
    text-align: left;
}}
.metric-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {INK_MUTE};
}}
.metric-value {{
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: {GOLD_BRT};
    margin-top: 0.15rem;
}}
.metric-sub {{ font-size: 0.78rem; color: {INK_MUTE}; margin-top: 0.1rem; }}

/* ---- Segment directory listing card ---- */
.segment-card {{
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 1.15rem 1.3rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    margin-bottom: 0.85rem;
    position: relative;
}}
.segment-card .stall-no {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {INK_MUTE};
    letter-spacing: 0.12em;
}}
.segment-card .stall-name {{
    font-family: 'Fraunces', serif;
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0.15rem 0 0.35rem 0;
}}
.segment-card .stall-desc {{ color: {INK_MUTE}; font-size: 0.88rem; line-height: 1.5; }}
.segment-card .stall-action {{
    font-size: 0.82rem;
    margin-top: 0.55rem;
    padding-top: 0.55rem;
    border-top: 1px dashed {LINE};
    color: {INK};
}}
.segment-card .stall-action b {{ color: {GOLD_BRT}; }}

/* ---- Tag pill ---- */
.tag-pill {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    border: 1px solid;
    margin-right: 0.4rem;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #061513 0%, #0A1F1B 100%);
    border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label {{
    color: {INK_MUTE} !important;
}}
.sidebar-brand {{
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: {INK};
    padding-bottom: 0.2rem;
    border-bottom: 1px solid {LINE};
    margin-bottom: 1rem;
}}
.sidebar-brand span {{ color: {GOLD_BRT}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {INK_MUTE};
    background: transparent;
    border-radius: 3px 3px 0 0;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD_BRT} !important;
    border-bottom: 2px solid {GOLD_BRT};
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 4px; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background: rgba(232,196,104,0.08);
    border: 1px solid {GOLD};
    color: {GOLD_BRT};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    border-radius: 3px;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background: {GOLD}; color: {BG_DEEP};
}}

/* Slider accent */
[data-testid="stSlider"] div[role="slider"] {{ background-color: {GOLD_BRT} !important; }}

hr {{ border-color: {LINE}; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & PIPELINE
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.rename(columns={
        "Annual Income (k$)": "Annual_Income",
        "Spending Score (1-100)": "Spending_Score"
    }, inplace=True)
    df.drop(columns=["CustomerID"], inplace=True)
    le = LabelEncoder()
    df["Gender_Code"] = le.fit_transform(df["Gender"])
    return df

@st.cache_data
def scale_features(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[["Age", "Annual_Income", "Spending_Score"]] = scaler.fit_transform(
        df[["Age", "Annual_Income", "Spending_Score"]]
    )
    return df_scaled[["Annual_Income", "Spending_Score"]]

@st.cache_data
def elbow_and_silhouette(df_2f):
    inertia, sil = [], []
    for k in range(1, 11):
        m = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_2f)
        inertia.append(m.inertia_)
        if k >= 2:
            sil.append(silhouette_score(df_2f, m.labels_))
    return inertia, sil

@st.cache_data
def run_kmeans(df_2f, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(df_2f)
    return labels, model.cluster_centers_

@st.cache_data
def run_hier(df_2f, k):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(df_2f)

@st.cache_data
def knn_distance(df_2f, n=4):
    nbrs = NearestNeighbors(n_neighbors=n).fit(df_2f)
    dist, _ = nbrs.kneighbors(df_2f)
    return np.sort(dist[:, n - 1])

@st.cache_data
def dbscan_grid(df_2f, eps_values, min_values):
    rows = []
    for eps in eps_values:
        for ms in min_values:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(df_2f)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise = int((labels == -1).sum())
            noise_pct = round(100 * noise / len(labels), 1)
            score = np.nan
            if n_clusters > 1:
                m = labels != -1
                if m.sum() > 1 and len(set(labels[m])) > 1:
                    score = silhouette_score(df_2f[m], labels[m])
            rows.append([eps, ms, n_clusters, noise, noise_pct, score])
    return pd.DataFrame(rows, columns=["eps", "min_samples", "Clusters", "Noise Points", "Noise %", "Silhouette"])

@st.cache_data
def run_dbscan(df_2f, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(df_2f)

def cluster_profile(dframe, col):
    return dframe.groupby(col)[["Age", "Annual_Income", "Spending_Score"]].mean().round(2)

def segment_label(row, ref_df):
    income_hi = row["Annual_Income"] > ref_df["Annual_Income"].median()
    spend_hi = row["Spending_Score"] > ref_df["Spending_Score"].median()
    if income_hi and spend_hi:
        return "High Income · High Spend", "Premium loyalty & concierge perks", CORAL
    if income_hi and not spend_hi:
        return "High Income · Low Spend", "Personalised nudges to re-engage", TEAL
    if not income_hi and spend_hi:
        return "Low Income · High Spend", "Value bundles & flash promos", PLUM
    if not income_hi and not spend_hi:
        return "Low Income · Low Spend", "Budget-friendly entry offers", SLATE
    return "Average Income · Average Spend", "Steady seasonal engagement", GOLD_BRT

df = load_data()
df_2f = scale_features(df)

# ============================================================================
# SIDEBAR — CONTROLS
# ============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🧭 The <span>Concourse</span></div>', unsafe_allow_html=True)
    st.caption("Mall customer segmentation — directory of every wing.")

    section = st.radio(
        "Navigate the concourse",
        ["Entrance · Overview", "Wing A · K-Means", "Wing B · Hierarchical",
         "Wing C · DBSCAN", "Atrium · Comparison", "Directory · Business Insights"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Filter the floor**")
    gender_filter = st.multiselect("Gender", options=sorted(df["Gender"].unique()), default=list(df["Gender"].unique()))
    age_range = st.slider("Age range", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))

    st.markdown("---")
    st.markdown("**Model dials**")
    k_kmeans = st.slider("K-Means clusters (k)", 2, 10, 5)
    k_hier = st.slider("Hierarchical clusters", 2, 10, 5)
    eps_val = st.slider("DBSCAN eps", 0.15, 1.0, 0.40, 0.05)
    min_samples_val = st.slider("DBSCAN min_samples", 3, 10, 5)

    st.markdown("---")
    st.caption("Built from the *UL_PR1* unsupervised-learning notebook — K-Means, Agglomerative & DBSCAN on Annual Income vs. Spending Score.")

mask = df["Gender"].isin(gender_filter) & df["Age"].between(age_range[0], age_range[1])
dff = df[mask].copy() if mask.sum() >= 10 else df.copy()
if mask.sum() < 10:
    st.warning("Too few customers match this filter — showing the full floor instead.")
    mask = pd.Series(True, index=df.index)

# ============================================================================
# HERO — DIRECTORY BOARD
# ============================================================================
st.markdown(f"""
<div class="directory-board">
    <div class="directory-eyebrow">Mall Concourse · Level 2 · Data Wing</div>
    <div class="directory-title">You are here: the <em>Customer<br>Segment Directory</em></div>
    <div class="directory-sub">
        200 shoppers, three ways of reading the floor. K-Means finds the centres of gravity,
        Hierarchical traces how groups nest inside each other, and DBSCAN flags who doesn't
        belong to any crowd at all. Walk the wings on the left to compare them.
    </div>
    <div class="directory-arrow-row">
        <span class="arrow-chip">◀ Entrance: EDA</span>
        <span class="arrow-chip">Wing A: K-Means</span>
        <span class="arrow-chip">Wing B: Hierarchical</span>
        <span class="arrow-chip">Wing C: DBSCAN ▶</span>
    </div>
</div>
""", unsafe_allow_html=True)

dff = df[mask]

# ============================================================================
# ENTRANCE — OVERVIEW & EDA
# ============================================================================
if section == "Entrance · Overview":
    st.markdown('<div class="wing-label">Entrance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Foot Traffic Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Everything downstream — every cluster, every business action — starts with what walks through this door.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        ("SHOPPERS ON FLOOR", f"{len(dff)}", f"of {len(df)} total"),
        ("MEDIAN AGE", f"{dff.Age.median():.0f}", "years"),
        ("MEDIAN INCOME", f"${dff.Annual_Income.median():.0f}k", "annual"),
        ("MEDIAN SPEND SCORE", f"{dff.Spending_Score.median():.0f}", "out of 100"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4], tiles):
        col.markdown(f"""<div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns([1.3, 1])
    with colA:
        fig = go.Figure()
        for col_name, color in zip(["Age", "Annual_Income", "Spending_Score"], [GOLD_BRT, CORAL, TEAL]):
            fig.add_trace(go.Histogram(x=dff[col_name], name=col_name.replace("_", " "),
                                        marker_color=color, opacity=0.55, nbinsx=20))
        fig.update_layout(barmode="overlay", title="Distribution — Age · Income · Spending Score")
        st.plotly_chart(plot_theme(fig, height=380), use_container_width=True)
    with colB:
        gender_counts = dff["Gender"].value_counts()
        fig2 = go.Figure(data=[go.Pie(
            labels=gender_counts.index, values=gender_counts.values, hole=0.62,
            marker=dict(colors=[GOLD_BRT, TEAL], line=dict(color=BG_DEEP, width=2)),
            textfont=dict(color=INK, family="Space Grotesk"),
        )])
        fig2.update_layout(title="Gender Split")
        st.plotly_chart(plot_theme(fig2, height=380), use_container_width=True)

    st.markdown('<div class="wing-label">Correlation Check</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why Income & Spending Are Chosen</div>', unsafe_allow_html=True)
    corr = dff[["Age", "Annual_Income", "Spending_Score", "Gender_Code"]].corr().round(2)
    fig3 = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#0E241F"], [0.5, "#3E9C82"], [1, "#E8C468"]],
        text=corr.values, texttemplate="%{text}", textfont=dict(color=INK),
        zmin=-1, zmax=1,
    ))
    fig3.update_layout(title="Correlation Heatmap")
    st.plotly_chart(plot_theme(fig3, height=420, showlegend=False), use_container_width=True)
    st.markdown("""<div class="section-note">
    No feature pair is strongly linearly correlated — typical for segmentation data. Annual Income
    and Spending Score give the clearest visual separation between customer groups, which is why
    every wing of this concourse clusters on that pair.</div>""", unsafe_allow_html=True)

# ============================================================================
# WING A — K-MEANS
# ============================================================================
elif section == "Wing A · K-Means":
    st.markdown('<div class="wing-label">Wing A</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">K-Means — Centres of Gravity</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-note">Each shopper is pulled toward the nearest of <b>{k_kmeans}</b> centroids. Adjust <i>k</i> from the sidebar dial to see the floor redraw itself.</div>', unsafe_allow_html=True)

    inertia, sil = elbow_and_silhouette(df_2f)
    colE, colS = st.columns(2)
    with colE:
        fig = go.Figure(go.Scatter(x=list(range(1, 11)), y=inertia, mode="lines+markers",
                                    line=dict(color=GOLD_BRT, width=3), marker=dict(size=8, color=CORAL)))
        fig.add_vline(x=k_kmeans, line_dash="dash", line_color=TEAL)
        fig.update_layout(title="Elbow Method", xaxis_title="k", yaxis_title="Inertia")
        st.plotly_chart(plot_theme(fig, height=340, showlegend=False), use_container_width=True)
    with colS:
        fig = go.Figure(go.Scatter(x=list(range(2, 11)), y=sil, mode="lines+markers",
                                    line=dict(color=TEAL, width=3), marker=dict(size=8, color=GOLD_BRT)))
        fig.add_vline(x=k_kmeans, line_dash="dash", line_color=CORAL)
        fig.update_layout(title="Silhouette Score", xaxis_title="k", yaxis_title="Score")
        st.plotly_chart(plot_theme(fig, height=340, showlegend=False), use_container_width=True)
    best_k = list(range(2, 11))[int(np.argmax(sil))]
    st.caption(f"Best silhouette score is at **k = {best_k}** — the dial defaults to 5, matching the notebook's chosen segmentation.")

    labels_full, centers_full = run_kmeans(df_2f, k_kmeans)
    df_all = df.copy()
    df_all["Cluster"] = labels_full.astype(str)
    dff_k = df_all[mask]

    fig = go.Figure()
    for i, cl in enumerate(sorted(dff_k["Cluster"].unique(), key=int)):
        sub = dff_k[dff_k["Cluster"] == cl]
        fig.add_trace(go.Scatter(x=sub.Annual_Income, y=sub.Spending_Score, mode="markers",
                                  name=f"Cluster {cl}", marker=dict(size=9, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
                                  line=dict(width=1, color=BG_DEEP))))
    fig.add_trace(go.Scatter(x=centers_full[:, 0] * df.Annual_Income.std() + df.Annual_Income.mean(),
                              y=centers_full[:, 1] * df.Spending_Score.std() + df.Spending_Score.mean(),
                              mode="markers", name="Centroid",
                              marker=dict(symbol="x", size=16, color=INK, line=dict(width=2, color=GOLD_BRT))))
    fig.update_layout(title="K-Means Clusters — Annual Income vs Spending Score",
                       xaxis_title="Annual Income (k$)", yaxis_title="Spending Score")
    st.plotly_chart(plot_theme(fig, height=460), use_container_width=True)

    st.markdown('<div class="wing-label">Storefront Profiles</div>', unsafe_allow_html=True)
    prof = cluster_profile(df_all, "Cluster")
    grid = st.columns(min(3, len(prof)))
    for i, (cl, row) in enumerate(prof.iterrows()):
        name, action, color = segment_label(row, df)
        grid[i % len(grid)].markdown(f"""
        <div class="segment-card" style="border-left:3px solid {color};">
            <div class="stall-no">STALL {int(cl):02d}</div>
            <div class="stall-name">{name}</div>
            <div class="stall-desc">Age {row.Age:.0f} · Income ${row.Annual_Income:.0f}k · Spend {row.Spending_Score:.0f}</div>
            <div class="stall-action">Action: <b>{action}</b></div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Full cluster profile table"):
        st.dataframe(prof.style.background_gradient(cmap="YlOrBr"), use_container_width=True)

# ============================================================================
# WING B — HIERARCHICAL
# ============================================================================
elif section == "Wing B · Hierarchical":
    st.markdown('<div class="wing-label">Wing B</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Agglomerative Clustering — Nested Neighbourhoods</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Ward linkage merges the two closest neighbourhoods at every step. The dendrogram below reads bottom-up: each merge height is how far apart those groups were.</div>', unsafe_allow_html=True)

    linked = linkage(df_2f, method="ward")
    dendro = ff.create_dendrogram(df_2f.values, linkagefun=lambda x: linked, color_threshold=15)
    dendro.update_layout(title="Dendrogram — Ward Linkage")
    for trace in dendro["data"]:
        trace["line"]["color"] = GOLD_BRT if trace["line"]["color"] == "grey" else TEAL
    dendro.add_hline(y=15, line_dash="dash", line_color=CORAL,
                      annotation_text="cut line", annotation_font_color=CORAL)
    st.plotly_chart(plot_theme(dendro, height=420, showlegend=False), use_container_width=True)

    hier_labels_full = run_hier(df_2f, k_hier)
    df_all = df.copy()
    df_all["Cluster"] = hier_labels_full.astype(str)
    dff_h = df_all[mask]

    col1, col2 = st.columns([1.1, 1])
    with col1:
        fig = go.Figure()
        for i, cl in enumerate(sorted(dff_h["Cluster"].unique(), key=int)):
            sub = dff_h[dff_h["Cluster"] == cl]
            fig.add_trace(go.Scatter(x=sub.Annual_Income, y=sub.Spending_Score, mode="markers",
                                      name=f"Cluster {cl}", marker=dict(size=9, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])))
        fig.update_layout(title="Hierarchical Clusters", xaxis_title="Annual Income (k$)", yaxis_title="Spending Score")
        st.plotly_chart(plot_theme(fig, height=400), use_container_width=True)
    with col2:
        km_labels_full, _ = run_kmeans(df_2f, k_hier)
        agree = (km_labels_full == hier_labels_full).mean()
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Ward vs Centroid**")
        st.markdown("""
- **K-Means** is centroid-based — every point joins its nearest centre.
- **Hierarchical** is connectivity-based — it grows from the bottom up, merging the closest pairs first.
- Boundary shoppers can land in different clusters between the two even when both use the same *k*.
        """)
        st.metric("Label agreement with K-Means (same k)", f"{agree*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="wing-label">Storefront Profiles</div>', unsafe_allow_html=True)
    prof = cluster_profile(df_all, "Cluster")
    st.dataframe(prof.style.background_gradient(cmap="YlOrBr"), use_container_width=True)

# ============================================================================
# WING C — DBSCAN
# ============================================================================
elif section == "Wing C · DBSCAN":
    st.markdown('<div class="wing-label">Wing C</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">DBSCAN — Density & the Unaffiliated Shopper</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-note">Current dial: <b>eps = {eps_val}</b>, <b>min_samples = {min_samples_val}</b>. Points in dense neighbourhoods form a cluster; everyone else is flagged as noise.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        dist = knn_distance(df_2f.values, n=4)
        fig = go.Figure(go.Scatter(y=dist, mode="lines", line=dict(color=CORAL, width=2.5)))
        fig.add_hline(y=eps_val, line_dash="dash", line_color=GOLD_BRT, annotation_text=f"eps={eps_val}")
        fig.update_layout(title="4-NN Distance Plot", xaxis_title="Points (sorted)", yaxis_title="4th NN distance")
        st.plotly_chart(plot_theme(fig, height=360, showlegend=False), use_container_width=True)
    with col2:
        grid = dbscan_grid(df_2f.values, [0.2, 0.3, 0.4, 0.5, 0.6], [3, 4, 5, 6])
        pivot = grid.pivot(index="min_samples", columns="eps", values="Silhouette")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#0E241F"], [0.5, "#3E9C82"], [1, "#E8C468"]],
            text=np.round(pivot.values, 2), texttemplate="%{text}", textfont=dict(color=INK),
        ))
        fig.update_layout(title="Silhouette — eps × min_samples grid", xaxis_title="eps", yaxis_title="min_samples")
        st.plotly_chart(plot_theme(fig, height=360, showlegend=False), use_container_width=True)

    labels_full = run_dbscan(df_2f.values, eps_val, min_samples_val)
    df_all = df.copy()
    df_all["Cluster"] = labels_full
    n_clusters = len(set(labels_full)) - (1 if -1 in labels_full else 0)
    noise_pct = (labels_full == -1).mean() * 100

    c1, c2, c3 = st.columns(3)
    for col, (label, val) in zip([c1, c2, c3], [
        ("CLUSTERS FOUND", str(n_clusters)),
        ("NOISE POINTS", f"{int((labels_full==-1).sum())}"),
        ("NOISE %", f"{noise_pct:.1f}%"),
    ]):
        col.markdown(f"""<div class="metric-tile"><div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)

    dff_d = df_all[mask]
    fig = go.Figure()
    for i, cl in enumerate(sorted([c for c in dff_d["Cluster"].unique() if c != -1])):
        sub = dff_d[dff_d["Cluster"] == cl]
        fig.add_trace(go.Scatter(x=sub.Annual_Income, y=sub.Spending_Score, mode="markers",
                                  name=f"Cluster {cl}", marker=dict(size=9, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])))
    noise = dff_d[dff_d["Cluster"] == -1]
    if len(noise):
        fig.add_trace(go.Scatter(x=noise.Annual_Income, y=noise.Spending_Score, mode="markers",
                                  name="Noise", marker=dict(symbol="x", size=11, color=INK_MUTE)))
    fig.update_layout(title="DBSCAN Clusters — Annual Income vs Spending Score",
                       xaxis_title="Annual Income (k$)", yaxis_title="Spending Score")
    st.plotly_chart(plot_theme(fig, height=460), use_container_width=True)
    st.markdown("""<div class="section-note">DBSCAN doesn't need a preset number of clusters and can flag genuine outliers —
    but it struggles when clusters have uneven density, which is common in retail spend data.</div>""", unsafe_allow_html=True)

# ============================================================================
# ATRIUM — COMPARISON
# ============================================================================
elif section == "Atrium · Comparison":
    st.markdown('<div class="wing-label">Atrium</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">All Three Wings, Side by Side</div>', unsafe_allow_html=True)

    km_labels, _ = run_kmeans(df_2f, 5)
    hier_labels = run_hier(df_2f, 5)
    db_labels = run_dbscan(df_2f.values, 0.4, 5)

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=("K-Means", "Hierarchical", "DBSCAN"))
    for i, cl in enumerate(sorted(set(km_labels))):
        m = km_labels == cl
        fig.add_trace(go.Scatter(x=df.Annual_Income[m], y=df.Spending_Score[m], mode="markers",
                                  marker=dict(size=7, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]), showlegend=False), row=1, col=1)
    for i, cl in enumerate(sorted(set(hier_labels))):
        m = hier_labels == cl
        fig.add_trace(go.Scatter(x=df.Annual_Income[m], y=df.Spending_Score[m], mode="markers",
                                  marker=dict(size=7, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]), showlegend=False), row=1, col=2)
    for i, cl in enumerate(sorted(set(db_labels) - {-1})):
        m = db_labels == cl
        fig.add_trace(go.Scatter(x=df.Annual_Income[m], y=df.Spending_Score[m], mode="markers",
                                  marker=dict(size=7, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]), showlegend=False), row=1, col=3)
    m = db_labels == -1
    if m.any():
        fig.add_trace(go.Scatter(x=df.Annual_Income[m], y=df.Spending_Score[m], mode="markers",
                                  marker=dict(symbol="x", size=9, color=INK_MUTE), showlegend=False), row=1, col=3)
    fig.update_layout(title="Clustering Algorithm Comparison — Mall Customer Segmentation")
    fig.update_xaxes(title_text="Annual Income (k$)")
    fig.update_yaxes(title_text="Spending Score", row=1, col=1)
    st.plotly_chart(plot_theme(fig, height=420, showlegend=False), use_container_width=True)

    km_sil = silhouette_score(df_2f, km_labels)
    hier_sil = silhouette_score(df_2f, hier_labels)
    mask_db = db_labels != -1
    db_sil = silhouette_score(df_2f[mask_db], db_labels[mask_db])

    metrics = pd.DataFrame({
        "Algorithm": ["K-Means", "Hierarchical", "DBSCAN"],
        "No. of Clusters": [len(set(km_labels)), len(set(hier_labels)), len(set(db_labels) - {-1})],
        "Silhouette Score": [round(km_sil, 3), round(hier_sil, 3), round(db_sil, 3)],
    })
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    with col2:
        fig = go.Figure(go.Bar(x=metrics.Algorithm, y=metrics["Silhouette Score"],
                                marker_color=[GOLD_BRT, TEAL, CORAL], text=metrics["Silhouette Score"], textposition="outside"))
        fig.update_layout(title="Silhouette Score by Algorithm", yaxis_title="Score")
        st.plotly_chart(plot_theme(fig, height=320, showlegend=False), use_container_width=True)

    st.info("**K-Means** is recommended for this floor plan — simplest to explain, clearest segments. Hierarchical is best for visualising nesting; DBSCAN is best for outlier/noise detection.")

# ============================================================================
# DIRECTORY — BUSINESS INSIGHTS
# ============================================================================
else:
    st.markdown('<div class="wing-label">Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Business Insights & Recommended Actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Five archetypes recur across every algorithm on this floor. Here is what to do about each one.</div>', unsafe_allow_html=True)

    labels_full, _ = run_kmeans(df_2f, 5)
    df_all = df.copy()
    df_all["Cluster"] = labels_full.astype(str)
    prof = cluster_profile(df_all, "Cluster")

    for cl, row in prof.iterrows():
        name, action, color = segment_label(row, df)
        share = (df_all["Cluster"] == cl).mean() * 100
        st.markdown(f"""
        <div class="segment-card" style="border-left:3px solid {color};">
            <div class="stall-no">STALL {int(cl):02d} · {share:.0f}% OF FLOOR</div>
            <div class="stall-name">{name}</div>
            <div class="stall-desc">Age {row.Age:.0f} · Income ${row.Annual_Income:.0f}k · Spend score {row.Spending_Score:.0f}</div>
            <div class="stall-action">Recommended action: <b>{action}</b></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="wing-label">Take It With You</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Download the Segmented Roster</div>', unsafe_allow_html=True)
    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download segmented customers (CSV)", csv, "mall_customers_segmented.csv", "text/csv")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(f"""
<div style="margin-top:2.5rem; padding-top:1rem; border-top:1px solid {LINE};
color:{INK_MUTE}; font-size:0.78rem; font-family:'JetBrains Mono',monospace;">
THE CONCOURSE · Unsupervised Learning Directory · K-Means · Agglomerative · DBSCAN
</div>
""", unsafe_allow_html=True)
