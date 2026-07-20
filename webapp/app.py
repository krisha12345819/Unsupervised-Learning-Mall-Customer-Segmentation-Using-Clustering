import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

px.defaults.template = "plotly_dark"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE — custom CSS theme
# ============================================================
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
    --accent:#8B5CF6;
    --accent2:#22D3EE;
    --amber:#F59E0B;
    --teal:#2DD4BF;
    --bg:#0B0E14;
    --bg2:#11151F;
    --card:#151A24;
    --card-border:rgba(255,255,255,0.06);
    --text:#E6E9EF;
    --text-dim:#8A93A6;
}
html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
    color: var(--text);
}
h1, h2, h3, .hero-title{
    font-family: 'Poppins', sans-serif !important;
}
h1, h2, h3, h4, h5, p, span, label, div{
    color: var(--text);
}

.stApp{
    background: radial-gradient(circle at 15% 0%, #161B2C 0%, #0B0E14 45%, #0B0E14 100%);
}

/* Hero header */
.hero{
    background: linear-gradient(120deg, #1F2340 0%, #2A2050 50%, #3B1F45 100%);
    padding: 2.4rem 2.6rem;
    border-radius: 22px;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 20px 45px -15px rgba(0,0,0,0.6);
    border: 1px solid rgba(139,92,246,0.25);
    position: relative;
    overflow: hidden;
}
.hero::after{
    content:"";
    position:absolute; right:-60px; top:-60px;
    width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle, rgba(34,211,238,0.18), transparent 70%);
}
.hero-title{
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 .3rem 0;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #C4B5FD, #67E8F9);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub{
    font-size: 1.02rem;
    opacity: 0.85;
    font-weight: 400;
    max-width: 720px;
    color: #C7CCDA;
}

/* KPI cards */
.kpi-card{
    background: var(--card);
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    box-shadow: 0 8px 24px -14px rgba(0,0,0,0.6);
    border: 1px solid var(--card-border);
    text-align:left;
    transition: transform .15s ease, border-color .15s ease;
}
.kpi-card:hover{ transform: translateY(-3px); border-color: rgba(139,92,246,0.4); }
.kpi-label{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #A78BFA;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.kpi-value{
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    font-family: 'Poppins', sans-serif;
}
.kpi-icon{ font-size: 1.4rem; }

/* Section cards — Streamlit's native bordered container */
div[data-testid="stVerticalBlock"][data-test-scroll-behavior="normal"]{
    background: rgba(255,255,255,0.02);
    border-radius: 18px !important;
    border-color: rgba(139,92,246,0.18) !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 10px 30px -20px rgba(0,0,0,0.7);
    margin-bottom: 0.4rem;
}

/* Insight pill cards */
.insight-card{
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    color: white;
    margin-bottom: 0.9rem;
    box-shadow: 0 10px 22px -12px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.08);
}
.insight-title{ font-weight:700; font-size:1.02rem; margin-bottom:.3rem; color:white;}
.insight-body{ font-size:0.9rem; opacity:0.92; line-height:1.4; color:white;}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#0D1017 0%, #14182A 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #DCE0EA !important; }
section[data-testid="stSidebar"] .stRadio label{ font-weight:500; }
section[data-testid="stSidebar"] hr{ border-color: rgba(255,255,255,0.08); }

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap: 6px; }
.stTabs [data-baseweb="tab"]{
    background: rgba(139,92,246,0.08);
    border-radius: 12px 12px 0 0;
    padding: 10px 16px;
    font-weight: 600;
    color: #C7CCDA;
}
.stTabs [aria-selected="true"]{
    background: linear-gradient(120deg,#8B5CF6,#22D3EE) !important;
    color: #0B0E14 !important;
}

/* Dataframes & sliders blend with dark bg */
[data-testid="stDataFrame"]{ border-radius: 12px; overflow: hidden; }
.stSlider [data-baseweb="slider"] div[role="slider"]{ background-color: #8B5CF6; }

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""")

PALETTE = ["#22D3EE", "#A78BFA", "#F472B6", "#FBBF24", "#34D399", "#FB7185", "#A3E635"]

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.rename(columns={
        "Annual Income (k$)": "Annual_Income",
        "Spending Score (1-100)": "Spending_Score"
    }, inplace=True)
    df.drop(columns=["CustomerID"], inplace=True)
    le = LabelEncoder()
    df["Gender_Encoded"] = le.fit_transform(df["Gender"])
    return df

df_raw = load_data()

@st.cache_data
def scale_features(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[["Age", "Annual_Income", "Spending_Score"]] = scaler.fit_transform(
        df[["Age", "Annual_Income", "Spending_Score"]]
    )
    return df_scaled, scaler

df_scaled, scaler = scale_features(df_raw)
df2f = df_scaled[["Annual_Income", "Spending_Score"]]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🛍️ Segmentation Studio")
    st.caption("Mall Customers · Unsupervised Learning")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Exploratory Analysis", "🎯 K-Means", "🌳 Hierarchical",
         "🔍 DBSCAN", "⚖️ Compare Algorithms", "💡 Business Insights", "🔮 Predict a Customer"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("#### Filters")
    age_range = st.slider("Age range", int(df_raw.Age.min()), int(df_raw.Age.max()),
                           (int(df_raw.Age.min()), int(df_raw.Age.max())))
    gender_filter = st.multiselect("Gender", options=df_raw.Gender.unique().tolist(),
                                    default=df_raw.Gender.unique().tolist())
    st.divider()
    st.caption("Built with ❤️ using Streamlit + Plotly + Scikit-learn")

mask = (df_raw.Age.between(*age_range)) & (df_raw.Gender.isin(gender_filter))
df = df_raw[mask].reset_index(drop=True)
df2f_view = df2f[mask.values].reset_index(drop=True)

def hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">{title}</div>
        <div class="hero-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def kpi(col, icon, label, value):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# OVERVIEW
# ============================================================
if page == "🏠 Overview":
    hero("Mall Customer Segmentation Studio",
         "Explore customer behavior and discover hidden shopper segments using K-Means, "
         "Hierarchical Clustering and DBSCAN — all interactive, all live.")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "👥", "Customers", f"{len(df)}")
    kpi(c2, "🎂", "Avg Age", f"{df.Age.mean():.1f}")
    kpi(c3, "💰", "Avg Income", f"${df.Annual_Income.mean():.1f}k")
    kpi(c4, "🛒", "Avg Spending Score", f"{df.Spending_Score.mean():.1f}")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        with st.container(border=True):
            st.markdown("#### Income vs Spending Score")
            fig = px.scatter(df, x="Annual_Income", y="Spending_Score", color="Gender",
                              size="Age", hover_data=["Age"],
                              color_discrete_sequence=[PALETTE[0], PALETTE[1]])
            fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.markdown("#### Gender Split")
            gender_counts = df.Gender.value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            fig2 = px.pie(gender_counts, names="Gender", values="Count", hole=0.55,
                           color_discrete_sequence=[PALETTE[0], PALETTE[1]])
            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Data Preview")
        st.dataframe(df.head(15), use_container_width=True)

# ============================================================
# EDA
# ============================================================
elif page == "📊 Exploratory Analysis":
    hero("Exploratory Data Analysis", "Understand the shape of your data before clustering it.")

    with st.container(border=True):
        st.markdown("#### Distributions")
        cols = st.columns(3)
        for i, col_name in enumerate(["Age", "Annual_Income", "Spending_Score"]):
            with cols[i]:
                fig = px.histogram(df, x=col_name, nbins=20, marginal="box",
                                    color_discrete_sequence=[PALETTE[i % len(PALETTE)]])
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   title=col_name)
                st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### Correlation Heatmap")
            corr = df[["Age", "Annual_Income", "Spending_Score", "Gender_Encoded"]].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        with st.container(border=True):
            st.markdown("#### Pairwise Relationships")
            fig = px.scatter_matrix(df, dimensions=["Age", "Annual_Income", "Spending_Score"],
                                     color="Gender", color_discrete_sequence=[PALETTE[0], PALETTE[1]])
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-card" style="background:linear-gradient(120deg,#7C3AED,#EC4899);">
        <div class="insight-title">📌 EDA Takeaway</div>
        <div class="insight-body">Annual Income and Spending Score show the clearest natural groupings among customers,
        while Age has only a weak relationship with the other two — making Income & Spending Score the ideal 2D basis for clustering.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# K-MEANS
# ============================================================
elif page == "🎯 K-Means":
    hero("K-Means Clustering", "Centroid-based segmentation — choose k and watch clusters form live.")

    c1, c2 = st.columns([1, 2])
    with c1:
        with st.container(border=True):
            st.markdown("#### ⚙️ Controls")
            k = st.slider("Number of clusters (k)", 2, 10, 5)
            st.markdown("Elbow & silhouette plots below help justify your choice of *k*.")

            inertia, sil_scores = [], []
            for kk in range(1, 11):
                m = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(df2f)
                inertia.append(m.inertia_)
                if kk >= 2:
                    sil_scores.append(silhouette_score(df2f, m.labels_))

            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=list(range(1, 11)), y=inertia, mode="lines+markers",
                                            line=dict(color=PALETTE[0], width=3), marker=dict(size=8)))
            fig_elbow.add_vline(x=k, line_dash="dash", line_color=PALETTE[1])
            fig_elbow.update_layout(title="Elbow Method", height=260, template="plotly_dark",
                                     margin=dict(l=10, r=10, t=40, b=10),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     xaxis_title="k", yaxis_title="Inertia")
            st.plotly_chart(fig_elbow, use_container_width=True)

            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(x=list(range(2, 11)), y=sil_scores, mode="lines+markers",
                                          line=dict(color=PALETTE[3], width=3), marker=dict(size=8)))
            fig_sil.add_vline(x=k, line_dash="dash", line_color=PALETTE[1])
            fig_sil.update_layout(title="Silhouette Score", height=260, template="plotly_dark",
                                   margin=dict(l=10, r=10, t=40, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   xaxis_title="k", yaxis_title="Score")
            st.plotly_chart(fig_sil, use_container_width=True)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_k = df.copy()
    df_k["Cluster"] = kmeans.fit_predict(df2f_view if len(df2f_view) == len(df_k) else df2f)

    with c2:
        with st.container(border=True):
            st.markdown(f"#### Clusters (k={k})")
            fig = px.scatter(df_k, x="Annual_Income", y="Spending_Score", color=df_k["Cluster"].astype(str),
                              color_discrete_sequence=PALETTE, size_max=10,
                              labels={"color": "Cluster"})
            centers = scaler.inverse_transform(
                np.column_stack([np.zeros(k), kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]])
            )
            fig.add_trace(go.Scatter(x=centers[:, 1], y=centers[:, 2], mode="markers",
                                      marker=dict(symbol="x", size=16, color="#F8FAFC", line=dict(width=2, color="#0B0E14")),
                                      name="Centroids"))
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Cluster Profiles")
        profile = df_k.groupby("Cluster")[["Age", "Annual_Income", "Spending_Score"]].mean().round(1)
        profile["Customers"] = df_k.groupby("Cluster").size()
        st.dataframe(profile.style.background_gradient(cmap="magma", axis=0), use_container_width=True)

    st.session_state["kmeans_model"] = kmeans
    st.session_state["kmeans_k"] = k

# ============================================================
# HIERARCHICAL
# ============================================================
elif page == "🌳 Hierarchical":
    hero("Agglomerative Hierarchical Clustering", "Connectivity-based segmentation using Ward linkage.")

    c1, c2 = st.columns([1, 2])
    with c1:
        with st.container(border=True):
            n_clusters_h = st.slider("Number of clusters", 2, 10, 5, key="hier_k")
            st.markdown("#### Dendrogram")
            linked = linkage(df2f, method="ward")
            fig = go.Figure()
            dendro = dendrogram(linked, no_plot=True)
            icoord, dcoord = np.array(dendro["icoord"]), np.array(dendro["dcoord"])
            for xs, ys in zip(icoord, dcoord):
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                          line=dict(color=PALETTE[0], width=1.4), showlegend=False))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="Customers", yaxis_title="Euclidean Distance")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Ward linkage minimizes the increase in within-cluster variance at each merge step.")

    hc = AgglomerativeClustering(n_clusters=n_clusters_h, linkage="ward")
    df_h = df.copy()
    df_h["Cluster"] = hc.fit_predict(df2f_view if len(df2f_view) == len(df_h) else df2f)

    with c2:
        with st.container(border=True):
            st.markdown(f"#### Clusters (n={n_clusters_h})")
            fig = px.scatter(df_h, x="Annual_Income", y="Spending_Score", color=df_h["Cluster"].astype(str),
                              color_discrete_sequence=PALETTE, labels={"color": "Cluster"})
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Cluster Profiles")
        profile = df_h.groupby("Cluster")[["Age", "Annual_Income", "Spending_Score"]].mean().round(1)
        profile["Customers"] = df_h.groupby("Cluster").size()
        st.dataframe(profile.style.background_gradient(cmap="magma", axis=0), use_container_width=True)

# ============================================================
# DBSCAN
# ============================================================
elif page == "🔍 DBSCAN":
    hero("DBSCAN Clustering", "Density-based segmentation that also detects noise / outlier customers.")

    c1, c2 = st.columns([1, 2])
    with c1:
        with st.container(border=True):
            st.markdown("#### ⚙️ Controls")
            eps = st.slider("eps", 0.1, 1.0, 0.4, 0.05)
            min_samples = st.slider("min_samples", 2, 10, 5)

            neighbors = NearestNeighbors(n_neighbors=4).fit(df2f)
            distances, _ = neighbors.kneighbors(df2f)
            distances = np.sort(distances[:, 3])
            fig_k = go.Figure()
            fig_k.add_trace(go.Scatter(y=distances, mode="lines", line=dict(color=PALETTE[2], width=3)))
            fig_k.add_hline(y=eps, line_dash="dash", line_color=PALETTE[1])
            fig_k.update_layout(title="4-NN Distance Plot", height=300, template="plotly_dark",
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="Points (sorted)", yaxis_title="Distance")
            st.plotly_chart(fig_k, use_container_width=True)
            st.caption("Pick eps where the curve bends sharply upward (the 'knee').")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_d = df.copy()
    labels = dbscan.fit_predict(df2f_view if len(df2f_view) == len(df_d) else df2f)
    df_d["Cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels) if len(labels) else 0

    with c2:
        with st.container(border=True):
            st.markdown(f"#### Clusters found: {n_clusters}  ·  Noise: {noise_pct:.1f}%")
            df_d["Cluster_str"] = df_d["Cluster"].astype(str).replace("-1", "Noise")
            cats = [c for c in df_d["Cluster_str"].unique() if c != "Noise"]
            color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(sorted(cats))}
            color_map["Noise"] = "#64748B"
            fig = px.scatter(df_d, x="Annual_Income", y="Spending_Score", color="Cluster_str",
                              color_discrete_map=color_map, labels={"Cluster_str": "Cluster"})
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    if n_clusters >= 2 and (labels != -1).sum() > n_clusters:
        mask_valid = labels != -1
        sil = silhouette_score(df2f[mask_valid.tolist() if len(mask_valid)==len(df2f) else mask_valid], labels[mask_valid])
        st.markdown(f"""
        <div class="insight-card" style="background:linear-gradient(120deg,#14B8A6,#3B82F6);">
            <div class="insight-title">📈 Silhouette Score: {sil:.3f}</div>
            <div class="insight-body">Higher is better. Try adjusting eps/min_samples to reduce noise while keeping the score high.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Not enough non-noise clusters to compute a silhouette score with current settings.")

# ============================================================
# COMPARE
# ============================================================
elif page == "⚖️ Compare Algorithms":
    hero("Algorithm Comparison", "K-Means vs Hierarchical vs DBSCAN — side by side, on the same data.")

    k = st.slider("k / n_clusters for K-Means & Hierarchical", 2, 10, 5)
    eps = st.slider("DBSCAN eps", 0.1, 1.0, 0.4, 0.05)
    min_samples = st.slider("DBSCAN min_samples", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df2f)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(df2f)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df2f)

    df_c = df.copy()
    df_c["KMeans"] = kmeans.labels_
    df_c["Hierarchical"] = hc.labels_
    df_c["DBSCAN"] = dbscan.labels_

    with st.container(border=True):
        fig = make_subplots(rows=1, cols=3, subplot_titles=("K-Means", "Hierarchical", "DBSCAN"))
        for i, algo in enumerate(["KMeans", "Hierarchical", "DBSCAN"]):
            labels = df_c[algo]
            for lbl in sorted(labels.unique()):
                sub = df_c[labels == lbl]
                color = "#64748B" if lbl == -1 else PALETTE[lbl % len(PALETTE)]
                name = "Noise" if lbl == -1 else f"C{lbl}"
                fig.add_trace(go.Scatter(x=sub.Annual_Income, y=sub.Spending_Score, mode="markers",
                                          marker=dict(color=color, size=7), name=name,
                                          legendgroup=f"{algo}{lbl}", showlegend=False),
                              row=1, col=i + 1)
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Metrics Summary")
        kmeans_sil = silhouette_score(df2f, kmeans.labels_)
        hier_sil = silhouette_score(df2f, hc.labels_)
        d_mask = dbscan.labels_ != -1
        dbscan_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        dbscan_sil = silhouette_score(df2f[d_mask], dbscan.labels_[d_mask]) if dbscan_clusters >= 2 else np.nan

        metrics = pd.DataFrame({
            "Algorithm": ["K-Means", "Hierarchical", "DBSCAN"],
            "No. of Clusters": [k, k, dbscan_clusters],
            "Silhouette Score": [round(kmeans_sil, 3), round(hier_sil, 3),
                                  round(dbscan_sil, 3) if not np.isnan(dbscan_sil) else "N/A"],
            "Handles Noise": ["No", "No", "Yes"],
            "Needs Cluster Count Upfront": ["Yes", "Yes", "No"],
        })
        st.dataframe(metrics.style.background_gradient(subset=["Silhouette Score"], cmap="magma"),
                     use_container_width=True, hide_index=True)

# ============================================================
# BUSINESS INSIGHTS
# ============================================================
elif page == "💡 Business Insights":
    hero("Business Insights", "Turning clusters into concrete actions for mall management.")

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(df2f)
    df_b = df.copy()
    df_b["Cluster"] = kmeans.labels_
    profile = df_b.groupby("Cluster")[["Age", "Annual_Income", "Spending_Score"]].mean()

    insight_defs = [
        ("💎", "High Income · High Spending", "Premium customers who contribute high revenue.",
         "Offer premium memberships and loyalty rewards.", "#7C3AED,#C026D3"),
        ("💼", "High Income · Low Spending", "High purchasing power but low engagement.",
         "Send personalized offers to encourage spending.", "#3B82F6,#14B8A6"),
        ("🏷️", "Low Income · High Spending", "Value-conscious shoppers who respond well to deals.",
         "Promote budget-friendly bundles and flash sales.", "#F59E0B,#EF4444"),
        ("🛒", "Low Income · Low Spending", "Occasional customers with limited purchases.",
         "Promote budget-friendly, essential products.", "#84CC16,#14B8A6"),
        ("🔁", "Average Income · Average Spending", "Regular customers with stable shopping habits.",
         "Offer seasonal discounts and loyalty points.", "#EC4899,#7C3AED"),
    ]

    cols = st.columns(len(insight_defs))
    for col, (icon, title, desc, action, grad) in zip(cols, insight_defs):
        with col:
            st.markdown(f"""
            <div class="insight-card" style="background:linear-gradient(135deg,{grad});min-height:230px;">
                <div style="font-size:1.6rem;">{icon}</div>
                <div class="insight-title">{title}</div>
                <div class="insight-body">{desc}</div>
                <hr style="border-color:rgba(255,255,255,0.3);margin:0.6rem 0;">
                <div class="insight-body"><b>Action:</b> {action}</div>
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### Segment Averages (K-Means, k=5)")
        profile_display = profile.copy().round(1)
        profile_display["Customers"] = df_b.groupby("Cluster").size()
        st.dataframe(profile_display.style.background_gradient(cmap="magma"), use_container_width=True)

# ============================================================
# PREDICT
# ============================================================
elif page == "🔮 Predict a Customer":
    hero("Predict a Customer's Segment", "Enter a hypothetical customer's profile and see which cluster they fall into.")

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(df2f)
    df_p = df.copy()
    df_p["Cluster"] = kmeans.labels_

    c1, c2 = st.columns([1, 1.6])
    with c1:
        with st.container(border=True):
            income_in = st.slider("Annual Income (k$)", int(df.Annual_Income.min()),
                                   int(df.Annual_Income.max()) + 20, 60)
            spend_in = st.slider("Spending Score (1-100)", 1, 100, 50)

    scaled_point = scaler.transform([[df.Age.mean(), income_in, spend_in]])[0][1:]
    pred_cluster = kmeans.predict([scaled_point])[0]

    cluster_names = {
        0: "Regular Shopper", 1: "Premium Customer", 2: "Budget Shopper",
        3: "High Potential (Low Engagement)", 4: "Deal Seeker"
    }
    label = cluster_names.get(pred_cluster, f"Cluster {pred_cluster}")

    with c2:
        with st.container(border=True):
            st.markdown(f"""
            <div class="insight-card" style="background:linear-gradient(120deg,#7C3AED,#EC4899);">
                <div class="insight-title">🎯 Predicted Segment: Cluster {pred_cluster} — {label}</div>
                <div class="insight-body">Based on Annual Income = {income_in}k and Spending Score = {spend_in}.</div>
            </div>
            """, unsafe_allow_html=True)

            fig = px.scatter(df_p, x="Annual_Income", y="Spending_Score", color=df_p["Cluster"].astype(str),
                              color_discrete_sequence=PALETTE, opacity=0.55, labels={"color": "Cluster"})
            fig.add_trace(go.Scatter(x=[income_in], y=[spend_in], mode="markers",
                                      marker=dict(color="#FDE047", size=20, symbol="star",
                                                  line=dict(width=1.5, color="#0B0E14")), name="New Customer"))
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
