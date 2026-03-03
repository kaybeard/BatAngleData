import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="MLB xSLG Explorer", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def standardize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    return s

def _key(x: str) -> str:
    return re.sub(r"[%\s_]", "", str(x).lower())

def find_col(cols, candidates):
    cols_keyed = {_key(c): c for c in cols}
    for cand in candidates:
        k = _key(cand)
        if k in cols_keyed:
            return cols_keyed[k]
    # fallback: substring match
    for c in cols:
        ck = _key(c)
        for cand in candidates:
            if _key(cand) in ck:
                return c
    return None

def to_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s
    s2 = s.astype(str).str.replace("%", "", regex=False)
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.strip()
    return pd.to_numeric(s2, errors="coerce")

def robust_zscores(x: pd.Series) -> pd.Series:
    x = x.dropna()
    if len(x) < 10:
        return pd.Series(dtype=float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(index=x.index, data=np.zeros(len(x)))
    return 0.6745 * (x - med) / mad

def scatter_fig(df_num: pd.DataFrame, xcol: str, ycol: str, title: str):
    x = df_num[xcol]
    y = df_num[ycol]
    m = x.notna() & y.notna()
    fig = plt.figure()
    plt.scatter(x[m], y[m])
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("MLB xSLG Explorer (Upload CSV)")
st.write("Upload your hitter CSV and explore correlations, trends, and outliers.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Load
df = pd.read_csv(uploaded)
df.columns = [standardize_name(c) for c in df.columns]

# Column detection
col_player = find_col(df.columns, ["player", "name", "batter", "hitter"])
col_xslg   = find_col(df.columns, ["xSLG", "x_slg", "expected slugging"])

if col_xslg is None:
    st.error("Couldn't find xSLG column. Rename it to 'xSLG' or adjust the candidates in the code.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Numeric copy
df_num = df.copy()
for c in df_num.columns:
    df_num[c] = to_numeric_series(df_num[c])

numeric_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()

# Sidebar controls
st.sidebar.header("Controls")
top_n_corr = st.sidebar.slider("How many top correlations to show", 5, 25, 12)
top_bucket = st.sidebar.slider("Top bucket size (by xSLG)", 5, 50, 15)
bottom_bucket = st.sidebar.slider("Bottom bucket size (by xSLG)", 5, 50, 15)
z_thresh = st.sidebar.slider("Outlier threshold (robust z)", 2.0, 5.0, 3.0, 0.1)

# Candidate fields you mentioned (best-effort matching)
col_ev   = find_col(df.columns, ["Exit Velo", "EV", "ExitVelocity", "Avg EV", "AvgExitVelo"])
col_wh   = find_col(df.columns, ["Well Hit %", "WellHit%", "Well Hit", "WellHit"])
col_gb   = find_col(df.columns, ["GB %", "GB%", "GroundBall%", "Ground Balls %", "GB"])
col_ss   = find_col(df.columns, ["Swing Speed", "SwingSpeed", "Bat Speed", "BatSpeed"])
col_la   = find_col(df.columns, ["Launch Angle", "LA", "LaunchAngle"])
col_hba  = find_col(df.columns, ["HBA", "Horizontal Bat Angle"])
col_vaa  = find_col(df.columns, ["VAA", "Vertical Attack Angle"])
col_vsa  = find_col(df.columns, ["VSA", "Vertical Swing Angle"])
col_vba  = find_col(df.columns, ["VBA", "Vertical Bat Angle"])
col_hfa  = find_col(df.columns, ["HFA", "Horizontal Fan Angle"])
col_cpt  = find_col(df.columns, ["Contact Point", "ContactPoint"])
col_con  = find_col(df.columns, ["Contact %", "Contact%", "ContactPct"])
col_iz   = find_col(df.columns, ["IZ Contact %", "In-Zone Contact %", "IZ Contact%", "InZoneContact%"])
col_ch   = find_col(df.columns, ["Chase %", "Chase%", "O-Swing%", "OSwing%"])
col_whf  = find_col(df.columns, ["Whiff %", "Whiff%", "SwingingStrike%"])
col_pull = find_col(df.columns, ["Pull %", "Pull%", "PullPct"])
col_rad  = find_col(df.columns, ["Swing Radius", "SwingRadius"])
col_tilt = find_col(df.columns, ["Tilt Change", "TiltChange"])

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_plots, tab_outliers, tab_model, tab_data = st.tabs(
    ["Overview", "Plots", "Outliers", "Model", "Data Preview"]
)

with tab_overview:
    st.subheader("Quick Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Numeric cols", f"{len(numeric_cols):,}")
    c3.metric("xSLG column", col_xslg)

    st.write("Detected key columns:")
    detected = {
        "Player": col_player,
        "xSLG": col_xslg,
        "Exit Velo": col_ev,
        "Well Hit %": col_wh,
        "GB %": col_gb,
        "Swing Speed": col_ss,
        "LA": col_la,
        "HBA": col_hba,
        "VAA": col_vaa,
        "VSA": col_vsa,
        "VBA": col_vba,
        "HFA": col_hfa,
        "Contact %": col_con,
        "IZ Contact %": col_iz,
        "Chase %": col_ch,
        "Whiff %": col_whf,
        "Pull %": col_pull,
        "Swing Radius": col_rad,
        "Contact Point": col_cpt,
        "Tilt Change": col_tilt,
    }
    st.dataframe(pd.DataFrame([detected]).T.rename(columns={0: "Column Name"}))

    # Correlations
    st.subheader("Correlations vs xSLG")
    corrs = df_num[numeric_cols].corr(numeric_only=True)[col_xslg].dropna().sort_values(ascending=False)
    corrs_show = pd.concat([corrs.head(top_n_corr), corrs.tail(top_n_corr)]).drop_duplicates()
    st.dataframe(corrs_show.rename("Pearson r").to_frame())

    fig = plt.figure()
    corrs_show.sort_values().plot(kind="barh")
    plt.title(f"Correlation with {col_xslg} (Top +/-)")
    plt.xlabel("Pearson r")
    plt.tight_layout()
    st.pyplot(fig)

with tab_plots:
    st.subheader("Scatter Plots (xSLG relationships)")

    candidates = [
        col_ev, col_wh, col_gb, col_ss, col_hba, col_vaa, col_la,
        col_vsa, col_vba, col_hfa, col_rad, col_cpt, col_tilt,
        col_con, col_iz, col_ch, col_whf, col_pull
    ]
    candidates = [c for c in candidates if c is not None and c in df_num.columns]

    if not candidates:
        st.warning("No candidate columns detected for plotting.")
    else:
        xcol = st.selectbox("Choose X variable", candidates, index=0)
        fig = scatter_fig(df_num, xcol, col_xslg, f"{col_xslg} vs {xcol}")
        st.pyplot(fig)

        st.divider()
        st.write("Quick set of common plots:")
        quick = [c for c in [col_ev, col_wh, col_gb, col_ss, col_hba, col_vaa, col_la] if c is not None]
        cols = st.columns(2)
        for i, qc in enumerate(quick):
            with cols[i % 2]:
                fig = scatter_fig(df_num, qc, col_xslg, f"{col_xslg} vs {qc}")
                st.pyplot(fig)

    st.divider()
    st.subheader("Top vs Bottom xSLG Buckets (boxplots)")
    metrics = [c for c in [col_ev, col_wh, col_gb, col_ss, col_hba, col_vaa, col_la, col_vsa, col_vba, col_hfa] if c is not None]
    metrics = [m for m in metrics if m in df_num.columns]

    if metrics:
        df_sorted = df_num.sort_values(col_xslg, ascending=False)
        top = df_sorted.head(top_bucket)
        bot = df_sorted.tail(bottom_bucket)

        chosen_metrics = st.multiselect("Metrics to compare", metrics, default=metrics[:min(7, len(metrics))])
        if chosen_metrics:
            data = []
            labels = []
            for m in chosen_metrics:
                data.append(top[m].dropna().values); labels.append(f"Top {top_bucket} {m}")
                data.append(bot[m].dropna().values); labels.append(f"Bot {bottom_bucket} {m}")

            fig = plt.figure(figsize=(10, max(4, 0.35 * len(labels))))
            plt.boxplot(data, labels=labels, vert=False)
            plt.title(f"Top {top_bucket} vs Bottom {bottom_bucket} by {col_xslg}")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No metrics detected for bucket boxplots.")

with tab_outliers:
    st.subheader("Outliers (robust z-score)")

    outlier_metrics = [c for c in [
        col_ev, col_wh, col_gb, col_ss, col_hba, col_vaa, col_la,
        col_vsa, col_vba, col_hfa, col_rad, col_cpt, col_tilt,
        col_con, col_iz, col_ch, col_whf, col_pull
    ] if c is not None and c in df_num.columns]

    if not outlier_metrics:
        st.warning("No outlier metrics detected.")
    else:
        chosen = st.multiselect("Metrics to scan for outliers", outlier_metrics, default=outlier_metrics[:min(7, len(outlier_metrics))])
        name_series = df[col_player] if col_player is not None else pd.Series(df.index.astype(str), name="row")

        all_outliers = []
        for m in chosen:
            z = robust_zscores(df_num[m])
            flagged = z[abs(z) >= z_thresh].sort_values()
            if flagged.empty:
                continue
            show = pd.DataFrame({
                "player": name_series.loc[flagged.index].values,
                "metric": m,
                "value": df_num.loc[flagged.index, m].values,
                "robust_z": flagged.values
            }).sort_values("robust_z")
            all_outliers.append(show)

        if not all_outliers:
            st.info("No outliers found at the selected threshold.")
        else:
            out_df = pd.concat(all_outliers, ignore_index=True).sort_values(["metric", "robust_z"])
            st.dataframe(out_df, use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download outliers as CSV",
                data=csv_bytes,
                file_name="outliers_robust_z.csv",
                mime="text/csv"
            )

with tab_model:
    st.subheader("Quick Linear Model for xSLG")
    default_feats = [c for c in [col_ev, col_wh, col_gb, col_ss, col_la, col_hba, col_vaa] if c is not None and c in df_num.columns]
    all_feats = [c for c in numeric_cols if c != col_xslg]

    feats = st.multiselect("Model features", all_feats, default=default_feats)

    if len(feats) < 2:
        st.info("Pick at least 2 features to fit a model.")
    else:
        model_df = df_num[[col_xslg] + feats].dropna()
        X = model_df[feats].values
        y = model_df[col_xslg].values

        lr = LinearRegression().fit(X, y)
        pred = lr.predict(X)
        resid = y - pred

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows used", f"{len(model_df):,}")
        c2.metric("R²", f"{r2_score(y, pred):.3f}")
        c3.metric("MAE", f"{mean_absolute_error(y, pred):.4f}")

        # Actual vs Pred
        fig1 = plt.figure()
        plt.scatter(pred, y)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Actual vs Predicted {col_xslg}")
        plt.tight_layout()
        st.pyplot(fig1)

        # Residuals
        fig2 = plt.figure()
        plt.scatter(pred, resid)
        plt.axhline(0)
        plt.xlabel("Predicted")
        plt.ylabel("Residual (Actual - Pred)")
        plt.title("Residuals vs Predicted")
        plt.tight_layout()
        st.pyplot(fig2)

        # Residual table
        name_series = df[col_player] if col_player is not None else pd.Series(model_df.index.astype(str), name="row")
        resid_series = pd.Series(resid, index=model_df.index)

        top_pos = resid_series.sort_values(ascending=False).head(10)
        top_neg = resid_series.sort_values(ascending=True).head(10)

        st.write("Top + residuals (higher xSLG than model expects):")
        st.dataframe(pd.DataFrame({
            "player": name_series.loc[top_pos.index].values,
            "residual": top_pos.values
        }))

        st.write("Top - residuals (lower xSLG than model expects):")
        st.dataframe(pd.DataFrame({
            "player": name_series.loc[top_neg.index].values,
            "residual": top_neg.values
        }))

with tab_data:
    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)