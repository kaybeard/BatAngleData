import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Bat Angle Explorer v2", layout="wide")


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
    # substring fallback
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


def read_csv_with_retries(uploaded_file):
    """
    Streamlit UploadedFile is file-like; must seek(0) before each read.
    Tries different skiprows and delimiter fallbacks.
    """
    attempts = [
        {"skiprows": None, "sep": None},
        {"skiprows": 1, "sep": None},
        {"skiprows": 2, "sep": None},
        {"skiprows": 3, "sep": None},
        {"skiprows": None, "sep": ";"},
        {"skiprows": 1, "sep": ";"},
        {"skiprows": 2, "sep": ";"},
    ]

    last_err = None
    for a in attempts:
        try:
            uploaded_file.seek(0)
            kwargs = {}
            if a["skiprows"] is not None:
                kwargs["skiprows"] = a["skiprows"]
            if a["sep"] is not None:
                kwargs["sep"] = a["sep"]

            df_try = pd.read_csv(uploaded_file, **kwargs)

            # empty / bad header checks
            if df_try is None or (df_try.shape[0] == 0 and df_try.shape[1] == 0):
                continue
            if len(df_try.columns) > 0 and all("Unnamed" in str(c) for c in df_try.columns):
                continue

            return df_try, a
        except Exception as e:
            last_err = e
            continue

    raise last_err if last_err is not None else ValueError("Could not read CSV with retries.")


def percentile_rank(series: pd.Series, value):
    """Percentile rank (0-100)."""
    s = series.dropna()
    if len(s) < 5 or pd.isna(value):
        return np.nan
    return 100.0 * (s <= value).mean()


def build_hover_text(df: pd.DataFrame, player_col: str, cols_for_hover):
    """
    Create rich hover text for plotly.
    """
    cols_for_hover = [c for c in cols_for_hover if c is not None and c in df.columns]
    parts = []
    parts.append(df[player_col].astype(str) if player_col else df.index.astype(str))

    for c in cols_for_hover:
        vals = df[c]
        # keep original formatting if possible
        parts.append(c + ": " + vals.astype(str))

    # join with <br> for plotly
    hover = parts[0]
    for p in parts[1:]:
        hover = hover + "<br>" + p
    return hover


# -----------------------------
# UI
# -----------------------------
st.title("Bat Angle Explorer v2 (Upload CSV)")
st.write("Version 2: Hover tooltips, Player Search, Bat Plane + Damage Zone views, outliers, and quick modeling.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Load CSV robustly
try:
    df_raw, used_params = read_csv_with_retries(uploaded)
except Exception as e:
    st.error("Could not read the CSV. It may not be a standard CSV or has an unusual format.")
    st.exception(e)
    st.stop()

df_raw.columns = [standardize_name(c) for c in df_raw.columns]

# Detect key columns
col_player = find_col(df_raw.columns, ["player", "name", "batter", "hitter"])
col_xslg   = find_col(df_raw.columns, ["xSLG", "x_slg", "expected slugging", "xslg"])
col_ev     = find_col(df_raw.columns, ["Exit Velo", "EV", "ExitVelocity", "Avg EV", "AvgExitVelo"])
col_wh     = find_col(df_raw.columns, ["Well Hit %", "WellHit%", "Well Hit", "WellHit"])
col_gb     = find_col(df_raw.columns, ["GB %", "GB%", "GroundBall%", "Ground Balls %", "GB"])
col_ss     = find_col(df_raw.columns, ["Swing Speed", "SwingSpeed", "Bat Speed", "BatSpeed"])
col_la     = find_col(df_raw.columns, ["Launch Angle", "LA", "LaunchAngle"])
col_hba    = find_col(df_raw.columns, ["HBA", "Horizontal Bat Angle"])
col_vaa    = find_col(df_raw.columns, ["VAA", "Vertical Attack Angle"])
col_vsa    = find_col(df_raw.columns, ["VSA", "Vertical Swing Angle"])
col_vba    = find_col(df_raw.columns, ["VBA", "Vertical Bat Angle"])
col_hfa    = find_col(df_raw.columns, ["HFA", "Horizontal Fan Angle"])
col_hba2   = col_hba  # alias for clarity
col_rad    = find_col(df_raw.columns, ["Swing Radius", "SwingRadius"])
col_cpt    = find_col(df_raw.columns, ["Contact Point", "ContactPoint"])
col_tilt   = find_col(df_raw.columns, ["Tilt Change", "TiltChange"])
col_con    = find_col(df_raw.columns, ["Contact %", "Contact%", "ContactPct"])
col_iz     = find_col(df_raw.columns, ["IZ Contact %", "In-Zone Contact %", "IZ Contact%", "InZoneContact%"])
col_chase  = find_col(df_raw.columns, ["Chase %", "Chase%", "O-Swing%", "OSwing%"])
col_whiff  = find_col(df_raw.columns, ["Whiff %", "Whiff%", "SwStr%", "SwingingStrike%"])
col_pull   = find_col(df_raw.columns, ["Pull %", "Pull%", "PullPct"])

# Convert numeric copy
df = df_raw.copy()
for c in df.columns:
    df[c] = to_numeric_series(df[c])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# If player col missing, create one
if col_player is None:
    col_player = "__row__"
    df_raw[col_player] = df_raw.index.astype(str)
    df[col_player] = df.index.astype(str)

# Sidebar controls
st.sidebar.header("Controls")
st.sidebar.caption("Tip: hover any point to see player name + metrics.")
min_rows = st.sidebar.slider("Minimum non-null rows required for a plot", 20, 200, 60, 5)
z_thresh = st.sidebar.slider("Outlier threshold (robust z)", 2.0, 5.0, 3.0, 0.1)
top_n_corr = st.sidebar.slider("Top correlations to show", 5, 30, 12)

# Optional filter: keep only rows with player name
df_plot = df.copy()
df_plot_raw = df_raw.copy()

# Tabs (new structure)
tab_overview, tab_batplane, tab_damage, tab_tradeoff, tab_outliers, tab_player, tab_model, tab_data = st.tabs(
    ["Overview", "Bat Plane", "Damage Zone", "Swing Tradeoff", "Outliers", "Player Search", "Model", "Data"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df_plot):,}")
    c2.metric("Numeric cols", f"{len(numeric_cols):,}")
    c3.metric("CSV params", f"{used_params}")
    c4.metric("Player column", col_player)

    st.subheader("Detected Columns (key)")
    detected = {
        "xSLG": col_xslg,
        "Exit Velo": col_ev,
        "Well Hit %": col_wh,
        "GB %": col_gb,
        "Swing Speed": col_ss,
        "Launch Angle": col_la,
        "HBA": col_hba,
        "VAA": col_vaa,
        "VSA": col_vsa,
        "VBA": col_vba,
        "HFA": col_hfa,
        "Contact %": col_con,
        "IZ Contact %": col_iz,
        "Chase %": col_chase,
        "Whiff %": col_whiff,
        "Pull %": col_pull,
        "Swing Radius": col_rad,
        "Contact Point": col_cpt,
        "Tilt Change": col_tilt,
    }
    st.dataframe(pd.DataFrame([detected]).T.rename(columns={0: "Column Name"}), use_container_width=True)

    st.subheader("Correlations (choose target)")
    target_default = col_ev or col_wh or col_xslg or (numeric_cols[0] if numeric_cols else None)
    target = st.selectbox(
        "Target variable for correlations (not necessarily xSLG)",
        [c for c in numeric_cols] if numeric_cols else [],
        index=(numeric_cols.index(target_default) if (target_default in numeric_cols) else 0) if numeric_cols else 0
    )

    if target:
        corrs = df_plot[numeric_cols].corr(numeric_only=True)[target].dropna().sort_values(ascending=False)
        corrs_show = pd.concat([corrs.head(top_n_corr), corrs.tail(top_n_corr)]).drop_duplicates()
        st.dataframe(corrs_show.rename("Pearson r").to_frame(), use_container_width=True)

        fig = px.bar(
            corrs_show.sort_values(),
            orientation="h",
            title=f"Correlation with {target} (Top +/-)",
            labels={"value": "Pearson r", "index": "Metric"},
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Bat Plane (HBA vs VAA) + hover
# -----------------------------
with tab_batplane:
    st.subheader("Bat Plane Map (HBA vs VAA)")
    st.caption("Best for understanding swing direction + attack angle. Hover points to see the player.")

    if col_hba is None or col_vaa is None:
        st.warning("Could not find HBA and/or VAA columns in your CSV.")
    else:
        color_candidates = [c for c in [col_ev, col_wh, col_xslg, col_la, col_ss, col_gb, col_con, col_whiff, col_pull] if c in df_plot.columns and c is not None]
        color_by = st.selectbox("Color points by", color_candidates, index=0 if color_candidates else 0)

        size_candidates = [None, col_ev, col_wh, col_ss]
        size_candidates = [c for c in size_candidates if c is None or (c in df_plot.columns)]
        size_by = st.selectbox("Point size (optional)", size_candidates, index=0)

        plot_df = df_plot[[col_player, col_hba, col_vaa] + ([color_by] if color_by else []) + ([size_by] if size_by else [])].dropna()
        if len(plot_df) < min_rows:
            st.info(f"Not enough rows for a reliable plot after dropping NAs. Rows available: {len(plot_df)}.")
        else:
            hover_cols = [color_by, col_ev, col_wh, col_la, col_ss, col_gb]
            plot_df["hover"] = build_hover_text(plot_df, col_player, hover_cols)

            fig = px.scatter(
                plot_df,
                x=col_hba,
                y=col_vaa,
                color=color_by if color_by else None,
                size=size_by if (size_by is not None) else None,
                hover_name=col_player,
                hover_data={"hover": True, col_player: False},
                title="Bat Plane Map (Hover for player + metrics)",
            )
            fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Damage Zone (EV vs LA) + hover
# -----------------------------
with tab_damage:
    st.subheader("Damage Zone: Exit Velo vs Launch Angle")
    st.caption("If you have EV and LA, this is the cleanest ‘contact quality’ view. Hover for player details.")

    if col_ev is None or col_la is None:
        st.warning("Could not find Exit Velo and/or Launch Angle (LA) columns in your CSV.")
    else:
        color_candidates = [c for c in [col_wh, col_xslg, col_pull, col_gb] if c in df_plot.columns and c is not None]
        color_by = st.selectbox("Color by", color_candidates, index=0 if color_candidates else 0)

        plot_df = df_plot[[col_player, col_ev, col_la] + ([color_by] if color_by else [])].dropna()
        if len(plot_df) < min_rows:
            st.info(f"Not enough rows for a reliable plot after dropping NAs. Rows available: {len(plot_df)}.")
        else:
            hover_cols = [color_by, col_wh, col_xslg, col_pull, col_gb]
            plot_df["hover"] = build_hover_text(plot_df, col_player, hover_cols)

            fig = px.scatter(
                plot_df,
                x=col_la,
                y=col_ev,
                color=color_by if color_by else None,
                hover_name=col_player,
                hover_data={"hover": True, col_player: False},
                title="EV vs LA (Hover for player + metrics)",
            )
            fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Swing Tradeoff (speed vs contact / whiff / chase)
# -----------------------------
with tab_tradeoff:
    st.subheader("Swing Speed Tradeoffs")
    st.caption("These plots help identify hitters who keep contact while adding speed (and the ones who don’t).")

    if col_ss is None:
        st.warning("Could not find Swing Speed column in your CSV.")
    else:
        y_opts = [c for c in [col_con, col_iz, col_whiff, col_chase, col_ev, col_wh] if c is not None and c in df_plot.columns]
        if not y_opts:
            st.warning("No tradeoff outcome columns detected (Contact%, IZ Contact%, Whiff%, Chase%, EV, Well Hit%).")
        else:
            ycol = st.selectbox("Choose Y variable", y_opts, index=0)
            color_candidates = [c for c in [col_ev, col_wh, col_xslg, col_la, col_pull, col_gb] if c is not None and c in df_plot.columns]
            color_by = st.selectbox("Color by (optional)", [None] + color_candidates, index=0)

            plot_df = df_plot[[col_player, col_ss, ycol] + ([color_by] if color_by else [])].dropna()
            if len(plot_df) < min_rows:
                st.info(f"Not enough rows for a reliable plot after dropping NAs. Rows available: {len(plot_df)}.")
            else:
                hover_cols = [ycol, color_by, col_ev, col_wh, col_con, col_whiff]
                plot_df["hover"] = build_hover_text(plot_df, col_player, hover_cols)

                fig = px.scatter(
                    plot_df,
                    x=col_ss,
                    y=ycol,
                    color=color_by if color_by else None,
                    hover_name=col_player,
                    hover_data={"hover": True, col_player: False},
                    title=f"{ycol} vs {col_ss} (Hover for player + metrics)",
                )
                fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Outliers (robust z) — not xSLG-centered
# -----------------------------
with tab_outliers:
    st.subheader("Outliers")
    st.caption("Outliers are detected per-metric using robust z-scores (MAD). Great for finding unusual mechanical profiles.")

    metric_candidates = [c for c in [
        col_ev, col_wh, col_gb, col_ss, col_la, col_hba, col_vaa, col_vsa, col_vba, col_hfa,
        col_pull, col_con, col_iz, col_whiff, col_chase, col_rad, col_cpt, col_tilt, col_xslg
    ] if c is not None and c in df_plot.columns]

    if not metric_candidates:
        st.warning("No recognizable numeric metrics found for outlier detection.")
    else:
        chosen = st.multiselect("Metrics to scan", metric_candidates, default=metric_candidates[:min(8, len(metric_candidates))])

        out_rows = []
        for m in chosen:
            z = robust_zscores(df_plot[m])
            flagged = z[abs(z) >= z_thresh].sort_values()
            for idx, zval in flagged.items():
                out_rows.append({
                    "player": df_raw.loc[idx, col_player],
                    "metric": m,
                    "value": df_plot.loc[idx, m],
                    "robust_z": zval
                })

        if not out_rows:
            st.info("No outliers found at the current threshold.")
        else:
            out_df = pd.DataFrame(out_rows).sort_values(["metric", "robust_z"])
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download outliers CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="outliers_v2.csv",
                mime="text/csv"
            )

# -----------------------------
# Player Search + Player Card (percentiles)
# -----------------------------
with tab_player:
    st.subheader("Player Search")
    st.caption("Select a player to view a simple scouting card + percentiles across key metrics.")

    players = df_raw[col_player].astype(str).fillna("").tolist()
    players_sorted = sorted(players)

    selected = st.selectbox("Choose player", players_sorted)

    row_idx = df_raw.index[df_raw[col_player].astype(str) == selected]
    if len(row_idx) == 0:
        st.warning("Player not found.")
    else:
        idx = row_idx[0]
        st.markdown(f"### {selected}")

        # Choose metrics for the player card
        card_metrics = [c for c in [
            col_ev, col_wh, col_la, col_gb, col_pull, col_ss,
            col_hba, col_vaa, col_vsa, col_vba, col_hfa,
            col_con, col_iz, col_whiff, col_chase,
            col_rad, col_cpt, col_tilt, col_xslg
        ] if c is not None and c in df_plot.columns]

        # Show raw values + percentiles
        rows = []
        for m in card_metrics:
            val = df_plot.loc[idx, m]
            pct = percentile_rank(df_plot[m], val)
            rows.append({
                "metric": m,
                "value": val,
                "percentile": None if np.isnan(pct) else round(pct, 1)
            })
        card = pd.DataFrame(rows)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**Player Metrics (raw)**")
            st.dataframe(card[["metric", "value"]], use_container_width=True)
        with c2:
            st.write("**Percentiles (0–100)**")
            st.dataframe(card[["metric", "percentile"]], use_container_width=True)

        st.divider()
        st.subheader("Player on key plots")

        # Bat plane highlight
        if col_hba and col_vaa and (not pd.isna(df_plot.loc[idx, col_hba])) and (not pd.isna(df_plot.loc[idx, col_vaa])):
            plane_df = df_plot[[col_player, col_hba, col_vaa]].dropna()
            plane_df["is_selected"] = (plane_df[col_player].astype(str) == selected).astype(int)

            fig = px.scatter(
                plane_df,
                x=col_hba,
                y=col_vaa,
                color="is_selected",
                hover_name=col_player,
                title="Bat Plane (selected player highlighted)",
                labels={"is_selected": "Selected"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Damage zone highlight
        if col_ev and col_la and (not pd.isna(df_plot.loc[idx, col_ev])) and (not pd.isna(df_plot.loc[idx, col_la])):
            dmg_df = df_plot[[col_player, col_la, col_ev]].dropna()
            dmg_df["is_selected"] = (dmg_df[col_player].astype(str) == selected).astype(int)

            fig = px.scatter(
                dmg_df,
                x=col_la,
                y=col_ev,
                color="is_selected",
                hover_name=col_player,
                title="EV vs LA (selected player highlighted)",
                labels={"is_selected": "Selected"},
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Model (optional; not centered on xSLG)
# -----------------------------
with tab_model:
    st.subheader("Quick Model (choose target)")
    st.caption("Use this for rough directional insight, not as a final predictive model.")

    if not numeric_cols:
        st.warning("No numeric columns detected.")
    else:
        target = st.selectbox("Target variable", numeric_cols, index=(numeric_cols.index(col_ev) if col_ev in numeric_cols else 0))
        features_all = [c for c in numeric_cols if c != target]

        # sensible defaults
        default_feats = [c for c in [col_ss, col_ev, col_wh, col_la, col_gb, col_hba, col_vaa, col_con, col_whiff, col_chase, col_pull] if c in features_all]
        feats = st.multiselect("Features", features_all, default=default_feats)

        if len(feats) < 2:
            st.info("Pick at least 2 features.")
        else:
            model_df = df_plot[[target] + feats].dropna()
            if len(model_df) < 50:
                st.warning(f"Only {len(model_df)} rows available after dropping NAs. Results may be noisy.")
            X = model_df[feats].values
            y = model_df[target].values

            lr = LinearRegression().fit(X, y)
            pred = lr.predict(X)
            resid = y - pred

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows used", f"{len(model_df):,}")
            c2.metric("R²", f"{r2_score(y, pred):.3f}")
            c3.metric("MAE", f"{mean_absolute_error(y, pred):.4f}")

            fig1 = px.scatter(x=pred, y=y, labels={"x": "Predicted", "y": "Actual"}, title="Actual vs Predicted")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.scatter(x=pred, y=resid, labels={"x": "Predicted", "y": "Residual"}, title="Residuals vs Predicted")
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Data
# -----------------------------
with tab_data:
    st.subheader("Data Preview")
    st.write("First 50 rows:")
    st.dataframe(df_raw.head(50), use_container_width=True)

    st.write("Column list:")
    st.code("\n".join(list(df_raw.columns)))