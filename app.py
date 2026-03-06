import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Bat Angle Explorer v3", layout="wide")


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
    s = series.dropna()
    if len(s) < 5 or pd.isna(value):
        return np.nan
    return 100.0 * (s <= value).mean()


def add_range_filter_ui(df_in: pd.DataFrame, label: str, col: str, key: str):
    if col is None or col not in df_in.columns:
        return None
    s = df_in[col].dropna()
    if s.empty:
        return None
    lo = float(np.nanmin(s))
    hi = float(np.nanmax(s))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    return st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi), key=key)


def apply_range(df_in, col, rng):
    if rng is None or col is None or col not in df_in.columns:
        return df_in
    lo, hi = rng
    return df_in[(df_in[col].isna()) | ((df_in[col] >= lo) & (df_in[col] <= hi))]


def make_hover_fields(df_plot, fields):
    return [c for c in fields if c is not None and c in df_plot.columns]


def build_customdata(df_plot, fields):
    return df_plot[fields].to_numpy()


def build_hovertemplate(player_col, fields):
    lines = [f"<b>%{{customdata[0]}}</b>"]
    for i, field in enumerate(fields[1:], start=1):
        lines.append(f"{field}: %{{customdata[{i}]}}")
    lines.append("<extra></extra>")
    return "<br>".join(lines)


def archetype_name(row, col_ss, col_con, col_whiff, col_pull, col_gb, col_hba, col_vaa):
    parts = []

    if col_ss and pd.notna(row.get(col_ss)):
        if row[col_ss] >= 75:
            parts.append("High-Speed")
        elif row[col_ss] <= 68:
            parts.append("Lower-Speed")

    if col_con and pd.notna(row.get(col_con)):
        if row[col_con] >= 80:
            parts.append("Contact")
        elif row[col_con] <= 72:
            parts.append("Miss")

    if col_pull and pd.notna(row.get(col_pull)):
        if row[col_pull] >= 45:
            parts.append("Pull")
        elif row[col_pull] <= 35:
            parts.append("All-Fields")

    if col_gb and pd.notna(row.get(col_gb)):
        if row[col_gb] >= 48:
            parts.append("Ground-Ball")
        elif row[col_gb] <= 38:
            parts.append("Air-Ball")

    if col_hba and pd.notna(row.get(col_hba)) and col_vaa and pd.notna(row.get(col_vaa)):
        if row[col_hba] >= 20 and row[col_vaa] >= 10:
            parts.append("Lift")
        elif row[col_hba] <= 10 and row[col_vaa] <= 6:
            parts.append("Flat-Plane")

    if not parts:
        return "Balanced"
    return " / ".join(parts[:3])


# -----------------------------
# UI
# -----------------------------
st.title("Bat Angle Explorer v3")
st.write("Version 3: filters, hover tooltips, player highlight mode, VSA-LA view, archetypes, and comps finder.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# -----------------------------
# Load CSV
# -----------------------------
try:
    df_raw, used_params = read_csv_with_retries(uploaded)
except Exception as e:
    st.error("Could not read the CSV.")
    st.exception(e)
    st.stop()

df_raw.columns = [standardize_name(c) for c in df_raw.columns]

# Detect columns
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
col_rad    = find_col(df_raw.columns, ["Swing Radius", "SwingRadius"])
col_cpt    = find_col(df_raw.columns, ["Contact Point", "ContactPoint"])
col_tilt   = find_col(df_raw.columns, ["Tilt Change", "TiltChange"])
col_con    = find_col(df_raw.columns, ["Contact %", "Contact%", "ContactPct"])
col_iz     = find_col(df_raw.columns, ["IZ Contact %", "In-Zone Contact %", "IZ Contact%", "InZoneContact%"])
col_chase  = find_col(df_raw.columns, ["Chase %", "Chase%", "O-Swing%", "OSwing%"])
col_whiff  = find_col(df_raw.columns, ["Whiff %", "Whiff%", "SwStr%", "SwingingStrike%"])
col_pull   = find_col(df_raw.columns, ["Pull %", "Pull%", "PullPct"])

if col_player is None:
    col_player = "__row__"
    df_raw[col_player] = df_raw.index.astype(str)

# Numeric copy
df = df_raw.copy()
for c in df.columns:
    if c == col_player:
        continue
    df[c] = to_numeric_series(df[c])
df[col_player] = df_raw[col_player].astype(str)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
min_rows = st.sidebar.slider("Minimum non-null rows required for a plot", 10, 200, 40, 5)
z_thresh = st.sidebar.slider("Outlier threshold (robust z)", 2.0, 5.0, 3.0, 0.1)
top_n_corr = st.sidebar.slider("Top correlations to show", 5, 30, 12)
n_clusters = st.sidebar.slider("Number of archetype clusters", 3, 8, 5, 1)
n_comps = st.sidebar.slider("Number of comps to return", 3, 10, 5, 1)

st.sidebar.divider()
st.sidebar.header("Filters")

f_vsa  = add_range_filter_ui(df, "VSA range", col_vsa, "f_vsa")
f_vba  = add_range_filter_ui(df, "VBA range", col_vba, "f_vba")
f_vaa  = add_range_filter_ui(df, "VAA range", col_vaa, "f_vaa")
f_hfa  = add_range_filter_ui(df, "HFA range", col_hfa, "f_hfa")
f_hba  = add_range_filter_ui(df, "HBA range", col_hba, "f_hba")
f_rad  = add_range_filter_ui(df, "Swing Radius range", col_rad, "f_rad")
f_tilt = add_range_filter_ui(df, "Tilt Change range", col_tilt, "f_tilt")
f_ss   = add_range_filter_ui(df, "Swing Speed range", col_ss, "f_ss")
f_ev   = add_range_filter_ui(df, "Exit Velo range", col_ev, "f_ev")
f_la   = add_range_filter_ui(df, "Launch Angle range", col_la, "f_la")
f_con  = add_range_filter_ui(df, "Contact % range", col_con, "f_con")
f_whf  = add_range_filter_ui(df, "Whiff % range", col_whiff, "f_whf")

df_filt = df.copy()
for col, rng in [
    (col_vsa, f_vsa), (col_vba, f_vba), (col_vaa, f_vaa), (col_hfa, f_hfa),
    (col_hba, f_hba), (col_rad, f_rad), (col_tilt, f_tilt), (col_ss, f_ss),
    (col_ev, f_ev), (col_la, f_la), (col_con, f_con), (col_whiff, f_whf)
]:
    df_filt = apply_range(df_filt, col, rng)

st.sidebar.caption(f"Rows after filters: {len(df_filt):,} / {len(df):,}")

highlight_candidates = sorted(df_filt[col_player].dropna().astype(str).unique().tolist())
highlight_player = st.sidebar.selectbox("Highlight player on plots", ["None"] + highlight_candidates)

# -----------------------------
# Archetypes / comps prep
# -----------------------------
cluster_features = [c for c in [col_vsa, col_vba, col_vaa, col_hfa, col_hba, col_rad, col_tilt, col_ss, col_con, col_whiff, col_pull, col_gb] if c is not None and c in df_filt.columns]

df_cluster = df_filt.copy()
if len(cluster_features) >= 3:
    cluster_base = df_cluster[[col_player] + cluster_features].dropna().copy()
else:
    cluster_base = pd.DataFrame()

if not cluster_base.empty and len(cluster_base) >= n_clusters:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_base[cluster_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_base["cluster_id"] = kmeans.fit_predict(X_scaled)

    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids, columns=cluster_features)

    centroid_df["archetype_label"] = centroid_df.apply(
        lambda row: archetype_name(row, col_ss, col_con, col_whiff, col_pull, col_gb, col_hba, col_vaa),
        axis=1
    )

    cluster_base = cluster_base.merge(
        centroid_df[["archetype_label"]].reset_index().rename(columns={"index": "cluster_id"}),
        on="cluster_id",
        how="left"
    )

    df_filt = df_filt.merge(
        cluster_base[[col_player, "cluster_id", "archetype_label"]],
        on=col_player,
        how="left"
    )
else:
    df_filt["cluster_id"] = np.nan
    df_filt["archetype_label"] = np.nan

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Overview", "Bat Plane", "Damage Zone", "VSA vs LA", "Swing Tradeoff",
    "Archetypes", "Outliers", "Player Search", "Comps Finder", "Model", "Data"
])

tab_overview, tab_batplane, tab_damage, tab_vsala, tab_tradeoff, tab_archetypes, tab_outliers, tab_player, tab_comps, tab_model, tab_data = tabs

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (filtered)", f"{len(df_filt):,}")
    c2.metric("Rows (total)", f"{len(df):,}")
    c3.metric("Numeric cols", f"{len(numeric_cols):,}")
    c4.metric("CSV params", f"{used_params}")

    st.subheader("Detected Columns")
    detected = {
        "Player": col_player,
        "VSA": col_vsa,
        "VBA": col_vba,
        "VAA": col_vaa,
        "HFA": col_hfa,
        "HBA": col_hba,
        "Swing Radius": col_rad,
        "Tilt Change": col_tilt,
        "Swing Speed": col_ss,
        "Exit Velo": col_ev,
        "Launch Angle": col_la,
        "Contact %": col_con,
        "Whiff %": col_whiff,
        "Pull %": col_pull,
        "GB %": col_gb,
        "Well Hit %": col_wh,
        "xSLG": col_xslg,
    }
    st.dataframe(pd.DataFrame([detected]).T.rename(columns={0: "Column Name"}), use_container_width=True)

    if numeric_cols:
        default_target = col_ev if col_ev in numeric_cols else numeric_cols[0]
        target = st.selectbox("Correlation target", numeric_cols, index=numeric_cols.index(default_target))
        corrs = df_filt[numeric_cols].corr(numeric_only=True)[target].dropna().sort_values(ascending=False)
        corrs_show = pd.concat([corrs.head(top_n_corr), corrs.tail(top_n_corr)]).drop_duplicates()

        fig = px.bar(
            corrs_show.sort_values(),
            orientation="h",
            title=f"Correlation with {target}",
            labels={"value": "Pearson r", "index": "Metric"}
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Shared plot helper
# -----------------------------
def add_highlight_column(plot_df):
    if highlight_player != "None":
        plot_df["highlight_group"] = np.where(plot_df[col_player].astype(str) == highlight_player, "Highlighted", "Other")
    else:
        plot_df["highlight_group"] = "All"
    return plot_df

# -----------------------------
# Bat Plane
# -----------------------------
with tab_batplane:
    st.subheader("Bat Plane Map: HBA vs VAA")

    if col_hba is None or col_vaa is None:
        st.warning("Could not find HBA and/or VAA.")
    else:
        color_options = [c for c in [col_ev, col_wh, col_xslg, "archetype_label", col_la, col_ss] if c in df_filt.columns]
        color_by = st.selectbox("Color points by", color_options, index=0)

        keep_cols = [col_player, col_hba, col_vaa, color_by]
        plot_df = df_filt[keep_cols].dropna().copy()
        plot_df = add_highlight_column(plot_df)

        if len(plot_df) < min_rows:
            st.info(f"Not enough rows after dropping NAs. Rows available: {len(plot_df)}.")
        else:
            hover_fields = make_hover_fields(
                plot_df,
                [col_player, col_ev, col_la, col_vsa, col_vba, col_vaa, col_hba, col_rad, col_tilt]
            )
            customdata = build_customdata(plot_df, hover_fields)
            hovertemplate = build_hovertemplate(col_player, hover_fields)

            symbol_arg = "highlight_group" if highlight_player != "None" else None

            fig = px.scatter(
                plot_df,
                x=col_hba,
                y=col_vaa,
                color=color_by,
                symbol=symbol_arg,
                hover_name=col_player,
                title="HBA vs VAA"
            )
            fig.update_traces(customdata=customdata, hovertemplate=hovertemplate, marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Damage Zone
# -----------------------------
with tab_damage:
    st.subheader("Damage Zone: Exit Velo vs Launch Angle")

    if col_ev is None or col_la is None:
        st.warning("Could not find Exit Velo and/or Launch Angle.")
    else:
        color_options = [c for c in [col_wh, col_xslg, col_pull, col_gb, "archetype_label"] if c in df_filt.columns]
        color_by = st.selectbox("Color by", color_options, index=0)

        plot_df = df_filt[[col_player, col_ev, col_la, color_by]].dropna().copy()
        plot_df = add_highlight_column(plot_df)

        if len(plot_df) < min_rows:
            st.info(f"Not enough rows after dropping NAs. Rows available: {len(plot_df)}.")
        else:
            hover_fields = make_hover_fields(
                plot_df,
                [col_player, col_ev, col_la, col_wh, col_xslg, col_pull, col_gb, col_vsa, col_vaa]
            )
            customdata = build_customdata(plot_df, hover_fields)
            hovertemplate = build_hovertemplate(col_player, hover_fields)

            symbol_arg = "highlight_group" if highlight_player != "None" else None

            fig = px.scatter(
                plot_df,
                x=col_la,
                y=col_ev,
                color=color_by,
                symbol=symbol_arg,
                hover_name=col_player,
                title="EV vs LA"
            )
            fig.update_traces(customdata=customdata, hovertemplate=hovertemplate, marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# VSA vs LA
# -----------------------------
with tab_vsala:
    st.subheader("VSA vs Launch Angle")
    st.caption("Useful for seeing how bat path translates into batted-ball angle.")

    if col_vsa is None or col_la is None:
        st.warning("Could not find VSA and/or Launch Angle.")
    else:
        color_options = [c for c in [col_ev, col_wh, col_xslg, col_gb, "archetype_label"] if c in df_filt.columns]
        color_by = st.selectbox("Color by", color_options, index=0, key="vsa_la_color")

        plot_df = df_filt[[col_player, col_vsa, col_la, color_by]].dropna().copy()
        plot_df = add_highlight_column(plot_df)

        if len(plot_df) < min_rows:
            st.info(f"Not enough rows after dropping NAs. Rows available: {len(plot_df)}.")
        else:
            hover_fields = make_hover_fields(
                plot_df,
                [col_player, col_vsa, col_la, col_ev, col_wh, col_vaa, col_hba, col_tilt]
            )
            customdata = build_customdata(plot_df, hover_fields)
            hovertemplate = build_hovertemplate(col_player, hover_fields)

            symbol_arg = "highlight_group" if highlight_player != "None" else None

            fig = px.scatter(
                plot_df,
                x=col_vsa,
                y=col_la,
                color=color_by,
                symbol=symbol_arg,
                hover_name=col_player,
                title="VSA vs Launch Angle"
            )
            fig.update_traces(customdata=customdata, hovertemplate=hovertemplate, marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Swing Tradeoff
# -----------------------------
with tab_tradeoff:
    st.subheader("Swing Speed Tradeoffs")

    if col_ss is None:
        st.warning("Could not find Swing Speed.")
    else:
        y_options = [c for c in [col_con, col_iz, col_whiff, col_chase, col_ev, col_wh] if c is not None and c in df_filt.columns]
        if not y_options:
            st.warning("No tradeoff outcome columns detected.")
        else:
            ycol = st.selectbox("Choose Y variable", y_options, index=0)
            color_options = [c for c in [col_ev, col_wh, col_xslg, col_la, "archetype_label"] if c in df_filt.columns]
            color_by = st.selectbox("Color by", color_options, index=0, key="tradeoff_color")

            plot_df = df_filt[[col_player, col_ss, ycol, color_by]].dropna().copy()
            plot_df = add_highlight_column(plot_df)

            if len(plot_df) < min_rows:
                st.info(f"Not enough rows after dropping NAs. Rows available: {len(plot_df)}.")
            else:
                hover_fields = make_hover_fields(
                    plot_df,
                    [col_player, col_ss, ycol, col_ev, col_wh, col_con, col_whiff, col_vsa, col_hba]
                )
                customdata = build_customdata(plot_df, hover_fields)
                hovertemplate = build_hovertemplate(col_player, hover_fields)

                symbol_arg = "highlight_group" if highlight_player != "None" else None

                fig = px.scatter(
                    plot_df,
                    x=col_ss,
                    y=ycol,
                    color=color_by,
                    symbol=symbol_arg,
                    hover_name=col_player,
                    title=f"{ycol} vs {col_ss}"
                )
                fig.update_traces(customdata=customdata, hovertemplate=hovertemplate, marker=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Archetypes
# -----------------------------
with tab_archetypes:
    st.subheader("Mechanical Archetypes")

    if cluster_base.empty:
        st.warning("Not enough valid columns/rows to build archetypes.")
    else:
        st.write("Cluster features used:")
        st.write(cluster_features)

        st.dataframe(
            cluster_base[[col_player, "cluster_id", "archetype_label"] + cluster_features].sort_values(["cluster_id", col_player]),
            use_container_width=True
        )

        archetype_summary = df_filt.groupby("archetype_label", dropna=True).size().reset_index(name="count").sort_values("count", ascending=False)
        st.plotly_chart(
            px.bar(archetype_summary, x="archetype_label", y="count", title="Players by Archetype"),
            use_container_width=True
        )

        if col_hba and col_vaa:
            plot_df = df_filt[[col_player, col_hba, col_vaa, "archetype_label"]].dropna().copy()
            plot_df = add_highlight_column(plot_df)

            hover_fields = make_hover_fields(
                plot_df,
                [col_player, "archetype_label", col_vsa, col_vba, col_vaa, col_hba, col_ss, col_con]
            )
            customdata = build_customdata(plot_df, hover_fields)
            hovertemplate = build_hovertemplate(col_player, hover_fields)

            symbol_arg = "highlight_group" if highlight_player != "None" else None

            fig = px.scatter(
                plot_df,
                x=col_hba,
                y=col_vaa,
                color="archetype_label",
                symbol=symbol_arg,
                hover_name=col_player,
                title="Archetypes on Bat Plane Map"
            )
            fig.update_traces(customdata=customdata, hovertemplate=hovertemplate, marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Outliers
# -----------------------------
with tab_outliers:
    st.subheader("Outliers")

    metric_candidates = [c for c in [
        col_vsa, col_vba, col_vaa, col_hfa, col_hba, col_rad, col_tilt,
        col_ss, col_ev, col_la, col_wh, col_con, col_whiff, col_pull, col_gb, col_xslg
    ] if c is not None and c in df_filt.columns]

    chosen = st.multiselect("Metrics to scan", metric_candidates, default=metric_candidates[:min(8, len(metric_candidates))])

    out_rows = []
    for m in chosen:
        z = robust_zscores(df_filt[m])
        flagged = z[abs(z) >= z_thresh].sort_values()
        for idx, zval in flagged.items():
            out_rows.append({
                "player": df_filt.loc[idx, col_player],
                "metric": m,
                "value": df_filt.loc[idx, m],
                "robust_z": round(zval, 2),
                "archetype": df_filt.loc[idx, "archetype_label"] if "archetype_label" in df_filt.columns else np.nan
            })

    if out_rows:
        out_df = pd.DataFrame(out_rows).sort_values(["metric", "robust_z"])
        st.dataframe(out_df, use_container_width=True)
        st.download_button(
            "Download outliers CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="outliers_v3.csv",
            mime="text/csv"
        )
    else:
        st.info("No outliers found at the current threshold.")

# -----------------------------
# Player Search
# -----------------------------
with tab_player:
    st.subheader("Player Search")

    players_sorted = sorted(df_filt[col_player].astype(str).dropna().unique().tolist())
    selected = st.selectbox("Choose player", players_sorted)

    row_idx = df_filt.index[df_filt[col_player].astype(str) == selected]
    if len(row_idx) == 0:
        st.warning("Player not found.")
    else:
        idx = row_idx[0]
        st.markdown(f"### {selected}")

        card_metrics = [c for c in [
            col_vsa, col_vba, col_vaa, col_hfa, col_hba, col_rad, col_tilt,
            col_ss, col_ev, col_la, col_wh, col_con, col_whiff, col_pull, col_gb, col_xslg
        ] if c is not None and c in df_filt.columns]

        rows = []
        for m in card_metrics:
            val = df_filt.loc[idx, m]
            pct = percentile_rank(df_filt[m], val)
            rows.append({
                "metric": m,
                "value": val,
                "percentile": None if np.isnan(pct) else round(pct, 1)
            })

        card = pd.DataFrame(rows)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Raw values**")
            st.dataframe(card[["metric", "value"]], use_container_width=True)
        with c2:
            st.write("**Percentiles**")
            st.dataframe(card[["metric", "percentile"]], use_container_width=True)

        if "archetype_label" in df_filt.columns:
            st.write(f"**Archetype:** {df_filt.loc[idx, 'archetype_label']}")

# -----------------------------
# Comps Finder
# -----------------------------
with tab_comps:
    st.subheader("Comps Finder")
    st.caption("Closest hitters by swing/shape profile using standardized distance.")

    comp_features = [c for c in [col_vsa, col_vba, col_vaa, col_hfa, col_hba, col_rad, col_tilt, col_ss, col_con, col_whiff, col_pull, col_gb] if c is not None and c in df_filt.columns]

    if len(comp_features) < 3:
        st.warning("Not enough comparable numeric swing columns for comps.")
    else:
        comp_base = df_filt[[col_player] + comp_features].dropna().copy()

        if len(comp_base) < 5:
            st.warning("Not enough rows after filtering for comps.")
        else:
            comp_players = sorted(comp_base[col_player].astype(str).unique().tolist())
            target_player = st.selectbox("Choose comp target", comp_players, key="comp_target")

            scaler = StandardScaler()
            comp_scaled = scaler.fit_transform(comp_base[comp_features])

            comp_scaled_df = pd.DataFrame(comp_scaled, columns=comp_features, index=comp_base.index)
            target_idx = comp_base.index[comp_base[col_player].astype(str) == target_player][0]

            target_vec = comp_scaled_df.loc[target_idx].values
            distances = np.sqrt(((comp_scaled_df.values - target_vec) ** 2).sum(axis=1))

            comp_base = comp_base.copy()
            comp_base["distance"] = distances
            comp_base = comp_base[comp_base[col_player].astype(str) != target_player]
            comp_base = comp_base.sort_values("distance").head(n_comps)

            display_cols = [col_player, "distance"] + comp_features[:6]
            st.dataframe(comp_base[display_cols], use_container_width=True)

# -----------------------------
# Model
# -----------------------------
with tab_model:
    st.subheader("Quick Model")

    if numeric_cols:
        default_target = col_ev if col_ev in numeric_cols else numeric_cols[0]
        target = st.selectbox("Target variable", numeric_cols, index=numeric_cols.index(default_target), key="model_target")
        feats_all = [c for c in numeric_cols if c != target]
        default_feats = [c for c in [col_vsa, col_vba, col_vaa, col_hba, col_hfa, col_rad, col_tilt, col_ss, col_la, col_con, col_whiff, col_pull] if c in feats_all]
        feats = st.multiselect("Features", feats_all, default=default_feats)

        if len(feats) >= 2:
            model_df = df_filt[[target] + feats].dropna()
            if len(model_df) >= 10:
                X = model_df[feats].values
                y = model_df[target].values

                lr = LinearRegression().fit(X, y)
                pred = lr.predict(X)
                resid = y - pred

                c1, c2, c3 = st.columns(3)
                c1.metric("Rows used", f"{len(model_df):,}")
                c2.metric("R²", f"{r2_score(y, pred):.3f}")
                c3.metric("MAE", f"{mean_absolute_error(y, pred):.4f}")

                st.plotly_chart(px.scatter(x=pred, y=y, labels={"x": "Predicted", "y": "Actual"}, title="Actual vs Predicted"), use_container_width=True)
                st.plotly_chart(px.scatter(x=pred, y=resid, labels={"x": "Predicted", "y": "Residual"}, title="Residuals vs Predicted"), use_container_width=True)
            else:
                st.info("Not enough rows after filtering for the model.")
        else:
            st.info("Pick at least 2 features.")

# -----------------------------
# Data
# -----------------------------
with tab_data:
    st.subheader("Data Preview")
    st.dataframe(df_filt.head(50), use_container_width=True)
    st.write("Columns:")
    st.code("\n".join(df_raw.columns.tolist()))