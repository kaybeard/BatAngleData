import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="MLB xSLG Explorer", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------

def standardize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def find_col(cols, candidates):

    def key(x):
        return re.sub(r"[%\s_]", "", x.lower())

    cols_keyed = {key(c): c for c in cols}

    for cand in candidates:
        k = key(cand)
        if k in cols_keyed:
            return cols_keyed[k]

    for c in cols:
        ck = key(c)
        for cand in candidates:
            if key(cand) in ck:
                return c

    return None


def to_numeric_series(s):

    if s.dtype.kind in "biufc":
        return s

    s2 = s.astype(str).str.replace("%", "", regex=False)
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.strip()

    return pd.to_numeric(s2, errors="coerce")


def robust_zscores(x):

    x = x.dropna()

    if len(x) < 10:
        return pd.Series(dtype=float)

    med = x.median()
    mad = (x - med).abs().median()

    if mad == 0:
        return pd.Series(index=x.index, data=np.zeros(len(x)))

    return 0.6745 * (x - med) / mad


def scatter_plot(df, xcol, ycol):

    x = df[xcol]
    y = df[ycol]

    mask = x.notna() & y.notna()

    fig = plt.figure()

    plt.scatter(x[mask], y[mask])

    plt.xlabel(xcol)
    plt.ylabel(ycol)

    plt.title(f"{ycol} vs {xcol}")

    plt.tight_layout()

    return fig


# -----------------------------
# UI
# -----------------------------

st.title("MLB xSLG Explorer (Upload CSV)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.stop()

# -----------------------------
# Load CSV (FIX INCLUDED HERE)
# -----------------------------

df = pd.read_csv(uploaded)

# Detect broken header rows
if all("Unnamed" in str(c) for c in df.columns):
    df = pd.read_csv(uploaded, skiprows=1)

if all("Unnamed" in str(c) for c in df.columns):
    df = pd.read_csv(uploaded, skiprows=2)

df.columns = [standardize_name(c) for c in df.columns]

# -----------------------------
# Identify columns
# -----------------------------

col_player = find_col(df.columns, ["player", "name", "batter"])
col_xslg = find_col(df.columns, ["xSLG", "x_slg"])

col_ev = find_col(df.columns, ["Exit Velo", "EV"])
col_wh = find_col(df.columns, ["Well Hit %"])
col_gb = find_col(df.columns, ["GB %"])
col_ss = find_col(df.columns, ["Swing Speed"])
col_la = find_col(df.columns, ["Launch Angle", "LA"])
col_hba = find_col(df.columns, ["HBA"])
col_vaa = find_col(df.columns, ["VAA"])

if col_xslg is None:

    st.error("Couldn't find xSLG column.")

    st.write("Detected columns:")
    st.write(list(df.columns))

    st.stop()

# -----------------------------
# Convert numeric columns
# -----------------------------

df_num = df.copy()

for c in df_num.columns:
    df_num[c] = to_numeric_series(df_num[c])

numeric_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()

# -----------------------------
# Correlation analysis
# -----------------------------

st.header("Correlation with xSLG")

corrs = df_num[numeric_cols].corr(numeric_only=True)[col_xslg].dropna()

corrs = corrs.sort_values(ascending=False)

st.dataframe(corrs)

fig = plt.figure()

corrs.head(15).sort_values().plot(kind="barh")

plt.title("Top correlations with xSLG")

plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Scatter plots
# -----------------------------

st.header("Key Scatter Plots")

plots = [col_ev, col_wh, col_gb, col_ss, col_hba, col_vaa, col_la]

plots = [p for p in plots if p is not None]

for p in plots:

    fig = scatter_plot(df_num, p, col_xslg)

    st.pyplot(fig)

# -----------------------------
# Top vs bottom xSLG
# -----------------------------

st.header("Top vs Bottom xSLG")

df_sorted = df_num.sort_values(col_xslg, ascending=False)

top = df_sorted.head(15)

bottom = df_sorted.tail(15)

metrics = [col_ev, col_wh, col_gb, col_ss, col_hba, col_vaa, col_la]

metrics = [m for m in metrics if m is not None]

data = []
labels = []

for m in metrics:

    data.append(top[m].dropna())
    labels.append(f"Top {m}")

    data.append(bottom[m].dropna())
    labels.append(f"Bottom {m}")

fig = plt.figure(figsize=(8,6))

plt.boxplot(data, labels=labels, vert=False)

plt.title("Top vs Bottom xSLG Profiles")

plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Outliers
# -----------------------------

st.header("Outliers")

name_series = df[col_player] if col_player else pd.Series(df.index)

outlier_metrics = [m for m in metrics]

outliers = []

for m in outlier_metrics:

    z = robust_zscores(df_num[m])

    flagged = z[abs(z) >= 3]

    for idx, val in flagged.items():

        outliers.append({
            "player": name_series.iloc[idx],
            "metric": m,
            "value": df_num.loc[idx, m],
            "z": val
        })

if outliers:

    out_df = pd.DataFrame(outliers)

    st.dataframe(out_df)

# -----------------------------
# Model
# -----------------------------

st.header("Quick xSLG Model")

model_features = [m for m in metrics if m is not None]

model_df = df_num[[col_xslg] + model_features].dropna()

X = model_df[model_features].values

y = model_df[col_xslg].values

lr = LinearRegression()

lr.fit(X, y)

pred = lr.predict(X)

st.write("R²:", round(r2_score(y, pred),3))
st.write("MAE:", round(mean_absolute_error(y, pred),4))

fig = plt.figure()

plt.scatter(pred, y)

plt.xlabel("Predicted xSLG")
plt.ylabel("Actual xSLG")

plt.title("Actual vs Predicted")

plt.tight_layout()

st.pyplot(fig)