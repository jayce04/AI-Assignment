# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Skincare Recommender", page_icon="✨", layout="wide")

# Data loading + TF-IDF
@st.cache_data(show_spinner=True)
def load_products(path="products_preprocessed.csv"):
    df = pd.read_csv(path)
    for c in ["price_usd","rating","reviews"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["skin_concern"] = df.get("skin_concern", "").fillna("").astype(str)
    df["skin_type"]    = df.get("skin_type", "").fillna("").astype(str)
    df["product_type"] = df.get("product_type", "").fillna("").astype(str)
    return df

@st.cache_resource(show_spinner=True)
def build_vectorizer_and_matrix(product_text: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(product_text.fillna("").astype(str))
    return vectorizer, tfidf_matrix

df = load_products("products_preprocessed.csv")
vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["product_content"])

# Recommender
def contentbased_recommender(
    product_type=None,
    skin_type=None,
    skin_concern=None,
    concern_match="all",
    max_price=None,
    n=10
):
    def _to_set(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return set()
        if isinstance(x, (list, tuple, set)):
            return {str(t).strip().lower() for t in x if str(t).strip()}
        # split on common separators
        return {t.strip().lower() for t in re.split(r"[;,/|]", str(x)) if t.strip()}

    req_type    = str(product_type).strip().lower() if product_type else None
    req_skin    = str(skin_type).strip().lower()    if skin_type    else None
    req_concern = _to_set(skin_concern)

    # profile → tfidf
    tokens = []
    if req_type:    tokens.append(req_type)
    if req_skin:    tokens.append(req_skin)
    if req_concern: tokens.extend(sorted(req_concern))
    profile_text = " ".join(tokens).strip() or "skincare"

    qv = vectorizer.transform([profile_text])
    sims = cosine_similarity(qv, tfidf_matrix).ravel()

    price_col   = pd.to_numeric(df.get("price_usd", np.nan), errors="coerce")
    rating_col  = pd.to_numeric(df.get("rating", np.nan),    errors="coerce").fillna(0.0)
    reviews_col = pd.to_numeric(df.get("reviews", 0),        errors="coerce").fillna(0).astype(int)

    rows = []
    for i, sim in enumerate(sims):
        row = df.iloc[i]

        # product_type exact match (if provided)
        if req_type and str(row.get("product_type","")).strip().lower() != req_type:
            continue

        # skin_type (single-valued)
        row_skin = str(row.get("skin_type","")).strip().lower()
        if req_skin and row_skin != req_skin:
            continue

        # concerns: all/any
        row_concern = _to_set(row.get("skin_concern",""))
        if req_concern:
            if concern_match == "all":
                if not req_concern.issubset(row_concern):
                    continue
            else:  # any
                if row_concern.isdisjoint(req_concern):
                    continue

        # budget
        p = price_col.iat[i]
        if max_price is not None and (pd.isna(p) or p > float(max_price)):
            continue

        rows.append({
            "product_name": row.get("product_name",""),
            "brand_name": row.get("brand_name",""),
            "product_type": row.get("product_type",""),
            "skin_type": row.get("skin_type",""),
            "skin_concern": row.get("skin_concern",""),
            "price_usd": row.get("price_usd",""),
            "rating": rating_col.iat[i],
            "reviews": reviews_col.iat[i],
            "similarity": float(sim)
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        by=["similarity","rating","reviews"],
        ascending=[False, False, False]
    ).head(n)

    out["similarity"] = out["similarity"].round(4)
    return out


# Sidebar Controls
st.sidebar.title("Filters")

# product type options (clean + sorted)
type_options = sorted({t.strip().lower() for t in df["product_type"].dropna().astype(str) if t.strip()} )
product_type = st.sidebar.selectbox("Product type", options=["(any)"] + type_options, index=0)
product_type = None if product_type == "(any)" else product_type

# skin type (single-valued)
skin_options = ["dry", "normal", "oily", "combination", "sensitive"]
skin_type = st.sidebar.selectbox("Skin type", options=["(any)"] + skin_options, index=0)
skin_type = None if skin_type == "(any)" else skin_type

# concerns (comma-separated lists)
def all_concerns_unique(df):
    s = df["skin_concern"].fillna("").astype(str)
    uniq = set()
    for txt in s:
        for t in re.split(r"[;,/|]", txt):
            t = t.strip().lower()
            if t:
                uniq.add(t)
    return sorted(uniq)

concern_options = all_concerns_unique(df)
chosen_concerns = st.sidebar.multiselect("Concerns", options=concern_options, default=[])

concern_match = st.sidebar.radio("Concern match", options=["all", "any"], horizontal=True, index=0)

max_price = st.sidebar.number_input("Max budget (USD)", min_value=0.0, value=100.0, step=1.0, format="%.2f")

n_items = st.sidebar.slider("How many items to return", min_value=5, max_value=50, value=10, step=1)


# Main UI
st.title("✨ Skincare Content-Based Recommender")

st.markdown(
    "Pick your **type**, **skin type**, **concerns**, **budget**, and how many items to return — "
    "then click **Recommend**."
)

if st.button("Recommend"):
    with st.spinner("Finding matches…"):
        recs = contentbased_recommender(
            product_type=product_type,
            skin_type=skin_type,
            skin_concern=chosen_concerns if chosen_concerns else None,
            concern_match=concern_match,
            max_price=max_price,
            n=n_items
        )
    if recs is None or recs.empty:
        st.warning("No matching products found. Try relaxing filters (e.g., use 'any' for concerns or increase budget).")
    else:
        cols = ["product_name","brand_name","product_type","skin_type","skin_concern",
                "price_usd","rating","reviews","similarity"]
        recs = recs[[c for c in cols if c in recs.columns]]

        st.subheader("Results")
        st.dataframe(
            recs,
            use_container_width=True,
            hide_index=True
        )

        csv = recs.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", data=csv, file_name="skincare_recommendations.csv", mime="text/csv")
