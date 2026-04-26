# Databricks notebook source
# MAGIC %md
# MAGIC # V3 Attribute Ranking — Sales-Weighted Multi-Mode Ranking
# MAGIC
# MAGIC Ranks product attributes using four modes:
# MAGIC - **semantic** — cosine similarity only
# MAGIC - **sales** — CDT sales weight only
# MAGIC - **combined** — MAUT-style multiplicative utility (semantic × sales)
# MAGIC - **rrf** — Reciprocal Rank Fusion
# MAGIC
# MAGIC Requires: `data_setup` notebook to have been run first (produces `ATTRIBUTE_WEIGHT`).

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# %pip install sentence-transformers scikit-learn --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Data (with Sales Weights)
df = spark.read.parquet('dbfs:/FileStore/amogh/cdt/final_df')
display(df)

# COMMAND ----------

# DBTITLE 1,Abbreviation Expansion & Category Cleanup
import re

ABBREVIATION_MAP = {
    "STTD": "stated", "QLFD": "qualified", "PRSNC": "presence", "PRSNG": "presence",
    "CRN": "corn", "SGR": "sugar", "FAT": "fat", "CAL": "calorie", "PROT": "protein",
    "CRBHY": "carbohydrate", "SRVNG": "serving", "RNGS": "ranges",
    "FDA": "fda", "ORG": "organic", "MRKTNG": "marketing", "CLM": "claim",
    "EQ": "equivalent", "NUM": "number", "QTY": "quantity", "TTL": "total",
    "SBSTT": "substitute", "FLR": "flavor",
}

def expand_abbreviations(text: str) -> str:
    if text is None:
        return ""
    return " ".join(
        ABBREVIATION_MAP.get(
            re.sub(r"[^A-Z_]", "", w.upper()),
            w.lower()
        )
        for w in str(text).split()
    )

def clean_category_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"^\d+\s*-\s*", "", name)
    name = re.sub(
        r"\b(SG|GR|PP|MT|HB|NF|BK|FR|DD|FL|OB|RX|RW|EX|EW|ALL)\b",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"[&/,\-]", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()

# COMMAND ----------

# DBTITLE 1,Add Cleaned Columns (Pandas)
df = df.toPandas()

df["category"] = df["CATEGORY_NAME"].map(clean_category_name)
df["attribute"] = df["ATTRIBUTE_DESCRIPTION"].map(expand_abbreviations)

# COMMAND ----------

# DBTITLE 1,Preview Cleaned Data
display(df)

# COMMAND ----------

# DBTITLE 1,Initialize Embedding Model & Imports
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

embed_model = SentenceTransformer("intfloat/e5-large-v2")

# COMMAND ----------

# DBTITLE 1,Attribute Ranking Function (Multi-Mode)
# ============================================================
# ATTRIBUTE RANKING – SINGLE ENTRY POINT
#
# Supported modes:
# - "semantic"      → cosine similarity ranking
# - "sales"         → CDT sales weight ranking
# - "combined"      → MAUT-style multiplicative utility
# - "rrf"           → Reciprocal Rank Fusion (ACM paper)
#
# INPUT:
# - category_internal_id
# - mode ∈ {"semantic", "sales", "combined", "rrf"}
#
# OUTPUT COLUMNS:
# - CATEGORY_INTERNAL_ID
# - CATEGORY_NAME
# - ATTRIBUTE_ID
# - ATTRIBUTE_DESCRIPTION
# - FINAL_WEIGHT
# ============================================================


def rank_attributes(
    df,
    category_internal_id: int,
    mode: str = "combined",
    w_semantic: float = 0.6,
    w_sales: float = 0.4,
    rrf_k: int = 60
):
    """
    Rank attributes for a category using different fusion methods.
    """

    # --- Filter to category -------------------------------------------
    cat_df = df[df["CATEGORY_INTERNAL_ID"] == category_internal_id].copy()
    if cat_df.empty:
        return cat_df

    # --- SALES-ONLY MODE ----------------------------------------------
    if mode == "sales":
        return (
            cat_df
            .sort_values("ATTRIBUTE_WEIGHT", ascending=False)
            .loc[:, [
                "CATEGORY_INTERNAL_ID",
                "CATEGORY_NAME",
                "ATTRIBUTE_ID",
                "ATTRIBUTE_DESCRIPTION",
                "ATTRIBUTE_WEIGHT"
            ]]
            .rename(columns={"ATTRIBUTE_WEIGHT": "FINAL_WEIGHT"})
            .reset_index(drop=True)
        )

    # --- SEMANTIC SCORING (needed for semantic / combined / rrf) ------
    attr_texts = cat_df["attribute"].tolist()
    category_text = cat_df["category"].iloc[0]

    attr_emb = embed_model.encode(attr_texts, normalize_embeddings=True)
    cat_emb = embed_model.encode(category_text, normalize_embeddings=True)

    cat_df["SEMANTIC_SCORE"] = np.dot(attr_emb, cat_emb)

    # --- SEMANTIC-ONLY MODE -------------------------------------------
    if mode == "semantic":
        return (
            cat_df
            .sort_values("SEMANTIC_SCORE", ascending=False)
            .loc[:, [
                "CATEGORY_INTERNAL_ID",
                "CATEGORY_NAME",
                "ATTRIBUTE_ID",
                "ATTRIBUTE_DESCRIPTION",
                "SEMANTIC_SCORE"
            ]]
            .rename(columns={"SEMANTIC_SCORE": "FINAL_WEIGHT"})
            .reset_index(drop=True)
        )

    # --- COMBINED MODE (MULTIPLICATIVE MAUT) --------------------------
    if mode == "combined":
        scaler = MinMaxScaler()

        cat_df["SEMANTIC_NORM"] = scaler.fit_transform(
            cat_df[["SEMANTIC_SCORE"]]
        )
        cat_df["SALES_NORM"] = scaler.fit_transform(
            cat_df[["ATTRIBUTE_WEIGHT"]]
        )

        epsilon = 1e-8
        cat_df["FINAL_WEIGHT"] = (
            (cat_df["SEMANTIC_NORM"] + epsilon) ** w_semantic
        ) * (
            (cat_df["SALES_NORM"] + epsilon) ** w_sales
        )

        return (
            cat_df
            .sort_values("FINAL_WEIGHT", ascending=False)
            .loc[:, [
                "CATEGORY_INTERNAL_ID",
                "CATEGORY_NAME",
                "ATTRIBUTE_ID",
                "ATTRIBUTE_DESCRIPTION",
                "FINAL_WEIGHT"
            ]]
            .reset_index(drop=True)
        )

    # --- RRF MODE (Reciprocal Rank Fusion) ----------------------------
    if mode == "rrf":
        cat_df["SEMANTIC_RANK"] = (
            cat_df["SEMANTIC_SCORE"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        cat_df["SALES_RANK"] = (
            cat_df["ATTRIBUTE_WEIGHT"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        cat_df["FINAL_WEIGHT"] = (
            1 / (rrf_k + cat_df["SEMANTIC_RANK"]) +
            1 / (rrf_k + cat_df["SALES_RANK"])
        )

        return (
            cat_df
            .sort_values("FINAL_WEIGHT", ascending=False)
            .loc[:, [
                "CATEGORY_INTERNAL_ID",
                "CATEGORY_NAME",
                "ATTRIBUTE_ID",
                "ATTRIBUTE_DESCRIPTION",
                "FINAL_WEIGHT"
            ]]
            .reset_index(drop=True)
        )

    raise ValueError("mode must be one of: semantic, sales, combined, rrf")

# COMMAND ----------

# DBTITLE 1,RRF: Category 879771
display(rank_attributes(df, 879771, mode="rrf"))

# COMMAND ----------

# DBTITLE 1,RRF: Category 879696
display(rank_attributes(df, 879696, mode="rrf"))

# COMMAND ----------

# DBTITLE 1,RRF: Category 879999
display(rank_attributes(df, 879999, mode="rrf"))

# COMMAND ----------

# DBTITLE 1,RRF: Category 880115
display(rank_attributes(df, 880115, mode="rrf"))

# COMMAND ----------

# DBTITLE 1,RRF: Category 880100
display(rank_attributes(df, 880100, mode="rrf"))

# COMMAND ----------

# DBTITLE 1,Combined: Category 879999
display(rank_attributes(df, 879999, mode="combined"))

# COMMAND ----------

# DBTITLE 1,Combined: Category 880115
display(rank_attributes(df, 880115, mode="combined"))

# COMMAND ----------

# DBTITLE 1,Combined: Category 880100
display(rank_attributes(df, 880100, mode="combined"))

# COMMAND ----------

# DBTITLE 1,Combined: Category 879771
display(rank_attributes(df, 879771, mode="combined"))

# COMMAND ----------

# DBTITLE 1,Combined: Category 879696
display(rank_attributes(df, 879696, mode="combined"))

# COMMAND ----------

# DBTITLE 1,Combined: Category 879999 (repeat)
display(rank_attributes(df, 879999, mode="combined"))
