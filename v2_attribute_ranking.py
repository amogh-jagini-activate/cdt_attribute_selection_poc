# Databricks notebook source
# MAGIC %md
# MAGIC # V2 Attribute Ranking — Pure Semantic Similarity
# MAGIC
# MAGIC Ranks product attributes within a category using **cosine similarity**
# MAGIC between category name and attribute description embeddings (e5-large-v2).
# MAGIC
# MAGIC **No sales weights or domain logic** — purely embedding-driven.
# MAGIC
# MAGIC Requires: `data_setup` notebook to have been run first.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# %pip install sentence-transformers scikit-learn --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Data
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

# DBTITLE 1,Rule-Based Attribute Filtering
# ─────────────────────────────────────────────────────────────────────────
# Hard filters (V2-specific):
#   1. ATT_SKU_COUNT ≥ 1% of CATEGORY_SKU_COUNT
#   2. ATT_TOTAL_SPENT > 0
# ─────────────────────────────────────────────────────────────────────────

# Compute category-level SKU count (max attribute coverage as proxy)
category_sku_count = (
    df.groupby("CATEGORY_INTERNAL_ID")["ATTRIBUTE_COVERAGE_SKU_COUNT"]
    .max()
    .rename("CATEGORY_SKU_COUNT")
)
df = df.merge(category_sku_count, on="CATEGORY_INTERNAL_ID", how="left")

before = len(df)

# Filter 1: attribute must appear on at least 1% of category SKUs
df = df[df["ATTRIBUTE_COVERAGE_SKU_COUNT"] >= 0.01 * df["CATEGORY_SKU_COUNT"]]

# Filter 2: attribute must have non-zero sales
df = df[df["ATTRIBUTE_TOTAL_SPENT"] > 0]

after = len(df)
print(f"Rows before filtering: {before}")
print(f"Rows after filtering:  {after}")
print(f"Removed: {before - after}")

# COMMAND ----------

# DBTITLE 1,Preview Cleaned Data
display(df)

# COMMAND ----------

# DBTITLE 1,Initialize Embedding Model
from sentence_transformers import SentenceTransformer

class EmbedModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize_embeddings=True):
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings
        )

# COMMAND ----------

# DBTITLE 1,Attribute Ranking Function (Cosine Similarity)
from sklearn.metrics.pairwise import cosine_similarity

embed_model = EmbedModel("intfloat/e5-large-v2")

def compute_top_attributes(df, category_internal_id):
    """
    Rank attributes for a category by cosine similarity
    between the category name and each attribute description.
    """

    # Filter to one category
    pdf = df[df["CATEGORY_INTERNAL_ID"] == category_internal_id].copy()

    if pdf.empty:
        raise ValueError(
            f"No rows found for CATEGORY_INTERNAL_ID={category_internal_id}"
        )

    # Deduplicate category × attribute
    pdf = (
        pdf
        .drop_duplicates(subset=["CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID"])
        .reset_index(drop=True)
    )

    # Build query and passages
    query = f"query: {pdf['category'].iloc[0]}"
    passages = [f"passage: {x}" for x in pdf["attribute"]]

    # Embeddings
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    a_emb = embed_model.encode(passages, normalize_embeddings=True)

    # Cosine similarity + ranking
    pdf["score"] = cosine_similarity(q_emb, a_emb).flatten()

    return (
        pdf
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# COMMAND ----------

# DBTITLE 1,Rank: Category 879771
display(
    compute_top_attributes(
        df,
        category_internal_id=879771
    )
)

# COMMAND ----------

# DBTITLE 1,Rank: Category 879999
display(
    compute_top_attributes(
        df,
        category_internal_id=879999
    )
)

# COMMAND ----------

# DBTITLE 1,Rank: Category 880115
display(compute_top_attributes(
    df,
    category_internal_id=880115
))

# COMMAND ----------

# DBTITLE 1,Rank: Category 880100
display(compute_top_attributes(
    df,
    category_internal_id=880100
))

# COMMAND ----------

# DBTITLE 1,Rank: Category 879696
display(compute_top_attributes(
    df,
    category_internal_id=879696
))
