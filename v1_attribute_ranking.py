# Databricks notebook source
# MAGIC %md
# MAGIC # V1 Attribute Ranking — Raw Semantic-Only Matching
# MAGIC
# MAGIC The simplest version: ranks product attributes within a category using
# MAGIC **cosine similarity** between raw category names and raw attribute descriptions.
# MAGIC
# MAGIC Key characteristics:
# MAGIC - **No text normalization or cleanup** (no abbreviation expansion, no prefix removal)
# MAGIC - **No sales, SKU count, or coverage signals**
# MAGIC - **All attributes treated equally**
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

# DBTITLE 1,Convert to Pandas
df = df.toPandas()

# COMMAND ----------

# DBTITLE 1,Preview Data
display(df)

# COMMAND ----------

# DBTITLE 1,Initialize Embedding Model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embed_model = SentenceTransformer("intfloat/e5-large-v2")

# COMMAND ----------

# DBTITLE 1,Attribute Ranking Function (Raw Cosine Similarity)
def compute_top_attributes(df, category_internal_id):
    """
    Rank attributes for a category by cosine similarity between
    the RAW category name and RAW attribute description.

    No text cleanup, no abbreviation expansion, no weighting.
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

    # Build query and passages using RAW text (no cleanup)
    query = f"query: {pdf['CATEGORY_NAME'].iloc[0]}"
    passages = [f"passage: {x}" for x in pdf["ATTRIBUTE_DESCRIPTION"]]

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
    compute_top_attributes(df, category_internal_id=879771)
)

# COMMAND ----------

# DBTITLE 1,Rank: Category 879999
display(
    compute_top_attributes(df, category_internal_id=879999)
)

# COMMAND ----------

# DBTITLE 1,Rank: Category 880115
display(
    compute_top_attributes(df, category_internal_id=880115)
)

# COMMAND ----------

# DBTITLE 1,Rank: Category 880100
display(
    compute_top_attributes(df, category_internal_id=880100)
)

# COMMAND ----------

# DBTITLE 1,Rank: Category 879696
display(
    compute_top_attributes(df, category_internal_id=879696)
)
