# Databricks notebook source
# MAGIC %md
# MAGIC # Unified Data Setup — Category × Attribute with Sales Weights
# MAGIC
# MAGIC This notebook prepares the **category × attribute** dataset used by ALL
# MAGIC attribute ranking versions (V2, V3, V4).
# MAGIC
# MAGIC It loads tables from the Azure data lake, builds the category-attribute-SKU
# MAGIC mapping, computes sales attribution with coverage & diversity adjustments,
# MAGIC and saves the final dataset to DBFS.
# MAGIC
# MAGIC **Output:** `dbfs:/FileStore/amogh/cdt/final_df`
# MAGIC
# MAGIC Each ranking notebook loads this output and uses whichever columns it needs:
# MAGIC - V2 uses only category / attribute columns (pure semantic)
# MAGIC - V3 uses ATTRIBUTE_WEIGHT (sales-weighted multi-mode)
# MAGIC - V4 uses only category / attribute columns (domain × tier weighting)

# COMMAND ----------

# DBTITLE 1,Azure Environment Setup
# MAGIC %run ./init_spark_env_actusvaldatalake

# COMMAND ----------

# DBTITLE 1,Imports & Autoreload
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import datetime
# MAGIC import dateutil
# MAGIC import pyspark.sql.functions as func

# COMMAND ----------

# DBTITLE 1,Retailer Configuration
retailerId = '5200'
retailer_name = 'Wakefern'

from cv_tables import CvTables
DATA_LAKE_STORAGE_NAME = 'actusvaldatalake'

RESOURCES_PATH = f"abfs://data-container@{DATA_LAKE_STORAGE_NAME}.dfs.core.windows.net/data/"
cv_tables = CvTables(spark, RESOURCES_PATH, retailerId)

# COMMAND ----------

# DBTITLE 1,Load Source Tables (Purchase History, Catalog, Hierarchy)
purchaseHistory = cv_tables.init_purchase_history()
catalog = cv_tables.init_catalog_df()
cp = cv_tables.init_df('/input/row/parquet/CV_R_ALL_PARENTS')
hierarchy_desc = cv_tables.init_df('/input/row/parquet/CV_R_CPG_BRAND_CATALOG_TREE')

# COMMAND ----------

# DBTITLE 1,Load Attribute Data
from pyspark.sql.functions import col

attr_mapping_path = '/input/row/parquet/CV_R_ALL_ATTRIBUTES/CV_R_PRODUCT_ATTRIBUTE_RELATION'
attr_mapping_raw = spark.read.format("delta").load(
    RESOURCES_PATH + retailerId + attr_mapping_path
)

attr_master_path = '/input/row/parquet/CV_R_ALL_ATTRIBUTES/CV_R_ATTRIBUTE_MASTER'
attr_master = spark.read.format("delta").load(
    RESOURCES_PATH + retailerId + attr_master_path
)

attr_value_path = '/input/row/parquet/CV_R_ALL_ATTRIBUTES/CV_R_ATTRIBUTE_VALUES'
attr_value = spark.read.format("delta").load(
    RESOURCES_PATH + retailerId + attr_value_path
)

model_input_product_attr = (
    attr_mapping_raw
    .join(
        attr_master
            .select('ATTRIBUTE_ID', 'ATTRIBUTE_TYPE', 'ATTRIBUTE_DESCRIPTION'),
        on=['ATTRIBUTE_ID'],
        how='left'
    )
    .join(
        attr_value
            .select(
                'ATTRIBUTE_ID',
                'ATTRIBUTE_VALUE_ID',
                'ATTRIBUTE_VALUE_DESCRIPTION'
            ),
        on=['ATTRIBUTE_ID', 'ATTRIBUTE_VALUE_ID'],
        how='left'
    )
)

# COMMAND ----------

# DBTITLE 1,Preview Attribute Data
display(model_input_product_attr)

# COMMAND ----------

# DBTITLE 1,Preview Distinct Attributes
display(model_input_product_attr.select("ATTRIBUTE_DESCRIPTION").distinct())

# COMMAND ----------

# DBTITLE 1,Schema Summary of All DataFrames
dfs = {
    "purchaseHistory": purchaseHistory,
    "catalog": catalog,
    "cp": cp,
    "hierarchy_desc": hierarchy_desc,
    "model_input_product_attr": model_input_product_attr
}

for name, df in dfs.items():
    print("=" * 80)
    print(f"DataFrame: {name}")
    print("=" * 80)
    print("\nSchema:")
    df.printSchema()
    print("\nSample data (5 rows):")
    df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Joins & Sales Weight Pipeline
# MAGIC
# MAGIC Associates sales with attributes through SKUs, computes coverage-aware
# MAGIC attribute weights per category × attribute.

# COMMAND ----------

# DBTITLE 1,Step 1: SKU → Category Mapping
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ------------------------------------------------------------------------------------
# "Which category does each SKU belong to?"
# ------------------------------------------------------------------------------------
sku_to_category = (
    cp
    .filter(
        (F.col("HIERARCHY") == "CATALOG") &
        (F.col("CHILD_LEVEL") == 0) &      # SKUs
        (F.col("PARENT_LEVEL") == 3)       # Categories
    )
    .select(
        F.col("CHILD_ID").alias("SKU_ID"),
        F.col("PARENT_ID").alias("CATEGORY_INTERNAL_ID")
    )
    .dropDuplicates()
    .join(
        catalog
        .filter(
            (F.col("PRODUCT_TREE_LEVEL") == 3) &
            (F.col("STATUS") == "A")
        )
        .select(
            F.col("INTERNAL_ID").alias("CATEGORY_INTERNAL_ID"),
            F.col("PRODUCT_TREE_LEVEL_DESCRIPTION").alias("CATEGORY_NAME")
        ),
        "CATEGORY_INTERNAL_ID",
        "left"
    )
    .dropna(subset=["CATEGORY_NAME"])
)

# COMMAND ----------

# DBTITLE 1,Step 2: SKU × Category × Attribute × Value
# ------------------------------------------------------------------------------------
# "Which attributes and attribute-values does each SKU have?"
# ------------------------------------------------------------------------------------
category_attribute_value = (
    model_input_product_attr.alias("m")
    .join(
        sku_to_category.alias("sc"),
        F.col("m.INTERNAL_ID") == F.col("sc.SKU_ID"),
        "inner"
    )
    .select(
        "SKU_ID",
        "CATEGORY_INTERNAL_ID",
        "CATEGORY_NAME",
        "ATTRIBUTE_ID",
        "ATTRIBUTE_DESCRIPTION",
        "ATTRIBUTE_VALUE_ID",
        "ATTRIBUTE_VALUE_DESCRIPTION"
    )
    .dropna(subset=["ATTRIBUTE_DESCRIPTION", "ATTRIBUTE_VALUE_DESCRIPTION"])
    .dropDuplicates()
)

# COMMAND ----------

# DBTITLE 1,Step 3: Aggregate Sales at SKU Level
# ------------------------------------------------------------------------------------
# "How much did each SKU sell in total?"
# ------------------------------------------------------------------------------------
sku_sales = (
    purchaseHistory
    .groupBy("CV_PRODUCT_ITEM_ID")
    .agg(
        F.sum("PURCHASE_SPENT").alias("TOTAL_SPENT")
    )
    .withColumnRenamed("CV_PRODUCT_ITEM_ID", "SKU_ID")
)

# COMMAND ----------

# DBTITLE 1,Step 4: Attach SKU Sales to Attribute Rows
# ------------------------------------------------------------------------------------
# Each SKU's total sales are copied to ALL of its attributes.
# Interpretation: "Sales that occurred on SKUs which have this attribute-value"
# ------------------------------------------------------------------------------------
cav_with_sales = (
    category_attribute_value
    .join(sku_sales, "SKU_ID", "left")
    .select(
        "CATEGORY_INTERNAL_ID",
        "CATEGORY_NAME",
        "ATTRIBUTE_ID",
        "ATTRIBUTE_DESCRIPTION",
        "ATTRIBUTE_VALUE_ID",
        "ATTRIBUTE_VALUE_DESCRIPTION",
        F.coalesce("TOTAL_SPENT", F.lit(0.0)).alias("TOTAL_SPENT"),
        "SKU_ID"
    )
)

# COMMAND ----------

# DBTITLE 1,Step 5: Category × Attribute × Value Sales
# ------------------------------------------------------------------------------------
# "Among SKUs with THIS attribute value, how much total sales occurred?"
# ------------------------------------------------------------------------------------
category_attribute_sales = (
    cav_with_sales
    .groupBy(
        "CATEGORY_INTERNAL_ID",
        "CATEGORY_NAME",
        "ATTRIBUTE_ID",
        "ATTRIBUTE_DESCRIPTION",
        "ATTRIBUTE_VALUE_ID",
        "ATTRIBUTE_VALUE_DESCRIPTION"
    )
    .agg(
        F.sum("TOTAL_SPENT").alias("ATTR_TOTAL_SPENT"),
        F.countDistinct("SKU_ID").alias("ATTR_SKU_COUNT")
    )
)

# COMMAND ----------

# DBTITLE 1,Step 5.5: Attribute Coverage
# ------------------------------------------------------------------------------------
# High coverage = attribute appears on many SKUs (may behave like a constant).
# ------------------------------------------------------------------------------------
attribute_coverage = (
    category_attribute_value
    .groupBy(
        "CATEGORY_INTERNAL_ID",
        "ATTRIBUTE_ID",
        "ATTRIBUTE_DESCRIPTION"
    )
    .agg(
        F.countDistinct("SKU_ID").alias("ATTRIBUTE_COVERAGE_SKU_COUNT")
    )
)

# COMMAND ----------

# DBTITLE 1,Step 5.6: Attribute Value Diversity
# ------------------------------------------------------------------------------------
# High diversity = attribute varies meaningfully (e.g., FLAVOR).
# Low diversity = boilerplate (e.g., POTASSIUM).
# ------------------------------------------------------------------------------------
attribute_value_diversity = (
    category_attribute_value
    .groupBy(
        "CATEGORY_INTERNAL_ID",
        "ATTRIBUTE_ID",
        "ATTRIBUTE_DESCRIPTION"
    )
    .agg(
        F.countDistinct("ATTRIBUTE_VALUE_ID").alias("VALUE_DIVERSITY")
    )
)

# COMMAND ----------

# DBTITLE 1,Step 5.7: Max Value Dominance per Attribute × Category
# ------------------------------------------------------------------------------------
# For each attribute in a category, what fraction of SKUs share the MOST
# common value?  High dominance → attribute looks diverse but one value
# dominates (e.g., COLOR = "White" on 92% of SKUs).
# ------------------------------------------------------------------------------------
_val_sku_counts = (
    category_attribute_value
    .groupBy("CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID", "ATTRIBUTE_VALUE_ID")
    .agg(F.countDistinct("SKU_ID").alias("VAL_SKU_COUNT"))
)

_attr_total_skus = (
    _val_sku_counts
    .groupBy("CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID")
    .agg(F.sum("VAL_SKU_COUNT").alias("ATTR_TOTAL_SKU"))
)

attribute_value_dominance = (
    _val_sku_counts
    .join(_attr_total_skus, ["CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID"], "inner")
    .withColumn("VAL_FRACTION", F.col("VAL_SKU_COUNT") / F.col("ATTR_TOTAL_SKU"))
    .groupBy("CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID")
    .agg(F.max("VAL_FRACTION").alias("MAX_VALUE_DOMINANCE"))
)

# COMMAND ----------

# DBTITLE 1,Step 6: Collapse to Category × Attribute (with Coverage & Diversity)
attribute_sales = (
    category_attribute_sales
    .groupBy(
        "CATEGORY_INTERNAL_ID",
        "CATEGORY_NAME",
        "ATTRIBUTE_ID",
        "ATTRIBUTE_DESCRIPTION"
    )
    .agg(
        F.sum("ATTR_TOTAL_SPENT").alias("ATTRIBUTE_TOTAL_SPENT"),
        F.sum("ATTR_SKU_COUNT").alias("ATTRIBUTE_VALUE_SKU_COUNT")
    )
    .join(
        attribute_coverage,
        ["CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID", "ATTRIBUTE_DESCRIPTION"],
        "left"
    )
    .join(
        attribute_value_diversity,
        ["CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID", "ATTRIBUTE_DESCRIPTION"],
        "left"
    )
    .join(
        attribute_value_dominance,
        ["CATEGORY_INTERNAL_ID", "ATTRIBUTE_ID"],
        "left"
    )
)

# COMMAND ----------

# DBTITLE 1,Step 7: Category Totals (Normalization Denominator)
# Total SKU count per category (from the SKU→Category mapping)
category_sku_counts = (
    sku_to_category
    .groupBy("CATEGORY_INTERNAL_ID")
    .agg(
        F.countDistinct("SKU_ID").alias("CATEGORY_SKU_COUNT")
    )
)

category_totals = (
    attribute_sales
    .groupBy("CATEGORY_INTERNAL_ID", "CATEGORY_NAME")
    .agg(
        F.sum("ATTRIBUTE_TOTAL_SPENT").alias("CATEGORY_TOTAL_SPENT")
    )
    .join(category_sku_counts, "CATEGORY_INTERNAL_ID", "left")
)

# COMMAND ----------

# DBTITLE 1,Step 8: Final Attribute Weights (Coverage + Diversity Aware)
# ------------------------------------------------------------------------------------
# - Penalize attributes that appear everywhere (high coverage)
# - Preserve attributes that vary meaningfully (high value diversity)
# - Soft correction, NOT hard filtering
# ------------------------------------------------------------------------------------

category_window = Window.partitionBy("CATEGORY_INTERNAL_ID")

final_df = (
    attribute_sales
    .join(
        category_totals,
        ["CATEGORY_INTERNAL_ID", "CATEGORY_NAME"],
        "left"
    )
    .withColumn(
        "COVERAGE_RATIO",
        F.col("ATTRIBUTE_COVERAGE_SKU_COUNT") /
        F.max("ATTRIBUTE_COVERAGE_SKU_COUNT").over(category_window)
    )
    .withColumn(
        "ADJUSTED_ATTRIBUTE_SALES",
        F.col("ATTRIBUTE_TOTAL_SPENT") *
        F.log1p(F.col("VALUE_DIVERSITY")) *
        (1 - F.col("COVERAGE_RATIO"))
    )
    .withColumn(
        "ATTRIBUTE_WEIGHT",
        F.when(
            F.col("CATEGORY_TOTAL_SPENT") > 0,
            F.col("ADJUSTED_ATTRIBUTE_SALES") / F.col("CATEGORY_TOTAL_SPENT")
        ).otherwise(F.lit(0.0))
    )
    .orderBy(
        "CATEGORY_NAME",
        F.desc("ATTRIBUTE_WEIGHT")
    )
)

# COMMAND ----------

# DBTITLE 1,Save to Parquet
final_df.write.mode("overwrite").parquet("dbfs:/FileStore/amogh/cdt/final_df")

# COMMAND ----------

# DBTITLE 1,Preview Final Data
display(final_df)

# COMMAND ----------

# DBTITLE 1,Preview: Pizza Categories
display(final_df.filter(F.col("CATEGORY_NAME").like("%-PIZZA%")))

# COMMAND ----------

# DBTITLE 1,Preview: Beverage Categories
display(final_df.filter(F.col("CATEGORY_NAME").like("%-BEVERAGES%")))

# COMMAND ----------

# DBTITLE 1,Preview: Chicken Categories
display(final_df.filter(F.col("CATEGORY_NAME").like("%-CHICKEN%")))

# COMMAND ----------

# DBTITLE 1,Preview: Toothpaste Category
display(final_df.filter(F.col("CATEGORY_NAME").like("%10900-HB TOOTHPASTE%")))

# COMMAND ----------

# DBTITLE 1,Preview: Nuts Category
display(final_df.filter(F.col("CATEGORY_NAME").like("47700-GR NUTS%")))

# COMMAND ----------

# DBTITLE 1,Preview: Snacks Category
display(final_df.filter(F.col("CATEGORY_NAME").like("%54400-SG SNACKS%")))

# COMMAND ----------

# DBTITLE 1,Preview: Frozen Pasta Category
display(final_df.filter(F.col("CATEGORY_NAME").like("%00200-FR FROZEN PASTA%")))
