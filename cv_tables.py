
class CvTables():
    def __init__(self, spark, resources_path, retailer_id):
        self.spark = spark
        self.resources_path = resources_path
        self.retailer_id = retailer_id
        
    def init_purchase_history(self):
        purchaseHistory = self.spark.read.parquet(self.resources_path + self.retailer_id + "/input/row/parquet/CV_R_STAT_P1_PURCHASE_HISTORY/*")
        purchaseHistory = purchaseHistory.where(purchaseHistory.CV_SHOPPER_ID > 0)
        return purchaseHistory

    def init_catalog_df(self):
        catalog_df = self.spark.read.parquet(self.resources_path + self.retailer_id +"/input/row/parquet/CV_R_CATALOG_TREE").distinct()
        return catalog_df
        
    def init_catalog_parent(self):
        return self.init_df("/input/row/parquet/CV_R_CATALOG_PARENTS")

    def init_sku_brand_df(self):
        sku_brand_df = self.spark.read.parquet(self.resources_path + self.retailer_id +"/input/row/parquet/CV_R_PROD_ADDL_INFO").distinct()
        return sku_brand_df
        
    def init_shopper_attributes(self):
        return self.init_df('/output/data_load/row/parquet/CV_R_STAT_P2_SHOPPING_ATTRIBUTES')
        
    def init_shopper_section_brand_attributes(self):
        return self.init_df('/output/data_load/row/parquet/CV_R_STAT_P2_SHOPPING_BRAND_ATTR')
    
    def init_shopper_virtual_cat_attributes(self):
        return self.init_df('/output/data_load/col/parquet/cv_r_virtual_category_attributes')
        
    def init_shopper_meta_attrs(self):
        return self.init_df('/output/data_load/col/parquet/cv_r_shpr_attributes_dim')
    
    def init_df(self, parquet_path):
        return self.spark.read.parquet(self.resources_path + self.retailer_id + parquet_path)
    
    def init_df_from_full_path(self, full_parquet_path):
        return self.spark.read.parquet(full_parquet_path)
    
    def add_section_brand(self, pur_df, catalog_df, sku_brand_df):
      pur_df = pur_df.join(catalog_df.where(catalog_df.PRODUCT_TREE_LEVEL == 0).select('INTERNAL_ID', 'CV_PARENT_ID').withColumnRenamed('CV_PARENT_ID', 'SECTION_ID'), pur_df.CV_PRODUCT_ITEM_ID == catalog_df.INTERNAL_ID, 'left')
      pur_df = pur_df.join(sku_brand_df.select('CV_PRODUCT_ITEM_ID', 'CV_BRAND_ID'), 'CV_PRODUCT_ITEM_ID', 'left')
      return pur_df
