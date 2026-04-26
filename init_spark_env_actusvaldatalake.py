# Databricks notebook source
# #DATA_LAKE_STORAGE_NAME = 'actvaldatalake'
# spark.conf.set("fs.azure.account.auth.type.actvaldatalake.dfs.core.windows.net", "OAuth")
# spark.conf.set("fs.azure.account.oauth.provider.type.actvaldatalake.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
# spark.conf.set("fs.azure.account.oauth2.client.id.actvaldatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-client-id"))
# spark.conf.set("fs.azure.account.oauth2.client.secret.actvaldatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-credential"))
# spark.conf.set("fs.azure.account.oauth2.client.endpoint.actvaldatalake.dfs.core.windows.net", "https://login.microsoftonline.com/"+dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-tenant-id") +"/oauth2/token")

# spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")
# spark.conf.set("spark.executor.heartbeatinterval","10000000")
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold","-1")

# COMMAND ----------

# #DATA_LAKE_STORAGE_NAME = 'actusprddatalake'
# spark.conf.set("fs.azure.account.auth.type.actusprddatalake.dfs.core.windows.net", "OAuth")
# spark.conf.set("fs.azure.account.oauth.provider.type.actusprddatalake.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
# spark.conf.set("fs.azure.account.oauth2.client.id.actusprddatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-client-id"))
# spark.conf.set("fs.azure.account.oauth2.client.secret.actusprddatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-credential"))
# spark.conf.set("fs.azure.account.oauth2.client.endpoint.actusprddatalake.dfs.core.windows.net", "https://login.microsoftonline.com/"+dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-tenant-id") +"/oauth2/token")

# spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")
# spark.conf.set("spark.executor.heartbeatinterval","10000000")
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold","-1")

# COMMAND ----------

# #DATA_LAKE_STORAGE_NAME = 'actusvaldatalake'
spark.conf.set("fs.azure.account.auth.type.actusvaldatalake.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.actusvaldatalake.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.actusvaldatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-client-id"))
spark.conf.set("fs.azure.account.oauth2.client.secret.actusvaldatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-credential"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint.actusvaldatalake.dfs.core.windows.net", "https://login.microsoftonline.com/"+dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-tenant-id") +"/oauth2/token")

spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")
spark.conf.set("spark.executor.heartbeatinterval","10000000")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold","-1")


# COMMAND ----------

# #DATA_LAKE_STORAGE_NAME = 'cvdevdatalake'


# spark.conf.set("fs.azure.account.auth.type.cvdevdatalake.dfs.core.windows.net", "OAuth")
# spark.conf.set("fs.azure.account.oauth.provider.type.cvdevdatalake.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
# spark.conf.set("fs.azure.account.oauth2.client.id.cvdevdatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-client-id"))
# spark.conf.set("fs.azure.account.oauth2.client.secret.cvdevdatalake.dfs.core.windows.net", dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-credential"))
# spark.conf.set("fs.azure.account.oauth2.client.endpoint.cvdevdatalake.dfs.core.windows.net", "https://login.microsoftonline.com/"+dbutils.secrets.get(scope = "key-vault-secrets", key = "sp-tenant-id") +"/oauth2/token")

# spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")
# spark.conf.set("spark.executor.heartbeatinterval","10000000")
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold","-1")

# COMMAND ----------

import numpy as np
import pandas as pd
import datetime
import dateutil

import pyspark.sql.functions as func
