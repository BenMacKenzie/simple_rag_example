# Databricks notebook source
# MAGIC %md
# MAGIC ## Install libraries & import packages

# COMMAND ----------

# MAGIC %pip install -qqqq -U pypdf==4.1.0 databricks-vectorsearch transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.2 mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configuration

# COMMAND ----------

import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

vectorsearch_index_name =  config['rag_chain']['retriever_config']['vector_search_index']
chunked_docs_table_name = config['data']['tables_config']['chunked_docs_table_name']
vector_search_endpoint_name = config['rag_chain']['databricks_resources']['vector_search_endpoint_name']
embedding_endpoint_name = config['vector_search']['embedding_endpoint_name']
pipeline_type = config['vector_search']['pipeline_type']


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Embed documents & sync to Vector Search index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Get the vector search index
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create the Vector Index using api **OR** do this through the UI
# MAGIC

# COMMAND ----------

# DBTITLE 1,Index Management Workflow
force_delete = False


def find_index(endpoint_name, index_name):
    all_indexes = vsc.list_indexes(name=vector_search_endpoint_name).get("vector_indexes", [])
    return index_name in map(lambda i: i.get("name"), all_indexes)

if find_index(endpoint_name=vector_search_endpoint_name, index_name=vectorsearch_index_name):
    if force_delete:
        vsc.delete_index(endpoint_name=vector_search_endpoint_name, index_name=vectorsearch_index_name)
        create_index = True
    else:
        create_index = False
else:
    create_index = True

if create_index:
    print("Embedding docs & creating Vector Search Index, this can take 15 minutes or much longer if you have a larger number of documents.")
 

    vsc.create_delta_sync_index_and_wait(
        endpoint_name=vector_search_endpoint_name,
        index_name=vectorsearch_index_name,
        primary_key="chunk_id",
        source_table_name=chunked_docs_table_name,
        pipeline_type=pipeline_type,
        embedding_source_column="chunked_text",
        embedding_model_endpoint_name=embedding_endpoint_name
    )



# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the index

# COMMAND ----------

# DBTITLE 1,Testing the Index
index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vectorsearch_index_name)
index.similarity_search(columns=["chunked_text", "chunk_id", "path"], query_text="what is ARES?")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step:
# MAGIC ### Deploy the agent 03_deploy_agent
