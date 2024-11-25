# Databricks notebook source
# MAGIC %md
# MAGIC ## Note that this is currenly in private preview

# COMMAND ----------

# MAGIC %pip install mlflow mlflow[databricks] databricks-agents
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.agents.eval import generate_evals_df
import pandas as pd
import tiktoken
import math
import yaml

# COMMAND ----------

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

tables_config = config['data']['tables_config']

parsed_docs_table_name = tables_config['parsed_docs_table_name']

EVALUATION_SET_FQN =  config['eval']['synthetic_evaluation_set_fqn']

# COMMAND ----------

docs = spark.sql(f"select path as doc_uri, doc_parsed_contents.parsed_content as content from {parsed_docs_table_name}")

# COMMAND ----------

display(docs)

# COMMAND ----------


# Update the guildelines as needed

guidelines = """
# Task Description
The Agent is a RAG chatbot that answers questions about about the company Kumo.ai  The Agent has access to a corpus of documents, and its task is to answer the user's questions by retrieving the relevant docs from the corpus and synthesizing a helpful, accurate response. 

# User personas
- An ML practioner who is interested in using Kumo.ai
- An experienced, highly technical Data Scientist or Data Engineer

# Example questions
- How does Kumo.ai solution differ from regular machine learning?
- What are the main advanatages of kumo.ai's solution?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

num_evals = 40

evals = generate_evals_df(
    docs,
    # The total number of evals to generate. The method attempts to generate evals that have full coverage over the documents
    # provided. If this number is less than the number of documents, is less than the number of documents,
    # some documents will not have any evaluations generated. See "How num_evals is used" below for more details.
    num_evals=num_evals,
    # A set of guidelines that help guide the synthetic generation. This is a free-form string that will be used to prompt the generation.
    guidelines=guidelines
)







# COMMAND ----------

evals_spark = spark.createDataFrame(evals)
evals_spark.write.format("delta").mode("overwrite").saveAsTable(EVALUATION_SET_FQN)

# COMMAND ----------

#baseline using 70B

def llama3_agent(input):
  client = mlflow.deployments.get_deploy_client("databricks")
  input.pop("databricks_options", None)
  return client.predict(endpoint="databricks-meta-llama-3-1-70b-instruct", inputs=input)

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results. You can use this as a baseline for your agent.
results = mlflow.evaluate(
  model=llama3_agent,
  data=evals,
  model_type="databricks-agent"
)


# COMMAND ----------

#now try with our agent
MODEL_SERVING_ENDPOINT_NAME = 'agents_benmackenzie_catalog-chatbots-kumo_bot'
def agent_fn(input):
  client = mlflow.deployments.get_deploy_client("databricks")
  return client.predict(endpoint=MODEL_SERVING_ENDPOINT_NAME, inputs=input)

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results. You can use this as a baseline for your agent.
results = mlflow.evaluate(
  model=agent_fn,
  data=evals,
  model_type="databricks-agent"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seems to be a bug when using serving endpoints.  For now run model locally

# COMMAND ----------

run_id = 'e089e4c0313c4500b06783af41409968'
pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------

results = mlflow.evaluate(
  model='runs:/e089e4c0313c4500b06783af41409968/chain',
  data=evals,
  model_type="databricks-agent"
)

