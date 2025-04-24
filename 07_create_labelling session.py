# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import yaml
import pandas as pd
from databricks import agents

# COMMAND ----------

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)



# COMMAND ----------

mlflow.set_experiment(config['mlflow']['experiment_name'])


# COMMAND ----------

from databricks.agents import review_app
import mlflow

# The review app is tied to the current MLFlow experiment.
my_app = review_app.get_review_app()

# You can use the following code to remove any existing agents.
# for agent in list(my_app.agents):
#     my_app.remove_agent(agent.agent_name)

# Add the llama3 70b model serving endpoint for labeling. You should replace this with your own model serving endpoint for your
# own agent.
# NOTE: An agent is required when labeling an evaluation dataset.
my_app.add_agent(
    agent_name="benmackenzie_catalog-cookbook-kumo_bot_3",
    model_serving_endpoint="agents_benmackenzie_catalog-cookbook-kumo_bot",
)

# Create a labeling session and collect guidelines and/or expected-facts from SMEs.
# Note: Each assigned user is given QUERY access to the serving endpoint above and write access.
# to the MLFlow experiment.
my_session = my_app.create_labeling_session(
  name="problem_responses",
  agent="benmackenzie_catalog-cookbook-kumo_bot_3",
  assigned_users = ["ben.mackenzie@databricks.com"],
  label_schemas = [review_app.label_schemas.GUIDELINES, review_app.label_schemas.EXPECTED_FACTS]
)

# Add the records from the dataset to the labeling session.
# Note: Each assigned user above is given SELECT access to the UC delta table.
#my_session.add_dataset("benmackenzie_catalog.cookbook.dataset_a")

# Share the following URL with your SMEs for them to bookmark. For the given review app linked to an experiment, this URL never changes.
print(my_app.url)

# You can also link them directly to the labeling session URL, however if you
# request new labeling sessions from SMEs there will be new URLs. Use the review app
# URL above to keep a permanent URL.
print(my_session.url)
