# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import yaml
import pandas as pd
from databricks import agents

# COMMAND ----------

# MAGIC %md
# MAGIC # Load your evaluation set from the previous step

# COMMAND ----------

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)



# COMMAND ----------

EVALUATION_SET_FQN =  config['eval']['synthetic_evaluation_set_fqn']
df = spark.table(EVALUATION_SET_FQN)
eval_df = df.toPandas()
display(eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the POC application

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the MLflow run of the POC application 

# COMMAND ----------

mlflow.set_experiment(config['mlflow']['experiment_name'])


# COMMAND ----------

#need to do this for mlflow experiment to show up in UI.  You only need to run it once per experiment
with mlflow.start_run(run_name="evaluation-init") as run:
    mlflow.set_tag("version", "v1.0")
    

# COMMAND ----------

#Use mlflow nav bar to find the run, navigate to it and copy run_id
run_id = 'f1e99e7d29014f7685bd8d01b074387f'


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the correct Python environment for the POC's app
# MAGIC
# MAGIC TODO: replace this with env_manager=virtualenv once that works

# COMMAND ----------

pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation on the POC app

# COMMAND ----------


with mlflow.start_run(run_id=run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the evaluation results
# MAGIC
# MAGIC You can explore the evaluation results using the above links to the MLflow UI.  If you prefer to use the data directly, see the cells below.

# COMMAND ----------

# Summary metrics across the entire evaluation set
eval_results.metrics

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables['eval_results']

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)
