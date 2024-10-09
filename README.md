
##### A simplified example of creating and deploying a RAG agent based on https://github.com/databricks/genai-cookbook



### How to Use

You should not need to make any changes to the notebooks.  You only need to update the config.yaml file.

##### Option 1:
if you already have a vector search index:

1. update config.yaml
2. run 03_deploy_agent notebook
3. run 04_create_evaluation_set notebook
4. run 05_evaluate_agent_quality notebook

##### Option 2:
if you already have chunked content in a delta table:

1. update config.yaml
2. run 02_create_vector_search_index notebook (or create index through UI).
3. run 03_deploy_agent notebook
4. run 04_create_evaluation_set notebook
5. run 05_evaluate_agent_quality notebook


##### Option 3:
you are starting from scratch

1. update config.yaml
2. run 01_chunk_pdf notebook
3. run 02_create_vector_search_index notebook (or create index through UI).
4. run 03_deploy_agent notebook
5. run 04_create_evaluation_set notebook
6. run 05_evaluate_agent_quality notebook


