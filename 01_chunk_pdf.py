# Databricks notebook source
# MAGIC %md
# MAGIC # POC PDF Data Preparation Pipeline
# MAGIC
# MAGIC This is an example notebook that provides a **starting point** for building a POC data pipeline which uses the configuration from: 
# MAGIC - Loads and parses PDF files from a UC Volume.
# MAGIC - Chunks the data.
# MAGIC - Converts the chunks into embeddings.
# MAGIC - Stores the embeddings in a Databricks Vector Search Index. 
# MAGIC
# MAGIC After you have initial feedback from your stakeholders, you can easily adapt and tweak this pipeline to fit more advanced logic. For example, if your retrieval quality is low, you could experiment with different parsing and chunk sizes once you gain working knowledge of your data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries & import packages

# COMMAND ----------

# MAGIC %pip install -qqqq -U pypdf==4.1.0 databricks-vectorsearch transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.2 mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pypdf import PdfReader
from typing import TypedDict, Dict
import warnings
import io 
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from functools import partial
import tiktoken
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configuration

# COMMAND ----------


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config = config['data']
data_pipeline_config = config['pipeline_config']
embedding_config = config['embedding_config']
tables_config = config['tables_config']


raw_files_table_name = tables_config['raw_files_table_name']
parsed_docs_table_name = tables_config['parsed_docs_table_name']
chunked_docs_table_name = tables_config['chunked_docs_table_name']
source_path = tables_config['source_path']






# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load the files from the UC Volume
# MAGIC
# MAGIC In Bronze/Silver/Gold terminology, this is your Bronze table.
# MAGIC
# MAGIC **NOTE:** You will have to upload some PDF files to this volume. See the `sample_pdfs` folder of this repo for some example PDFs to upload to the UC Volume.
# MAGIC
# MAGIC TODO: Another notebook to load sample PDFs if the customer does't have them

# COMMAND ----------

# Load the raw riles
raw_files_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.pdf")
    .load(source_path)
)

# Save to a table
raw_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    raw_files_table_name
)

# reload to get correct lineage in UC
raw_files_df = spark.read.table(raw_files_table_name)

# For debugging, show the list of files, but hide the binary content
display(raw_files_df.drop("content"))

# Check that files were present and loaded
if raw_files_df.count() == 0:
    display(
        f"`{source_path}` does not contain any files.  Open the volume and upload at least file."
    )
    raise Exception(f"`{source_path}` does not contain any files.")





# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Parse the PDF files into text
# MAGIC
# MAGIC In Bronze/Silver/Gold terminology, this is your Silver table.
# MAGIC
# MAGIC Although not reccomended for your POC, if you want to change the parsing library or adjust it's settings, modify the contents of the `parse_bytes_pypdf` UDF.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### PyPDF based parsing function
# MAGIC
# MAGIC This function parses an individual PDF with `pypdf` library.

# COMMAND ----------

from typing import TypedDict
from datetime import datetime
import warnings
import io
import traceback
import os
from urllib.parse import urlparse

# PDF libraries
from pypdf import PdfReader

# HTML libraries
from markdownify import markdownify as md
import markdownify
import re

## DOCX libraries
import pypandoc
import tempfile

# Schema of the dict returned by `file_parser(...)`
class ParserReturnValue(TypedDict):
    # DO NOT CHANGE THESE NAMES - these are required by Evaluation & Framework
    # Parsed content of the document
    doc_content: str  # do not change this name
    # The status of whether the parser succeeds or fails, used to exclude failed files downstream
    parser_status: str  # do not change this name
    # Unique ID of the document
    doc_uri: str  # do not change this name

    # OK TO CHANGE THESE NAMES
    # Optionally, you can add additional metadata fields here
    last_modified: datetime


# Parser function.  Replace this function to provide custom parsing logic.
def file_parser(
    raw_doc_contents_bytes: bytes,
    doc_path: str,
    modification_time: datetime,
    doc_bytes_length: int,
) -> ParserReturnValue:
    """
    Parses the content of a PDF document into a string.

    This function takes the raw bytes of a PDF document and its path, attempts to parse the document using PyPDF,
    and returns the parsed content and the status of the parsing operation.

    Parameters:
    - raw_doc_contents_bytes (bytes): The raw bytes of the document to be parsed (set by Spark when loading the file)
    - doc_path (str): The DBFS path of the document, used to verify the file extension (set by Spark when loading the file)
    - modification_time (timestamp): The last modification time of the document (set by Spark when loading the file)
    - doc_bytes_length (long): The size of the document in bytes (set by Spark when loading the file)

    Returns:
    - ParserReturnValue: A dictionary containing the parsed document content and the status of the parsing operation.
      The 'doc_content' key will contain the parsed text as a string, and the 'parser_status' key will indicate
      whether the parsing was successful or if an error occurred.
    """
    try:
        filename, file_extension = os.path.splitext(doc_path)

        if file_extension == ".pdf":
            pdf = io.BytesIO(raw_doc_contents_bytes)
            reader = PdfReader(pdf)

            parsed_content = [
                page_content.extract_text() for page_content in reader.pages
            ]

            parsed_document = {
                "doc_content": "\n".join(parsed_content),
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".html":
            from markdownify import markdownify as md

            html_content = raw_doc_contents_bytes.decode("utf-8")

            markdown_contents = md(
                str(html_content).strip(), heading_style=markdownify.ATX
            )
            markdown_stripped = re.sub(r"\n{3,}", "\n\n", markdown_contents.strip())

            parsed_document = {
                "doc_content": markdown_stripped,
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".docx":
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file.write(raw_doc_contents_bytes)
                temp_file_path = temp_file.name
                md = pypandoc.convert_file(temp_file_path, "markdown", format="docx")

                parsed_document = {
                    "doc_content": md.strip(),
                    "parser_status": "SUCCESS",
                }
        else:
            raise Exception(f"No supported parser for {doc_path}")

        # Extract the required doc_uri
        # convert from `dbfs:/Volumes/catalog/schema/pdf_docs/filename.pdf` to `Volumes/catalog/schema/pdf_docs/filename.pdf`
        modified_path = urlparse(doc_path).path.lstrip('/')
        parsed_document["doc_uri"] = modified_path


        # Add the modified time
        parsed_document["last_modified"] = modification_time

        return parsed_document

    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "doc_content": "",
            "parser_status": f"ERROR: {status}",
        }

# COMMAND ----------

# MAGIC %run ./utils/typed_dicts_to_spark_schema

# COMMAND ----------

from utils.file_loading import load_files_to_df, apply_parsing_udf

# COMMAND ----------

raw_files_df = load_files_to_df(
    spark=spark,
    source_path=source_path,
)

parsed_files_df = apply_parsing_udf(
    raw_files_df=raw_files_df,
    # Modify this function to change the parser, extract additional metadata, etc
    parse_file_udf=file_parser,
    # The schema of the resulting Delta Table will follow the schema defined in ParserReturnValue
    parsed_df_schema=typed_dicts_to_spark_schema(ParserReturnValue)
)

# Write to a Delta Table
parsed_files_df.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(DOCS_DELTA_TABLE)

# Display for debugging
print(f"Parsed {parsed_files_df.count()} documents.")
parsed_files_df.display()



# COMMAND ----------

# MAGIC %md
# MAGIC ### Parser UDF
# MAGIC
# MAGIC This UDF wraps your parser into a UDF so Spark can parallelize the data processing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the parsers in Spark
# MAGIC
# MAGIC This cell runs the configured parsers in parallel via Spark.  Inspect the outputs to verify that parsing is working correctly.

# COMMAND ----------

# Run the parsing
parsed_files_staging_df = raw_files_df.withColumn("parsing", parser_udf("content")).drop("content")


# Check and warn on any errors
errors_df = parsed_files_staging_df.filter(
    func.col(f"parsing.parser_status")
    != "SUCCESS"
)

num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} documents had parse errors.  Please review.")
    display(errors_df)

# Filter for successfully parsed files
parsed_files_df = parsed_files_staging_df.filter(parsed_files_staging_df.parsing.parser_status == "SUCCESS").withColumn("doc_parsed_contents", func.col("parsing.doc_parsed_contents")).drop("parsing")

# Write to Delta Table
parsed_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(parsed_docs_table_name)

# reload to get correct lineage in UC
parsed_files_df = spark.table(parsed_docs_table_name)

# Display for debugging
print(f"Parsed {parsed_files_df.count()} documents.")

display(parsed_files_df)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Chunk the parsed text
# MAGIC
# MAGIC
# MAGIC In Bronze/Silver/Gold terminology, this is your Gold table.
# MAGIC
# MAGIC Although not reccomended for your POC, if you want to change the chunking library or adjust it's settings, modify the contents of the `parse_bytes_pypdf` UDF.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recursive Token Based Text Splitter
# MAGIC Uses the embedding model's tokenizer to split the document into chunks.
# MAGIC
# MAGIC Per LangChain's docs: This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.
# MAGIC
# MAGIC Configuration parameters:
# MAGIC - `chunk_size_tokens`: Number of tokens to include in each chunk
# MAGIC - `chunk_overlap_tokens`: Number of tokens to overlap between chunks e.g., the last `chunk_overlap_tokens` tokens of chunk N are the same as the first `chunk_overlap_tokens` tokens of chunk N+1
# MAGIC
# MAGIC Docs: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
# MAGIC
# MAGIC IMPORTANT: You need to ensure that `chunk_size_tokens` + `chunk_overlap_tokens` is LESS THAN your embedding model's context window.

# COMMAND ----------

class ChunkerReturnValue(TypedDict):
    chunked_text: str
    chunker_status: str

def chunk_parsed_content_langrecchar(
    doc_parsed_contents: str, chunk_size: int, chunk_overlap: int, embedding_config
) -> ChunkerReturnValue:
    try:
        # Select the correct tokenizer based on the embedding model configuration
        if (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "hugging_face"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "tiktoken"
        ):
            tokenizer = tiktoken.encoding_for_model(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        chunks = text_splitter.split_text(doc_parsed_contents)
        return {
            "chunked_text": [doc for doc in chunks],
            "chunker_status": "SUCCESS",
        }
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            "chunked_text": [],
            "chunker_status": f"ERROR: {e}",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunker UDF
# MAGIC
# MAGIC This UDF wraps your chunker into a UDF so Spark can parallelize the data processing.

# COMMAND ----------

chunker_conf = data_pipeline_config.get("chunker")

chunker_udf = func.udf(
    partial(
        chunk_parsed_content_langrecchar,
        chunk_size=chunker_conf.get("config").get("chunk_size_tokens"),
        chunk_overlap=chunker_conf.get("config").get("chunk_overlap_tokens"),
        embedding_config=embedding_config,
    ),
    returnType=StructType(
        [
            StructField("chunked_text", ArrayType(StringType()), nullable=True),
            StructField("chunker_status", StringType(), nullable=True),
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the chunker in Spark
# MAGIC
# MAGIC This cell runs the configured chunker in parallel via Spark.  Inspect the outputs to verify that chunking is working correctly.

# COMMAND ----------

# Run the chunker
chunked_files_df = parsed_files_df.withColumn(
    "chunked",
    chunker_udf("doc_parsed_contents.parsed_content"),
)

# Check and warn on any errors
errors_df = chunked_files_df.filter(chunked_files_df.chunked.chunker_status != "SUCCESS")

num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} chunks had parse errors.  Please review.")
    display(errors_df)

# Filter for successful chunks
chunked_files_df = chunked_files_df.filter(chunked_files_df.chunked.chunker_status == "SUCCESS").select(
    "path",
    func.explode("chunked.chunked_text").alias("chunked_text"),
    func.md5(func.col("chunked_text")).alias("chunk_id")
)

# Write to Delta Table
chunked_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    chunked_docs_table_name
)


# Enable CDC for Vector Search Delta Sync
spark.sql(
    f"ALTER TABLE {chunked_docs_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

print(f"Produced a total of {chunked_files_df.count()} chunks.")

# Display without the parent document text - this is saved to the Delta Table
display(chunked_files_df)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step    
# MAGIC ####  Use UI to create a vector index or run 02_create_vector_search_index
