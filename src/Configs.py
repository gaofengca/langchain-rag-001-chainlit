# config.py
import os

# LLM
LLAMA_MODEL_PATH = "path_to_llama_model"  # Path to your local Llama 8b model
OLLAMA_MODEL = "llama3:latest"

# Lance DB
LANCEDB_URI = "E:/lanceDB"   # Path to store LanceDB vectors
if os.name == 'posix': # check if the current os is macos
    LANCEDB_URI = "/Users/feng/ai/vector_db"
TABLE_NAME = "TEST_TABLE"
## index
METRIC = 'l2'
INDEX_TYPE = 'IVF_HNSW_SQ'
M = 16 # Number of neighbors to maintain for each node
EF_CONSTRUCTION = 200 # Larger value increases index quality, but slower construction

# splitter
CHUNK_SIZE = 1000  # Text chunk size
CHUNK_OVERLAP = 20  # Text chunk overlap
MAX_TOKENS = 4096  # Max tokens for Llama model
SPACE_SEPARATOR = " "

# query
QUERY_LIMIT = 10
DEDUPLICATION_LIMIT = 5
SIMILARITY_THRESHOLD = 0.95
