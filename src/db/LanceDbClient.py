# LanceDbClient.py
from typing import List
import lancedb
from lancedb.table import Table
from scipy.spatial.distance import cosine

from log.Logger import Logger
from Configs import SIMILARITY_THRESHOLD, METRIC, INDEX_TYPE, M, EF_CONSTRUCTION, QUERY_LIMIT, TABLE_NAME, DEDUPLICATION_LIMIT
import pyarrow as pa


class LanceDbClient:
    def __init__(self, uri: str):
        self.logger = Logger().get_logger()
        self.db_uri = uri

    def get_db_table(self) -> Table:
        """
        Creates a LanceDB vector store and loads embeddings into it.

        Parameters:
        - db_path: Path to the LanceDB database file.

        Returns:
        - table: A LanceDB table.
        """

        # Create or open the LanceDB database
        db = lancedb.connect(
            uri = self.db_uri
        )

        # create or open table
        schema = pa.schema(
            [
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 4096)),
            ])
        if TABLE_NAME in db.table_names():
            table = db.open_table(TABLE_NAME)
        else:
            print('creating table...')
            table = db.create_table(
                TABLE_NAME,
                schema=schema
            )
            print('table created')
        return table

    def add_knowledge_base(self, embeddings: List[List[float]], inputs: List[str], table: Table):
        """
        Batch insert embeddings
        """

        batch = []
        for i in range(len(inputs)):
            text = inputs[i]
            vector = embeddings[i]
            if not self.highly_similar_to_existing(vector, text, table):
                batch.extend([{'text': text, 'vector': vector}])

        self.logger.info('original input chunks: %s, inserting after similarity checks: %s', len(inputs), len(batch))
        if len(batch) > 0:
            table.add(batch)

        # create index
        self.create_index(table)

    def search_vector_db(self, table: Table, query: List) -> List:
        """
        Search for the closest vectors in the vector database
        """
        table.distance_metric = "l2"
        return table.search(query, query_type='vector').limit(QUERY_LIMIT).select(['text']).to_list()

    def highly_similar_to_existing(self, embedding: List[float], text: str, table: Table) -> bool:
        # Retrieve top-k most similar vectors from LanceDB
        results = table.search(embedding).limit(DEDUPLICATION_LIMIT).select(['vector']).to_list()

        for result in results:
            # Calculate cosine similarity between the new embedding and the existing ones
            similarity = 1 - cosine(embedding, result['vector'])
            if similarity > SIMILARITY_THRESHOLD:
                self.logger.warning("Highly similar info detected for: %s, similarity=%f, skipping", text, similarity)
                return True
        return False

    def create_index(self, table: Table):
        table.create_index(metric = METRIC,
                           index_type = INDEX_TYPE,
                           m = M,
                           ef_construction = EF_CONSTRUCTION,
        )





