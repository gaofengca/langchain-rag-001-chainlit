from langchain_text_splitters import CharacterTextSplitter
from Configs import SPACE_SEPARATOR, CHUNK_OVERLAP, CHUNK_SIZE


class Splitter:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            separator = SPACE_SEPARATOR
        )



