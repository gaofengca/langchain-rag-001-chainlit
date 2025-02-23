# Main.py
import json
import os
from typing import List
import chainlit as cl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from Configs import MAX_TOKENS, LANCEDB_URI, OLLAMA_MODEL
from Splitter import Splitter
from db.LanceDbClient import LanceDbClient


class Main:
    def __init__(self, vector_db_uri):
        self.llama_model = self.load_llama_model()
        self.embed_model = self.load_embed_model()
        self.splitter = Splitter()
        self.lance_db_client = LanceDbClient(vector_db_uri)

    @staticmethod
    def load_llama_model():
        """
        Loads the local Llama model (8B) with the specified max tokens.
        """
        ollama_llm = OllamaLLM(model=OLLAMA_MODEL)
        ollama_llm.num_predict = MAX_TOKENS
        return ollama_llm

    @staticmethod
    def load_embed_model():
        """
        Loads Llama embed model
        """
        return OllamaEmbeddings(
            model=OLLAMA_MODEL,
            # base_url='http://127.0.0.1:11434'
        )

    def run_rag_pipeline(self, input_strings: List[str]):
        # Step 1: Chunk the input text into manageable pieces
        chunks = []
        for input_string in input_strings:
            chunks.extend(self.splitter.text_splitter.split_text(input_string))

        # Step 2: Embed the text chunks
        embeddings = self.embed_model.embed_documents(chunks)

        # Step 3: get vector database
        table = self.lance_db_client.get_db_table()

        # Step 4: Add knowledge base
        if len(chunks) > 0:
            self.lance_db_client.add_knowledge_base(embeddings=embeddings, inputs=chunks, table=table)

    async def query_func(self, query_str: str):
        table = self.lance_db_client.get_db_table()

        # Step 1: Generate embedding for the query
        print("Query:", query_str)
        embedded_query = self.embed_model.embed_query(query_str)

        # Step 2: Search the vector database for relevant chunks
        search_results = self.lance_db_client.search_vector_db(table=table, query=embedded_query)
        search_results_str = []
        if search_results is not None:
            search_results_str = [search_result.get('text') for search_result in search_results]

        print("Search Results:", search_results_str)

        # Step 3: Prepare the context for the Llama model and generate the answer
        final_prompt = f"Use the following context to answer the question:\n\n{search_results}\n\nQuestion: {query_str}"

        # print("Final Prompt:", final_prompt)
        # Step 4: Load the Llama model and generate the answer

        print("calling llama_mode...")
        answer = self.llama_model.invoke(final_prompt)
        print("Answer:", answer)
        print("-----------------")
        return answer


rag = Main(LANCEDB_URI)

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    print(message.content)
    answer = await rag.query_func(message.content)
    
    # Send a response back to the user
    await cl.Message(
        content=f"{answer}",
    ).send()