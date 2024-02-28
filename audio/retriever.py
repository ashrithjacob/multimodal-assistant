import os
import numpy as np
import chromadb
import json
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from datetime import datetime

# Get current date and time


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class VectorStore:
    def __init__(self) -> None:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-large"
        )
        # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        # setup Chroma in-memory, for easy prototyping. Can add persistence easily!
        # self.client = chromadb.Client(Settings(persist_directory="./"))
        self.client = chromadb.PersistentClient(path="./")
        # Create collection. get_collection, get_or_create_collection, delete_collection also available!
        self.collection = self.client.get_or_create_collection(
            "feature_store", embedding_function=openai_ef
        )

    def get_entry_numbers(self):
        number_of_entries = self.collection.count()
        return number_of_entries

    def add(self, features):
        number_of_entries = self.get_entry_numbers()
        self.collection.add(
            documents=[
                str(features)
            ],  # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            # metadatas=[],  # filter on these!
            ids=[str(number_of_entries)],  # unique for each document
        )

    def retrieve(self, query, n=2):
        results = self.collection.query(
            query_texts=query,
            n_results=n,
            # where={"metadata_field": "is_equal_to_this"}, # optional filter
            # where_document={"$contains":"search_string"}  # optional filter
        )
        return results

    def injest_chunks(self, chunks):
        for chunk in chunks:
            self.add(chunk)

    def extract_text_blocks(self, txt_file_path, n):
        text_blocks = []
        chunk = ""
        with open(txt_file_path, "r") as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                text_part = line.split("text:")[1].strip()
                chunk += text_part + " "
                if idx % n == 2:
                    text_blocks.append(chunk)
                    chunk = ""
                elif idx + n >= len(lines):
                    text_blocks.append(chunk)
        return text_blocks

    def extract_text(self, line):
        text = line.split("|")[-1]
        text = text.split("text:")[-1].strip()
        return text


class Retriever:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_single_timestamp",
                "description": "Get the description of a conversation at a specific timestamp",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "The timestamp to get the description for in dd/mm/yyyy hh:mm:ss format",
                        },
                    },
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_start_end_timestamp",
                "description": "Get the start and end timestamp of a conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "string",
                            "description": "the start time mentioned provided in dd/mm/yyyy hh:mm:ss format",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "End time mentioned provided in dd/mm/yyyy hh:mm:ss format",
                        },
                    },
                    "required": ["start_time", "end_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_topic",
                "description": "Get the topic of the conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic of conversation",
                        },
                    },
                    "required": ["topic"],
                },
            },
        },
    ]
    GPT_MODEL = "gpt-3.5-turbo-0613"
    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                seed =123,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e
        
    def run(prompt):
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%d/%m/%Y")
        print(formatted_date)
        messages = []
        messages.append(
            {
                "role": "system",
                "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
            }
        )
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": f"For reference today is {formatted_date}"})
        chat_response = Retriever.chat_completion_request(messages, tools=Retriever.tools)
        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message)
        fn_name=assistant_message.tool_calls[0].function.name
        args=json.loads(assistant_message.tool_calls[0].function.arguments)
        print(args)
        print(fn_name)

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def extract_text(line):
    text = line.split("|")[-1]
    text = text.split("text:")[-1].strip()
    return text


def read_first_three_lines(file_path):
    # Open the file in read mode
    with open(file_path, "r", encoding="utf-8") as file:
        # Read the first three lines
        first_three_lines = [extract_text(file.readline().strip()) for _ in range(3)]
    return first_three_lines


def get_cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)


if __name__ == "__main__":
    text_file = "./text/sharktank.txt"
    prompt = "For how long did we talk about sand castles?"
    Retriever.run(prompt)
    #store = VectorStore()
    # chunks = store.extract_text_blocks(txt_file_path=text_file, n=3)
    # store.injest_chunks(chunks)
    #print(store.retrieve(prompt, 2))
