import os
import requests
import json
import shutil
import time
from pathlib import Path
from flask import Flask, request
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate


app = Flask(__name__)
CORS(app)  # This will allow all domains by default

folder_path = "db"
cached_llm = Ollama(model="llama3", base_url="http://ollama.trahman.me")
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are Talha's friendly assistant. Your task is to answer questions or provide information about Talha based on the context provided. Keep your responses concise, no more than three lines. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

@app.route('/askme', methods=['POST'])
def ask():
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")
    # Simulate a delay
    time.sleep(3)

    load_data()
    
    # Response JSON
    response = {
        "answer": "According to the context, Talha has friends named Riyaz, Shanto, and Muntasir. They spend their weekends together, engaging in activities like gaming sessions, movie marathons, and lively debates about tech trends."
    }
    
    return response

@app.route("/ask", methods=["POST"])
def askJsonPost():
    print("Post /ask called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    load_data()

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    response_answer = {"answer": result.get("answer", "")}
    return response_answer


def load_data():
    new_data = False
    # Define the directory containing the files
    data_directory = 'data'
    response = requests.get("http://api.trahman.me/api/blogs")
    if response.status_code != 200:
        print({"error": "Failed to fetch data from API"})
        exit()

    # Extract data from the JSON
    json_data = response.json()
    blogs = json_data.get("data", [])

    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Process each blog entry
    for blog in blogs:
        blog_id = blog.get("id", "unknown")
        
        # Define file paths
        txt_file_path = os.path.join(data_directory, f'{blog_id}.txt')
        old_txt_file_path = os.path.join(data_directory, f'{blog_id}.txt.old')

        # Check if the .txt.old file exists for this ID
        if os.path.exists(old_txt_file_path):
            print(f"Skipping {blog_id} as .txt.old file already exists")
            continue

        # Extract and save details to a .txt file
        new_data = True
        details = blog.get("attributes", {}).get("details", "")
        with open(txt_file_path, 'w') as f:
            f.write(details)

        print(f"Saved details for blog ID {blog_id}")

    if not new_data:
        return

    # loader = DirectoryLoader('./data', glob='**/*.json', show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.attributes', 'text_content':False})
    loader = DirectoryLoader('./data', glob='**/*.txt', show_progress=True)
    
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    # Load the chunks into a vector store
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    # Iterate over all files in the data directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.txt'):
            # Define file paths
            file_path = os.path.join(data_directory, filename)
            new_file_path = os.path.join(data_directory, filename + '.old')
            
            # Move the file
            shutil.move(file_path, new_file_path)
            print(f"Moved '{file_path}' to '{new_file_path}'")



def start_app():
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    start_app()
