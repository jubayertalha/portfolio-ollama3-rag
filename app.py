import os
import requests
import json
from pathlib import Path
from flask import Flask, request
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

folder_path = "db"
cached_llm = Ollama(model="llama3", base_url="http://ollama.trahman.me")
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a friendly assistant good at analysing json like documents. Provide short and near to precise answer no more than three lines. Don't add raw context data. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

@app.route("/ask", methods=["POST"])
def askJsonPost():
    print("Post /ask called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    # Fetch JSON data from the API
    response = requests.get("http://api.trahman.me/api/blogs")
    if response.status_code != 200:
        return {"error": "Failed to fetch data from API"}

    # Extract data from the JSON
    json_data = response.json()
    blogs = json_data.get("data", [])

    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Save each blog entry to a file named with its ID
    for blog in blogs:
        blog_id = blog.get("id", "unknown")
        file_path = os.path.join('data', f'{blog_id}.json')
        with open(file_path, 'w') as f:
            json.dump(blog, f)

    loader = DirectoryLoader('./data', glob='**/*.json', show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.attributes', 'text_content':False})
    
    docs = loader.load()
    print(f"docs len={len(docs)}")

    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    # Load the chunks into a vector store
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    # Create the retriever and chain
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



def start_app():
    app.run(host="0.0.0.0", port=80, debug=True)


if __name__ == "__main__":
    start_app()
