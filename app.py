from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import os
from multiprocessing import Pool
from tqdm import tqdm
import glob
from typing import List
from dotenv import load_dotenv
import pandas as pd
# model imports
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# model imports

from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS
import chromadb
nv_path = os.path.join(os.path.dirname(__file__), '.', '.env')
load_dotenv(dotenv_path=nv_path)

llm = None
filepath = os.environ['FILE_PATH']
file_list = os.listdir(filepath)
# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', '../application/static/files')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}
# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
# Chroma client
chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
app = Flask(__name__, template_folder='templates',static_folder='static')
# ==================================================================
#                   INGESTER
# ===================================================================
def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory, embeddings):
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()['documents']:
        return False
    return True
def ingester():
    global embeddings
    global chroma_client
    if does_vectorstore_exist(persist_directory, embeddings):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
    db.persist()
    db = None

    print(f"Ingestion complete!")
# ==================================================================
#                   INGESTER
# ===================================================================
# ==================================================================
#                   MODEL
# ===================================================================
def main(query):
    # Parse the command line arguments
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "GPT4All":
            global llm
            if llm:
                pass
            else:
                llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose GPT4All")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    res = qa(query)
    answer = res['result']
    return answer
# ==================================================================
#                   MODEL
# ===================================================================

def delete_document_with_embeddings(document):
    collection = chroma_client.get_collection(name="langchain")
    metadata_list = collection.get()['metadatas']
    for metadata in metadata_list:
        if document == metadata['source'].split('\\')[-1]:
            embedding_to_delete = metadata['source']
            collection.delete(
                where={"source": embedding_to_delete}
            )
            break
    file_to_delete = f"static/files/{document}"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"{file_to_delete} has been removed.")
    else:
        print(f"{file_to_delete} does not exist.")
    return 'file has been deleted'


@app.get('/')
def index():
    return render_template('index.html',files=file_list)
@app.route('/upload',methods=['POST'])
async def upload():
    uploaded_file=request.files['file']
    uploaded_file.save(f'static/files/{uploaded_file.filename}')
    ingester()
    global file_list
    file_list = os.listdir(filepath)
    return 'uploaded successfully'

@app.route('/delete_document',methods=['POST'])
def delete_document():
    file = request.get_json()
    try:
        delete_document_with_embeddings(file['document'])
        global file_list
        file_list = os.listdir(filepath)
        return {'status':200}
    except Exception as e:
        print(e)
        return {'status':400}

@app.route('/qanda',methods=['POST'])
def qanda():
    query = request.get_json()
    try:
        return main(query)
    except Exception as e:
        print(e)
        return 'sorry, I am not able to answer your query'

if __name__ == '__main__':
    app.run(debug=True)