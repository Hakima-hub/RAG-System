from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re
import ollama
import numpy as np

EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
max_documents = 15
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=700)

documents = []
chunks = []      
embeddings = []   

def embed(texts):
    return ollama.embed(model=EMBEDDING_MODEL, input=texts)["embeddings"]


def add_document():
    path = input("Path to .pdf or .txt file: ").strip().strip('"')

    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")

    new_chunks = splitter.split_documents(loader.load())

    for chunk in new_chunks:
        chunk.page_content = re.sub(r"\s+", " ", chunk.page_content).strip()

    chunks.extend(new_chunks)
    embeddings.extend(embed([c.page_content for c in new_chunks]))
    print(f"Added {len(new_chunks)} chunks. Total: {len(chunks)}")

def add_documents():
    while len(documents) < max_documents:
        left = max_documents - len(documents)
        path = input(f"Path ({left} left, press Enter to finish): ").strip().strip('"')
        if not path:
            break
        if path in documents:
            print("Already added.")
            continue
        try:
            add_document(path)
        except Exception as error:
            print(f"Could not add that file: {error}")
 
    if len(documents) >= max_documents:
        print(f"Limit of {max_documents} documents reached.")
    print(f"Documents: {len(documents)} | Chunks: {len(chunks)}")
    
def Enter_query():
    if not chunks:
        print("Add  document(s).")
        return

    query = input("Enter your query: ")
    user_query_embedding = embed([query])

    scores = cosine_similarity(user_query_embedding, np.array(embeddings))[0]
    top_k = np.argsort(scores)[::-1][:3]   

    for index in top_k:
        print(f"\nScore: {scores[index]:.3f}")
        print(chunks[index].page_content)


while True:
    choice = input("\n1) Add document(s) \n2) Enter a query  \n3) Close: ")
    if choice == "1":
        add_document()
    elif choice == "2":
        Enter_query()
    elif choice == "3":
        break