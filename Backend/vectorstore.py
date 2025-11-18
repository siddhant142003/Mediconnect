
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings

PINECONE_API_KEY = "pcsk_2opvqo_S2iK86qTXpwYZT819vyvMVymYjCSS34CZ76mhNJorahitiXhS8JLr7b8joZjyhS"
index_name = "patient-index"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index
index = pc.Index(index_name)

# Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
