from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

loader = PyPDFLoader("app/loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf")
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
splits = text_splitter.split_documents(pages)
text_chunks = [doc.page_content for doc in splits]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = [model.encode(chunk) for chunk in text_chunks]
embedding_array = np.array(embeddings) #to store it into faiss
embedding_dim = embedding_array.shape[1]  # Dimension of your embeddings
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_array)

def retrieving_chunks(query):
    # Example query
    # query = "Quelles sont les prestations et le financement des services sociaux destinés aux étudiants dans le cadre de la vie universitaire ?"
    query_embedding = model.encode(query)  # Generate embedding for the query

    # Convert to NumPy array and reshape for FAISS
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Perform the search
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)
    matching_chunks = [text_chunks[i] for i in indices[0]]
    return '/n'.join(matching_chunks)

with open('./config.json') as config_file:
    config = json.load(config_file)
    
api_key = config['api_key']
client = Groq(
    api_key=api_key,
)

def llm_generation(question, context):
    chat_completion = client.chat.completions.create(
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": "repondre au question en se basant sur le context donnée"
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": "question : "+question+",context: "+context
            }
        ],
        model="llama3-70b-8192",
    )

    return chat_completion.choices[0].message.content



# global model
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# def loader(path = "./loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf"):
#     loader = PyPDFLoader(path)
#     return loader.load_and_split()
    
# def txt2chunks(text_laoder):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
#     splits = text_splitter.split_documents(text_laoder)
#     text_chunks = [doc.page_content for doc in splits]
#     return text_chunks

# def embedding(text_chunks):
#     embeddings = [model.encode(chunk) for chunk in text_chunks]
#     embedding_array = np.array(embeddings) #to store it into faiss

# def storing_embedding(embedding_array):
#     embedding_dim = embedding_array.shape[1]  # Dimension of your embeddings
#     index = faiss.IndexFlatL2(embedding_dim)
#     index.add(embedding_array)
#     return index

# def retrieving_chunks(query, index, text_chunks):
#     # Example query
#     # query = "Quelles sont les prestations et le financement des services sociaux destinés aux étudiants dans le cadre de la vie universitaire ?"
#     query_embedding = model.encode(query)  # Generate embedding for the query

#     # Convert to NumPy array and reshape for FAISS
#     query_embedding = np.array([query_embedding], dtype=np.float32)

#     # Perform the search
#     k = 3  # Number of nearest neighbors to retrieve
#     distances, indices = index.search(query_embedding, k)
#     matching_chunks = [text_chunks[i] for i in indices[0]]
#     context = '/n'.join(matching_chunks)
#     return context

# def llm_answer(question, context):
#     with open('config.json') as config_file:
#         config = json.load(config_file)
#         api_key = config['api_key']
#     client = Groq(
#     api_key=api_key,
#     )
#     chat_completion = client.chat.completions.create(
#         messages=[
#             # Set an optional system message. This sets the behavior of the
#             # assistant and can be used to provide specific instructions for
#             # how it should behave throughout the conversation.
#             {
#                 "role": "system",
#                 "content": "repondre au question en se basant sur le context donnée"
#             },
#             # Set a user message for the assistant to respond to.
#             {
#                 "role": "user",
#                 "content": "question : "+question+",context: "+context
#             }
#         ],
#         model="llama3-70b-8192",
#     )

#     return chat_completion.choices[0].message.content)
