from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

query1 = input("Enter the first word you'd like to embed:")
query2 = input("Enter the second word you'd like to embed:")
vector = embeddings.embed_query(query1)
vector2 = embeddings.embed_query(query2)
# prints first 10 vectors
print(f'The first word you entered is {query1}, it has a vector of length {len(vector)}')
print(vector[:10])
similarity = np.dot(vector, vector2) / (np.linalg.norm(vector) * np.linalg.norm(vector2))

print(f'The distance between the two words is {similarity:.4f}')