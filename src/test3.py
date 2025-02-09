import streamlit as st
from pinecone import Pinecone
import re
import pandas as pd
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder

# Read the data
df = pd.read_csv('src/preprocessed_data_cleaned.csv')

# Initialize BM25Encoder
bm25 = BM25Encoder()
bm25.fit(df['text_chunk'])

# Initialize Pinecone
pc = Pinecone(api_key="3661dc2a-3710-4669-a187-51faaa0cc557")
index = pc.Index("hybridsearch")

# Define OpenAI client
client = None  # Initialize to None until user provides API key

# Define the Streamlit application
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def find_keyword_chunks(text, keyword, chunk_surround=10):
    text = text.lower()
    keyword = keyword.lower()
    keyword_positions = [match.start() for match in re.finditer(keyword, text)]
    chunks = []
    for pos in keyword_positions:
        start = max(0, pos - chunk_surround)
        end = min(len(text), pos + chunk_surround + len(keyword))
        chunks.append(text[start:end])
    return chunks

def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    openai_api_key = st.text_input("Enter OpenAI API Key:")
    global client
    client = OpenAI(api_key=openai_api_key)

    if st.button("Search"):
        if search_text:
            try:
                dense = get_embedding(search_text)
                sparse = bm25.encode_queries(search_text)
                hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)

                query_result = index.query(
                    top_k=top_k,
                    vector=hdense,
                    sparse_vector=hsparse,
                    namespace='sparse',
                    include_metadata=True
                )

                if query_result and 'matches' in query_result:
                    results = []
                    displayed_urls = set()
                    for match in query_result['matches']:
                        url = match.get('metadata', {}).get('url', 'N/A')
                        if url in displayed_urls:
                            continue
                        displayed_urls.add(url)
                        score = match.get('score', 'N/A')
                        text = match.get('metadata', {}).get('text', '')
                        chunks_with_keyword = find_keyword_chunks(text, search_text, 100)
                        results.append({"Score": score, "Text":chunks_with_keyword, "URL": url})

                    if results:
                        st.table(pd.DataFrame(results))
                    else:
                        st.write("No results found.")
                else:
                    st.write("No results found.")
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Please enter search text.")

if __name__ == "__main__":
    main()
