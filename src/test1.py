import streamlit as st
from pinecone import Pinecone
import re
import pandas as pd
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder
import spacy

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Read the data
df = pd.read_csv('src/preprocessed_data_cleaned.csv')

# Initialize BM25Encoder
bm25 = BM25Encoder()
bm25.fit(df['text_chunk'])

openai_api_key = st.text_input("Enter openai api key")
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
pc = Pinecone(api_key="3661dc2a-3710-4669-a187-51faaa0cc557")
index = pc.Index("hybridsearch")

client = OpenAI(api_key=openai_api_key)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def highlight_search_word(text, search_text):
    highlighted_text = re.sub(f"({search_text})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return highlighted_text

def ensure_keyword_in_text(text, search_text):
    if search_text.lower() in text.lower():
        highlighted_text = highlight_search_word(text, search_text)
    else:
        words = text.split()
        extended_text = ' '.join(words[:10])
        highlighted_text = highlight_search_word(extended_text, search_text)
    return highlighted_text

def find_keyword_variations_chunks(text, keyword, chunk_surround=100):
    lower_text = text.lower()
    keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    keyword_positions = [match.start() for match in re.finditer(keyword_pattern, lower_text)]
    
    chunks = []
    for pos in keyword_positions:
        # Adjust start position to the beginning of a word
        start = max(0, pos - chunk_surround)
        while start > 0 and text[start-1] != ' ':
            start -= 1
        
        # Adjust end position to the end of a word
        end = min(len(text), pos + chunk_surround + len(keyword))
        while end < len(text) and text[end] != ' ':
            end += 1
        
        # Ensure complete words at the boundaries
        chunk = text[start:end]
        chunks.append(chunk)
    return chunks

def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    if st.button("Search"):
        if search_text:
            dense = get_embedding(search_text)
            sparse = bm25.encode_queries(search_text)
            hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)
            displayed_urls = set()
            
            try:
                query_result = index.query(
                    top_k=top_k,
                    vector=hdense,
                    sparse_vector=hsparse,
                    namespace='sparse',
                    include_metadata=True
                )
                
                if query_result and 'matches' in query_result:
                    results = []
                    for match in query_result['matches']:
                        url = match.get('metadata', {}).get('url', 'N/A')
                        if url in displayed_urls:
                            continue
                        displayed_urls.add(url)

                        score = match.get('score', 'N/A')
                        text = match.get('metadata', {}).get('text', '')
                        
                        chunks = find_keyword_variations_chunks(text, search_text, 50)
                        highlighted_chunks = [highlight_search_word(chunk, search_text) for chunk in chunks]
                        highlighted_text = ' ... '.join(highlighted_chunks)
                        
                        short_text = ' '.join(highlighted_text.split()[:20]) + "..."
                        results.append({"Score": score, "Text": highlighted_text, "URL": url})

                    st.table(results)
                else:
                    st.write("No results found.")
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Please enter search text.")

if __name__ == "__main__":
    main()
