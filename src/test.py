import streamlit as st
import pinecone
import re
import pandas as pd
from embedding_pipeline import get_embedding
from upserting_pipeline import index
from pinecone_text.sparse import BM25Encoder

# Read the data
df = pd.read_csv('data1.csv')

# Initialize BM25Encoder
bm25 = BM25Encoder()
bm25.fit(df['text_chunk'])

# Define the Streamlit application



def hybrid_scale(dense, sparse, alpha: float):
    """Hybrid vector scaling using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: float between 0 and 1 where 0 == sparse only
               and 1 == dense only
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


def highlight_search_word(text,search_text):
    highlighted_text = re.sub(f"({search_text})",r"<mark>\1<mark",text, flags=re.IGNORECASE)
    return highlighted_text

def ensure_keyword_in_text(text, search_text):
    if search_text.lower() in text.lower():
        highlighted_text=highlight_search_word(text,search_text)
    else:
        words=text.split()
        extended_text=' '.join(words[:10])
        highlighted_text=highlight_search_word(extended_text,search_text)
    return highlighted_text
def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    if st.button("Search"):
        if search_text:
            # Embed the search text
            dense = get_embedding(search_text)
            sparse = bm25.encode_queries(search_text)

            # Debug: Print embeddings
            hdense, hsparse = hybrid_scale(dense, sparse, alpha=0.5)
            displayed_urls = set()
            
            # Query Pinecone
            try:
                query_result = index.query(
                    top_k=top_k,
                    vector=hdense,
                    sparse_vector=hsparse,
                    namespace='sparse',
                    include_metadata=True
                )
                
                # Debug: Print query results
                
                # Display results in a table
                if query_result and 'matches' in query_result:
                    results = []
                    for match in query_result['matches']:
                        url = match.get('metadata', {}).get('url', 'N/A')
                        if url in displayed_urls:
                            continue  # Skip URLs that have already been displayed
                        displayed_urls.add(url)

                        score = match.get('score', 'N/A')
                        text = match.get('metadata', {}).get('text', '')
                        highlighted_text = ensure_keyword_in_text(text, search_text)
                        short_text = ' '.join(highlighted_text.split()[:20]) + "..."
                        results.append({"Score": score, "Text": short_text, "URL": url})

                    st.table(results)
                else:
                    st.write("No results found.")
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Please enter search text.")

if __name__ == "__main__":
    main()