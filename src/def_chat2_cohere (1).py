import streamlit as st
from pinecone import Pinecone
import re
import pandas as pd
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder

# Read the data
df = pd.read_csv('preprocessed_data_cleaned.csv')

# Initialize BM25Encoder
bm25 = BM25Encoder()
bm25.fit(df['text_chunk'])

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
index = pc.Index("hybridsearch")

import cohere
co = cohere.Client(st.secrets["cohere_api_key"])

# Define OpenAI client
client = None  # Initialize to None until user provides API key

# Define the Streamlit application
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def find_keyword_snippets(text, keyword, snippet_length=100):
    import re
    snippets = []
    
    # Create a list of search patterns: the entire keyword and individual words
    search_patterns = [keyword] + keyword.split()
    
    # Find all occurrences of the search patterns in the text
    for pattern in search_patterns:
        for match in re.finditer(re.escape(pattern), text, re.IGNORECASE):
            start = max(match.start() - snippet_length, 0)
            end = min(match.end() + snippet_length, len(text))
            snippet = text[start:end]
            
            # Highlight the keyword
            highlighted_snippet = re.sub(re.escape(pattern), f"**{pattern}**", snippet, flags=re.IGNORECASE)
            
            # Add ellipses if the snippet is not at the start or end of the text
            if start > 0:
                highlighted_snippet = "..." + highlighted_snippet
            if end < len(text):
                highlighted_snippet = highlighted_snippet + "..."
                
            snippets.append(highlighted_snippet)
    
    return " ".join(snippets)

def process_results(results, keyword, snippet_length=50):
    processed_results = []
    
    for result in results:
        text = result["Text"]
        snippets = find_keyword_snippets(text, keyword, snippet_length)
        processed_result = {"URL": result["URL"], "Snippets": snippets}
        processed_results.append(processed_result)
    
    return processed_results

def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    search_intent = st.text_input("Enter your search intent")
    #top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    global client
    client = OpenAI(api_key=openai_api_key)

    if st.button("Search"):
        if search_text:          
            dense = get_embedding(search_text)
            sparse = bm25.encode_queries(search_text)
            hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)

            query_result = index.query(
                top_k=6,
                vector=hdense,
                sparse_vector=hsparse,
                namespace='sparse',
                include_metadata=True
            )
            results_1 = []
            displayed_urls = set()
            for match in query_result['matches']:
                url = match.get('metadata', {}).get('url', 'N/A')
                if url in displayed_urls:
                    continue
                displayed_urls.add(url)
                score = match.get('score', 'N/A')
                text = match.get('metadata', {}).get('text', '')
                results_1.append({"Text": text, "URL": url})
            processed_results_1 = process_results(results_1, keyword=search_text)
            docs = [x["metadata"]['text'] for x in query_result['matches']]
            rerank_docs = co.rerank(query=search_intent, documents=docs, top_n=10, model="rerank-english-v2.0")
            docs_reranked = [query_result['matches'][result.index] for result in rerank_docs.results]
            results = []
            displayed_urls = set()
            for match in docs_reranked:
                url = match.get('metadata', {}).get('url', 'N/A')
                if url in displayed_urls:
                    continue
                displayed_urls.add(url)
                score = match.get('score', 'N/A')
                text = match.get('metadata', {}).get('text', '')
                results.append({"Text": text, "URL": url})
            processed_results = process_results(results, keyword=search_text)
            # if query_result and 'matches' in query_result:
            #     results = []
            #     displayed_urls = set()
            #     for match in query_result['matches']:
            #         url = match.get('metadata', {}).get('url', 'N/A')
            #         if url in displayed_urls:
            #             continue
            #         displayed_urls.add(url)
            #         score = match.get('score', 'N/A')
            #         text = match.get('metadata', {}).get('text', '')
            #         summary = refine_results(text, search_intent)
            #         if is_relevant(summary, search_intent):
            #             results.append({"Score": score, "AI Summary": summary, "URL": url})
            st.text('Unranked Results')
            st.table(pd.DataFrame(processed_results_1))
            st.text('Reranked Results')
            st.table(pd.DataFrame(processed_results))
if __name__ == "__main__":
    main()
