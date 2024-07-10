import streamlit as st
from pinecone import Pinecone
import re
import nltk
import pandas as pd
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from pinecone_text.sparse import BM25Encoder

# Read the data
df = pd.read_csv('src/preprocessed_data_cleaned.csv')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
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

def get_keyword_variations(keyword):
    variations = set()
    words = keyword.split()  # Split phrase into individual words
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                variations.add(lemma.name().replace('_', ' '))
        variations.add(word)  # Include the original word

    return list(variations)


def find_keyword_snippets(text, keyword, snippet_length=5):
    snippets = []
    words = text.split()

    # First, check for the whole phrase
    phrase_pattern = r'\b' + re.escape(keyword) + r'\b'
    match = re.search(phrase_pattern, text, re.IGNORECASE)
    
    if match:
        # If the whole phrase is found, highlight it as a unit
        start_idx = match.start()
        end_idx = match.end()
        start_word_idx = len(text[:start_idx].split())
        end_word_idx = len(text[:end_idx].split())
        snippet_start_idx = max(start_word_idx - snippet_length, 0)
        snippet_end_idx = min(end_word_idx + snippet_length, len(words))

        snippet_words = words[snippet_start_idx:snippet_end_idx]
        snippet = ' '.join(snippet_words)

        # Highlight the whole phrase
        highlighted_snippet = re.sub(phrase_pattern, f"**{match.group(0)}**", snippet, flags=re.IGNORECASE)
        
        return highlighted_snippet
    else:
        # If the whole phrase is not found, search for individual words and their variations
        keyword_variations = get_keyword_variations(keyword)
        search_patterns = [r'\b' + re.escape(word) + r'\b' for word in set(keyword_variations)]
        found_snippets = set()
        
        for pattern in search_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_idx = match.start()
                end_idx = match.end()
                start_word_idx = len(text[:start_idx].split())
                end_word_idx = len(text[:end_idx].split())
                snippet_start_idx = max(start_word_idx - snippet_length,0)
                snippet_end_idx = min(end_word_idx + snippet_length, len(words))

                snippet_words = words[snippet_start_idx:snippet_end_idx]
                snippet = ' '.join(snippet_words)

                # Highlight the keyword
                highlighted_snippet = re.sub(pattern, f"**{match.group(0)}**", snippet, flags=re.IGNORECASE)
                
                # Only add unique snippets
                if highlighted_snippet not in found_snippets:
                    snippets.append(highlighted_snippet)
                    found_snippets.add(highlighted_snippet)
        
        if snippets:
            return '\n'.join(snippets[:2])
        else:
            return " "

def process_results(results, keyword, snippet_length=10):
    processed_results = []
    
    for result in results:
        score=result['Score']
        url = result['URL']
        text = result['Text']
        # if url.endswith('/'):
        #     url = url.rstrip('/')
        
        # parts = url.rsplit('/', 1)  # Split on the last slash
        # title = parts[-1] if len(parts) > 1 else url
        # if not title:
        #     title = "Untitled"
        
        snippets = find_keyword_snippets(text, keyword, snippet_length)
        processed_result = {"Score": score,"URL": result["URL"], "Snippets": snippets}
        processed_results.append(processed_result)
    
    return processed_results

def remove_empty_snippets(df):
    df_cleaned = df[df['Snippets'].str.strip() != '']
    return df_cleaned

def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    #top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    openai_api_key = st.text_input("Enter OpenAI API Key:")
    global client
    client = OpenAI(api_key=openai_api_key)

    if st.button("Search"):
        if search_text:
            try:
                dense = get_embedding(search_text)
                sparse = bm25.encode_queries(search_text)
                hdense, hsparse = hybrid_scale(dense, sparse, alpha=0.05)

                query_result = index.query(
                    top_k=20,
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
                        results.append({"Score": score, "Text": text, "URL": url})
                    processed_results_1 = process_results(results, keyword=search_text)
                    processed_results = pd.DataFrame(processed_results_1)
        
                    cleaned_results=remove_empty_snippets(processed_results)

                    if results:
                        st.table(pd.DataFrame(cleaned_results))
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

