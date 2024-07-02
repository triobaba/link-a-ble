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
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def refine_results(search_intent, text):
    prompt = (
        "You are a very helpful assistant, helping link builders find the most relevant blogs based on their question. "
        "A returned result for a keyword search would be given to you. Refine this text and only return the results that fit the search intent. "
        "The search intent is a detailed description of what the user is looking for. Only return the results that closely fits the key aspects of the search intent the returned result should be a summary of the result. "
        "For example, if the search intent is about the impacts of artificial intelligence, only return results that closely discuss those impacts"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"content: {text}\n\nsearch intent: {search_intent}"}
        ],
        temperature=0)
    result= response.choices[0].message.content
    refined_results = []
    for line in result.split("\n"):
        if "does not fit the search intent" not in line and  "does not match" not in line  and "do not match " not in line:
            refined_results.append(line)
    
    return "\n".join(refined_results)



def is_relevant(summary, search_intent):
    # Check if the summary contains keywords from the search intent
    return any(keyword.lower() in summary.lower() for keyword in search_intent.split())

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
            try:
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
                        summary = refine_results(text, search_intent)
                        if is_relevant(summary, search_intent):
                            results.append({"Score": score, "AI Summary": summary, "URL": url})

                    if results:
                        st.table(pd.DataFrame(results))
                    else:
                        st.write("No relevant results found.")
                else:
                    st.write("No results found.")
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Please enter search text.")

if __name__ == "__main__":
    main()
