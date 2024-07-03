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
        "The search intent is a detailed description of what the user is looking for. Only return the results that closely fit the key aspects of the search intent. The returned result should be a summary of the result."
        "For example, if the search intent is about the impacts of artificial intelligence, only return results that closely discuss those impacts."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"content: {text}\n\nsearch intent: {search_intent}"}
        ],
        temperature=0)
    
    result = response.choices[0].message.content
    negative_phrases = [
        "does not fit the search intent", 
        "does not match", 
        "do not match", 
        "not relevant", 
        "irrelevant", 
        "no match",
        "may not be the best fit "
    ]

    refined_results = []
    for line in result.split("\n"):
        if not any(phrase in line.lower() for phrase in negative_phrases):
            refined_results.append(line)
    
    return "\n".join(refined_results)

def is_relevant(summary, search_intent):
    # Check if the summary contains keywords from the search intent
    return any(keyword.lower() in summary.lower() for keyword in search_intent.split())


def score_relevance(summary, search_intent):
    prompt = (
        f"Given the summary: '{summary}' and the search intent: '{search_intent}', "
        "return a relevance score from 1 to 10, where 10 means the summary perfectly fits the search intent and 1 means it does not fit at all. the score should be calculated of how these summary  fits the search intent relative to one another.Each scores must be distinctive"
        "Please you must return only the numerical score."
    )
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0)
    
    response_message = response.choices[0].text.strip()
    #st.write(f"API Response: {response_message}")  # Debugging: Display the API response

    try:
        # Use regular expression to extract the score from the response
        score_match = re.search(r'\b\d+\b', response_message)
        if score_match:
            score = int(score_match.group(0))
        else:
            score = 0  # Default score in case of parsing error
    except (ValueError, IndexError) as e:
        st.write(f"Error parsing the relevance score: {e}")  # Debugging: Display error message
        score = 0  # Default score in case of parsing error
    
    #st.write(f"Extracted Score: {score}")  # Debugging: Display the extracted score
    return score


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
                        pinecone_score = match.get('score', 'N/A')
                        text = match.get('metadata', {}).get('text', '')
                        summary = refine_results(text, search_intent)
                        if is_relevant(summary, search_intent):
                            relevance_score = score_relevance(summary, search_intent)
                            results.append({"Pinecone Score": pinecone_score, "Relevance Score": relevance_score, "AI Summary": summary,'Text': text[:70], "URL": url})

                    if results:
                        results = sorted(results, key=lambda x: (-x["Relevance Score"]))
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
