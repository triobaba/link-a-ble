import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
import pandas as pd







openai_api_key = st.text_input("Enter openai api key")
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
pc = Pinecone(api_key="3661dc2a-3710-4669-a187-51faaa0cc557")
index = pc.Index("semantic-search-test")


client = OpenAI(api_key=openai_api_key)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding




def main():
    st.title("Pinecone Search Application")
    search_text = st.text_input("Enter search text:")
    select_options=["ada","3-small"]
    selected_namespace= st.selectbox("Select the Namespace you would like to query",options= select_options)

    # User inputs
    

    top_k = st.number_input("Enter top_k:", min_value=1, value=5)

    if st.button("Search"):
        if search_text and top_k:
            # Embed the search text
            embeddings = get_embedding(search_text)

            # Query Pinecone
            #index = pinecone.Index(index_namespace)
            query_result = index.query(vector=embeddings, top_k=top_k, namespace= selected_namespace, include_values=False, include_metadata=True)

            # Display results in a table
            if query_result and 'matches' in query_result:
                results = []
                for match in query_result['matches']:
                    score = match.get('score', 'N/A')
                    text = ' '.join(match.get('metadata', {}).get('text', '').split()[:10]) + "..."
                    url = match.get('metadata', {}).get('url', 'N/A')
        
                    results.append({"Score": score, "Text": text, "URL": url})

                st.table(results)
            else:
                st.write("No results found.")
        else:
            st.write("Please enter both search text and index namespace.")

if __name__ == "__main__":
    main()