import os
from openai import OpenAI
import pandas as pd

openai_api_key = os.getenv('OPENAI_API_KEY')
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # the maximum for text-embedding-3-small is 8191

client = OpenAI(api_key=openai_api_key)


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def converts_string_to_list(string):
    'converts the string representation of the embeddings list to a list of floats'
    if not isinstance(string, str):
        raise ValueError(f"Expected a string, but got {type(string)}")
    if not (string.startswith('[') and string.endswith(']')):
        raise ValueError(f"Expected a string in list format, but got {string}")
    return [float(x) for x in string[1:-1].split(',')]
   


def main():
    #load data 
    df= pd.read_csv('preprocessed_data_cleaned.csv')

    #get embeedings

    df['embeddings'] = df['text_chunk'].apply(get_embedding)

    #save the embeddings to a new csv file

    df.to_csv('text_with_embeddings.csv')

    #load  the dataframw with embeddings

    df = pd.read_csv('text_with_embeddings.csv')


    #convert the string representation of the lists to a list of floats

    df['embeddings']= df['embeddings'].apply(converts_string_to_list)


    #save the final embeddings to a new csv file

    df.to_csv('final_embeddings_ada.csv')





if __name__ == "__main__":
    main()
