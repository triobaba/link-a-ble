import os
import time
from pinecone import Pinecone,ServerlessSpec
import concurrent.futures
import pandas as pd
import numpy as np
import ast

# load the final embeddings
df=pd.read_csv("final_embeddings.csv")

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('pinecone_api_key')

# configure client
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec('aws', 'us-east-1')

# define index name
index_name='semantic-search-test'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embeddings-small
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
print(index.describe_index_stats())



batch_size = 100

# Iterate over the DataFrame in batches
for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch_json_data = []

    # Construct JSON objects for the current batch
    for ind, row in df.iloc[start:end].iterrows():
        json_obj = {
            "id": str(row['Unnamed: 0']),
            "values": list(map(float, row['embeddings'].strip('[]').split(','))),
            "metadata": {
                "text": row['text_chunk'],
                "url": row['url']
            }
        }
        batch_json_data.append(json_obj) 

    # Upsert the current batch
    index.upsert(batch_json_data,namespace='3-small')


print('done')

print(index.describe_index_stats())