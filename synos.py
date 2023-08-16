from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
import pinecone
import csv
import pandas as pd
def docs(input_text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    api_key = "d9b7dbe2-d68d-4381-937d-06ddb5cd761c"
    env = "asia-southeast1-gcp-free"
    pinecone.init(
        api_key=api_key,
        environment=env
    )
    index_name = 'semantic-search'
    # now connect to the index
    index = pinecone.GRPCIndex(index_name)
    # create the query vector
    xq = model.encode(input_text).tolist()
    # now query
    xc = index.query(xq, top_k=5, include_metadata=True)
    return xc
def extract_data(query):
    output_data = docs(query)  # Assuming `docs()` function works correctly

    # Create a list of dictionaries containing text and score
    matches_info = [{'text': match['metadata']['text'], 'score': match['score']} for match in output_data['matches']]

    # Read the CSV file into a DataFrame
    df = pd.read_csv("file1.csv", index_col='id')

    # Create an empty DataFrame to store the filtered rows
    filtered_df = pd.DataFrame(columns=df.columns)

    # Iterate through each match and its info
    for match_info in matches_info:
        text = match_info['text']
        score = match_info['score']
        
        # Filter rows based on substring
        filtered_rows = df[df['synopsis'].str.contains(text)]
        filtered_rows['score'] = score
        
        # Add filtered rows to the new DataFrame
        filtered_df = pd.concat([filtered_df, filtered_rows])

    # Sort the DataFrame based on the 'score' column
    final_df = filtered_df.sort_values(by='score', ascending=False).drop_duplicates()
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_csv("file3.csv")
    # Return the final DataFrame
    final_json = final_df.to_json(orient='records')

    # Return the final JSON
    return final_json




   

# final = extract_data('war war war')

# print(final)