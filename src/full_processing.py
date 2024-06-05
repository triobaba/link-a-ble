import pandas as pd
import os
import re

def clean_text(text):
    """
    Function to clean the text by removing special characters, digits, converting to lowercase,
    and removing extra whitespace.
    """
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_text(text, chunk_size=500):
    """
    Function to preprocess the text by cleaning it and breaking it into chunks of specified size.
    """
    # Clean the text
    text = clean_text(text)
    # Split the text into words
    words = text.split()
    # Break down the text into chunks of chunk_size words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def process_csv(input_file_path, chunk_size=500):
    """
    Function to process the CSV file, apply text preprocessing, and save the processed data to a new CSV file.
    """
    # Load the CSV file
    data = pd.read_csv(input_file_path)

    # Apply the preprocessing function to the text_chunk column
    data['cleaned_text'] = data['text'].apply(lambda x: preprocess_text(x, chunk_size))

    # Create a new DataFrame with each chunk of text and its respective URL and title
    processed_data = []
    for index, row in data.iterrows():
        for chunk in row['cleaned_text']:
            processed_data.append([chunk, row['url']])
    
    # Convert the list to a DataFrame
    processed_df = pd.DataFrame(processed_data, columns=['text_chunk', 'url'])

    # Determine the output file path
    output_file_path = os.path.join(os.path.dirname(input_file_path), 'preprocessed_data_cleaned.csv')

    # Save the processed DataFrame to the output CSV file
    processed_df.to_csv(output_file_path, index=False)

    print(f"Processed data saved to {output_file_path}")

# Example usage
input_file_path = 'dataset.csv'  # Replace with your input file path
process_csv(input_file_path, chunk_size=500)
