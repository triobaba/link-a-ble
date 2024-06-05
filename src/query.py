import markdown
from embedding_pipeline import get_embedding,converts_string_to_list
from upserting_pipeline import index
text='artificial inteligence'


res=get_embedding(text)


query_results1 = index.query(
    namespace= 'ada',
    vector=res ,
    top_k=3,
    include_values=False,
    include_metadata=True,
    
)
#output parser

def parse_query_results(query_result1):
    matches = query_result1.get('matches', [])
    
    for idx, match in enumerate(matches, start=1):
        score = match.get('score', 'N/A')
        text = match.get('metadata', {}).get('text', '')
        url = match.get('metadata', {}).get('url', 'N/A')
        
        # Extract a very short sentence from the text
        short_sentence = ' '.join(text.split()[:10])  # Adjust the number of words as needed
        
        # Print the formatted result
        print(f"{idx}. {score} {short_sentence}... {url}")

# Call the function to display results
parse_query_results(query_results1)

