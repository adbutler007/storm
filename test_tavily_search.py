import os
from tavily import TavilyClient
import json

def test_tavily_search():
    # Load the API key from secrets.toml
    api_key = None
    with open('secrets.toml', 'r') as f:
        for line in f:
            if line.startswith('TAVILY_API_KEY'):
                api_key = line.split('=')[1].strip().strip('"')
                break
    
    if not api_key:
        print("API key not found in secrets.toml")
        return

    # Initialize Tavily client
    client = TavilyClient(api_key=api_key)

    # Test queries
    queries = [
        "History of gold as an investment",
        "Economic theory behind gold as a hedge against inflation",
        "Current trends in gold investment",
        ""  # Empty query to test error handling
    ]

    for query in queries:
        print(f"\nTesting query: '{query}'")
        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_answer=False,
                include_images=False,
                include_raw_content=False
            )
            
            # Print the full response for debugging
            print("Full response:")
            print(json.dumps(response, indent=2))

            # Check if 'results' is in the response
            if 'results' in response:
                print(f"Number of results: {len(response['results'])}")
                for i, result in enumerate(response['results'], 1):
                    print(f"\nResult {i}:")
                    print(f"Title: {result.get('title', 'N/A')}")
                    print(f"URL: {result.get('url', 'N/A')}")
                    print(f"Content: {result.get('content', 'N/A')[:100]}...")  # First 100 chars of content
            else:
                print("No 'results' found in the response.")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_tavily_search()