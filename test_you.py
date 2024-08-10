import os
import requests
import json

def test_you_search():
    # Load the API key from secrets.toml
    with open('secrets.toml', 'r') as f:
        for line in f:
            if line.startswith('YDC_API_KEY'):
                api_key = line.split('=')[1].strip().strip('"')
                break
    
    if not api_key:
        print("API key not found in secrets.toml")
        return

    # Set up the request
    headers = {"X-API-Key": api_key}
    query = "Gold as an investment asset"
    url = f"https://api.ydc-index.io/search?query={query}"

    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        results = response.json()

        # Print the results
        print(json.dumps(results, indent=2))

        # Check if 'hits' is in the response
        if 'hits' in results:
            print(f"\nNumber of hits: {len(results['hits'])}")
            for i, hit in enumerate(results['hits'][:3], 1):  # Print details of first 3 hits
                print(f"\nHit {i}:")
                print(f"Title: {hit.get('title', 'N/A')}")
                print(f"URL: {hit.get('url', 'N/A')}")
                print(f"Description: {hit.get('description', 'N/A')[:100]}...")  # First 100 chars of description
        else:
            print("\nNo 'hits' found in the response.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_you_search()