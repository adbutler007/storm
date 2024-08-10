import logging
import os
from typing import Callable, Union, List
from tavily import TavilyClient

import dspy
import pandas as pd
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from utils import WebPageHelper
import time

class YouRM(dspy.Retrieve):
    def __init__(self, ydc_api_key=None, k=3, is_valid_source: Callable = None):
        super().__init__(k=k)
        if not ydc_api_key and not os.environ.get("YDC_API_KEY"):
            raise RuntimeError("You must supply ydc_api_key or set environment variable YDC_API_KEY")
        elif ydc_api_key:
            self.ydc_api_key = ydc_api_key
        else:
            self.ydc_api_key = os.environ["YDC_API_KEY"]
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'YouRM': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with You.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                headers = {"X-API-Key": self.ydc_api_key}
                results = requests.get(
                    f"https://api.ydc-index.io/search?query={query}",
                    headers=headers,
                ).json()

                authoritative_results = []
                for r in results['hits']:
                    if self.is_valid_source(r['url']) and r['url'] not in exclude_urls:
                        authoritative_results.append(r)
                if 'hits' in results:
                    collected_results.extend(authoritative_results[:self.k])
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint,
                    headers=headers,
                    params={**self.params, 'q': query}
                ).json()

                for d in results['webPages']['value']:
                    if self.is_valid_source(d['url']) and d['url'] not in exclude_urls:
                        url_to_results[d['url']] = {'url': d['url'], 'title': d['name'], 'description': d['snippet']}
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)

        return collected_results

class TavilyRM(dspy.Retrieve):
    def __init__(self, tavily_api_key=None, k=3, is_valid_source: Callable = None, search_depth="advanced"):
        super().__init__(k=k)
        if not tavily_api_key and not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError("You must supply tavily_api_key or set environment variable TAVILY_API_KEY")
        elif tavily_api_key:
            self.tavily_api_key = tavily_api_key
        else:
            self.tavily_api_key = os.environ["TAVILY_API_KEY"]
        
        self.client = TavilyClient(api_key=self.tavily_api_key)
        self.search_depth = search_depth
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {'TavilyRM': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Tavily for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []

        for query in queries:
            if not query.strip():
                logging.warning(f"Skipping empty query: '{query}'")
                continue

            try:
                response = self.client.search(
                    query=query,
                    search_depth=self.search_depth,
                    max_results=self.k,
                    include_answer=False,
                    include_images=False,
                    include_raw_content=False
                )

                if 'results' in response:
                    for r in response['results']:
                        if self.is_valid_source(r['url']) and r['url'] not in exclude_urls:
                            collected_results.append({
                                'description': r.get('content', ''),
                                'snippets': [r.get('content', '')],
                                'title': r.get('title', ''),
                                'url': r['url']
                            })
                else:
                    logging.error(f"No 'results' key in response for query: '{query}'")
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        return collected_results[:self.k]

class VectorRM(dspy.Retrieve):
    def __init__(self,
                 collection_name: str = "my_documents",
                 embedding_model: str = 'BAAI/bge-m3',
                 device: str = "mps",
                 k: int = 3,
                 chunk_size: int = 500,
                 chunk_overlap: int = 100,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        super().__init__(k=k)
        self.usage = 0
        
        # Set custom endpoint base URL internally
        self.base_url = "https://8013-01j2vebmta0r3ptbr6cm5e56a0.cloudspaces.litng.ai/"
        self.endpoints = {
            'text': f"{self.base_url}/retrieve_ensemble"
        }
        
        # Store other parameters
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {'CustomRM': usage}

    def query_endpoint(self, endpoint: str, query: str) -> List[dict]:
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    json={"text": query},
                    headers={"accept": "application/json", "Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for node in data.get("nodes", []):
                    metadata = node.get("metadata", {})
                    results.append({
                        'description': metadata.get('window', ''),
                        'snippets': [node.get('text', '')],
                        'title': metadata.get('title', ''),
                        'url': metadata.get('source_url', ''),
                        'type': 'text' if endpoint == self.endpoints['text'] else 'figure' if endpoint == self.endpoints['figures'] else 'table'
                    })
                return results
            except Exception as e:
                logging.error(f'Error occurs when querying {endpoint} (attempt {attempt + 1}/{self.max_retries}): {e}')
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f'Max retries reached for {endpoint}. Returning empty list.')
                    return []

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        self.usage += len(queries)
        collected_results = []

        for query in queries:
            for endpoint in self.endpoints.values():
                collected_results.extend(self.query_endpoint(endpoint, query))

        # Sort results by relevance (assuming the order returned by the API indicates relevance)
        # and limit to self.k results
        return collected_results[:self.k]

    # Placeholder methods to maintain interface similarity with VectorRM
    def init_online_vector_db(self, url: str, api_key: str):
        pass

    def init_offline_vector_db(self, vector_store_path: str):
        pass

    def update_vector_store(self, file_path: str, content_column: str, **kwargs):
        pass

    def get_vector_count(self):
        return 0
