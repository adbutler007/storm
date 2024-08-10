"""
This STORM Wiki pipeline powered by GPT-3.5/4 and local retrieval model that uses Qdrant.
You need to set up the following environment variables to run this script:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_API_TYPE: OpenAI API type (e.g., 'openai' or 'azure')
    - QDRANT_API_KEY: Qdrant API key (needed ONLY if online vector store was used)

You will also need an existing Qdrant vector store either saved in a folder locally offline or in a server online.
If not, then you would need a CSV file with documents, and the script is going to create the vector store for you.
The CSV should be in the following format:
content  | title  |  url  |  description
I am a document. | Document 1 | docu-n-112 | A self-explanatory document.
I am another document. | Document 2 | docu-l-13 | Another self-explanatory document.

Notice that the URL will be a unique identifier for the document so ensure different documents have different urls.

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""

import os
import sys
from argparse import ArgumentParser

sys.path.append('./src')
from storm_wiki.engine import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from rm import VectorRM
from lm import OpenAIModel, ClaudeModel
from utils import load_api_key


def main():
    load_api_key(toml_file_path='secrets.toml')
    engine_lm_configs = STORMWikiLMConfigs()
    claude_kwargs = {
        'api_key': os.getenv("ANTHROPIC_API_KEY"),
        'temperature': 1.0,
        'top_p': 0.9
    }
    openai_kwargs = {
        'api_key': os.getenv("OPENAI_API_KEY"),
        'api_provider': os.getenv('OPENAI_API_TYPE'),
        'temperature': 1.0,
        'top_p': 0.9,
        'api_base': os.getenv('AZURE_API_BASE')
    }
    # STORM is a LM system so different components can be powered by different models.
    # For a good balance between cost and quality, you can choose a cheaper/faster model for conv_simulator_lm 
    # which is used to split queries, synthesize answers in the conversation. We recommend using stronger models
    # for outline_gen_lm which is responsible for organizing the collected information, and article_gen_lm
    # which is responsible for generating sections with citations.
    conv_simulator_lm = OpenAIModel(model='gpt-4o-mini', max_tokens=1500, **openai_kwargs)
    question_asker_lm = OpenAIModel(model='gpt-4o-mini', max_tokens=1500, **openai_kwargs)
    outline_gen_lm = ClaudeModel(model='claude-3-5-sonnet-20240620', max_tokens=2000, **claude_kwargs)
    article_gen_lm = OpenAIModel(model='gpt-4o-mini', max_tokens=4000, **openai_kwargs)
    article_polish_lm = ClaudeModel(model='claude-3-5-sonnet-20240620', max_tokens=4000, **claude_kwargs)

    engine_lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    engine_lm_configs.set_question_asker_lm(question_asker_lm)
    engine_lm_configs.set_outline_gen_lm(outline_gen_lm)
    engine_lm_configs.set_article_gen_lm(article_gen_lm)
    engine_lm_configs.set_article_polish_lm(article_polish_lm)

    # Initialize the engine arguments
    engine_args = STORMWikiRunnerArguments(
        output_dir="results/claude_multi_topic",
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=15,
        max_thread_num=3,
    )

    # Setup VectorRM to retrieve information from your own data
    rm = VectorRM(collection_name="my_documents", device="mps", k=15)

    # initialize the vector store, either online (store the db on Qdrant server) or offline (store the db locally):
    #if args.vector_db_mode == 'offline':
    rm.init_offline_vector_db(vector_store_path="None")
    #elif args.vector_db_mode == 'online':
    rm.init_online_vector_db(url="none", api_key="none")

    # Update the vector store with the documents in the csv file
    #if args.update_vector_store:
    rm.update_vector_store(
            file_path="None",
            content_column='content',
            title_column='title',
            url_column='url',
            desc_column='description',
            batch_size=64
        )

    # Initialize the STORM Wiki Runner
    runner = STORMWikiRunner(engine_args, engine_lm_configs, rm)

    # run the pipeline
    topic = input('Topic: ')
    runner.run(
        topic=topic,
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=True,
    )
    runner.post_run()
    runner.summary()


if __name__ == "__main__":
    main()
