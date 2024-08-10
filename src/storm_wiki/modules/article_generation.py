import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy
from interface import ArticleGenerationModule
from storm_wiki.modules.callback import BaseCallbackHandler
from storm_wiki.modules.storm_dataclass import StormInformationTable, StormArticle, StormInformation
from utils import ArticleTextProcessing
import re
import requests
from urllib.parse import urlparse, urlunparse

def fix_truncated_image_urls(article_content):
    def verify_image_url(url, extensions=['.png', '.jpg']):
        parsed_url = urlparse(url)
        for ext in extensions:
            test_url = urlunparse(parsed_url._replace(path=parsed_url.path + ext))
            try:
                response = requests.head(test_url, timeout=5)
                if response.status_code == 200:
                    return test_url
            except requests.RequestException as e:
                logging.warning(f"Error verifying URL {test_url}: {str(e)}")
                return None

    def replace_truncated_url(match):
        url = match.group(1)
        if url.endswith('.'):  # This checks for the incomplete file stub ending with a dot
            verified_url = verify_image_url(url[:-1])  # Remove the trailing dot before verification
            if verified_url:
                return f"![]({verified_url})"
            else:
                # If verification fails, append .jpg as a fallback
                return f"![]({url}jpg)"
        return match.group(0)

    # Updated pattern to match incomplete image embeds
    pattern = r'!\[\]\((https?://[^\s)]+)\.'
    fixed_content = re.sub(pattern, replace_truncated_url, article_content)
    return fixed_content

class StormArticleGenerationModule(ArticleGenerationModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage, 
    """

    def __init__(self,
                 article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 retrieve_top_k: int = 5,
                 max_thread_num: int = 10):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)

    def generate_section(self, topic, section_name, information_table, section_outline, section_query):
        collected_info: List[StormInformation] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(queries=section_query,
                                                                    search_top_k=self.retrieve_top_k)
        output = self.section_gen(topic=topic,
                                  outline=section_outline,
                                  section=section_name,
                                  collected_info=collected_info)
        return {"section_name": section_name, "section_content": output.section, "collected_info": collected_info}

    def generate_article(self,
                         topic: str,
                         information_table: StormInformationTable,
                         article_with_outline: StormArticle,
                         callback_handler: BaseCallbackHandler = None) -> StormArticle:
        """
        Generate article for the topic based on the information table and article outline.

        Args:
            topic (str): The topic of the article.
            information_table (StormInformationTable): The information table containing the collected information.
            article_with_outline (StormArticle): The article with specified outline.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the article generation process. Defaults to None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        sections_to_write = article_with_outline.get_first_level_section_names()

        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(f'No outline for {topic}. Will directly search with the topic.')
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic]
            )
            section_output_dict_collection = [section_output_dict]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    # We don't want to write a separate introduction section.
                    if section_title.lower().strip() == 'introduction':
                        continue
                    # We don't want to write a separate conclusion section.
                    if section_title.lower().strip().startswith(
                            'conclusion') or section_title.lower().strip().startswith('summary'):
                        continue
                    section_query = article_with_outline.get_outline_as_list(root_section_name=section_title,
                                                                             add_hashtags=False)
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True)
                    section_outline = "\n".join(queries_with_hashtags)
                    future_to_sec_title[
                        executor.submit(self.generate_section,
                                        topic, section_title, information_table, section_outline, section_query)
                    ] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            fixed_section_content = fix_truncated_image_urls(section_output_dict["section_content"])
            article.update_section(parent_section_name=topic,
                                   current_section_content=section_output_dict["section_content"],
                                   current_section_info_list=section_output_dict["collected_info"])
        article.post_processing()
        return article

class ConvToSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(self, topic: str, outline: str, section: str, collected_info: List[StormInformation]):
        info = ''
        for idx, storm_info in enumerate(collected_info):
            info += f'[{idx + 1}]\n' + '\n'.join(storm_info.snippets)
            info += '\n\n'

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output)

        return dspy.Prediction(section=section)


class WriteSection(dspy.Signature):
    """You are tasked with writing a Wikipedia-style section based on collected information. Follow these instructions carefully to create an informative and well-structured section.

    First, review the collected information provided:

    Now, follow these steps to write the Wikipedia section:

    1. Begin with the main section title using a single "#" followed by the section title.

    2. Organize the information into logical subsections. Use "##" for subsection titles. Avoid using more than two levels of headings (i.e., don't use ### or ####). Instead, incorporate lower-level topics into paragraphs under their respective subsections.

    3. Write in a neutral, encyclopedic tone. Present facts and information objectively, avoiding personal opinions or biases.

    4. Use [1], [2], ..., [n] inline to indicate references. For example: "The capital of the United States is Washington, D.C.[1][3]." Do not include a separate References or Sources section at the end.

    5. If there are relevant figures in the collected information, include them using this format. Note this is just an example and you should use the actual figure caption, and full url with file name and extension, sources and disclaimers, and [n] reference id from the collected information.

    Example:

    Figure caption sourced verbatim from the collected information.

    ![](full image url verbatim from the collected information including full image file name and extension)

    Associated sources and disclaimers verbatim from the collected information, if any. Add [n] inline where n is the article source of the figure to reference the figure.

    Important: Always include the entire path to the image url and close all brackets so the image embed will render properly when parsed from markdown.

    6. If there are relevant tables in the collected information, include them using the provided markdown grid format. Note this is just an example and you should use the actual table caption, table content, sources and disclaimers, and [n] reference id from the collected information.

    Example:
    Table caption sourced verbatim from the collected information...

    +------------+--------+---------------------+
    | Header 1   | Header | Header 3            |
    +============+========+=====================+
    | Row 1, Col | Row 1, | Row 1, Col 3        |
    +------------+--------+---------------------+
    | Row 2, Col | Row 2, | Row 2, Col 3        |
    +------------+--------+---------------------+

    Associated sources and disclaimers verbatim from the collected information, if any... Add [n] inline where n is the article source of the table to reference the table.

    7. Ensure that the content flows logically from one subsection to another, maintaining coherence throughout the entire section.

    8. Aim for a comprehensive yet concise presentation of the information. Prioritize the most important and relevant facts related to the section title.

    9. If there are any contradictions or uncertainties in the collected information, present multiple viewpoints or indicate that the information is disputed, using appropriate references.

    10. Proofread your writing for clarity, grammar, and adherence to Wikipedia-style formatting.

    Please provide your completed Wikipedia section based on these instructions, using the collected information and adhering to the specified format. Begin your response with the main section title and continue with the content, including any relevant subsections, figures, and tables as needed. Do not include a separate reference section.
    """

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section precisely adhering to the instructions with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str
    )