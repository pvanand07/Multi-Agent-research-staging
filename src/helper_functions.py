import psycopg2
import streamlit as st
from datetime import datetime


def insert_data(
    user_id: str, user_query: str, subtopic_query: str, response: str, html_report: str
) -> None:
    """
    Push the user interaction to the database
    """
    conn = psycopg2.connect(
        dbname="postgres",
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        host="aws-0-us-west-1.pooler.supabase.com",
        port="5432",
    )
    cur = conn.cursor()
    insert_query = """
    INSERT INTO research_pro_chat_v2 (user_id, user_query, subtopic_query, response, html_report, created_at)
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    cur.execute(
        query=insert_query,
        vars=(
            user_id,
            user_query,
            subtopic_query,
            response,
            html_report,
            datetime.now(),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


import mistune
from mistune.plugins.table import table
from jinja2 import Template
import ast
from fpdf import FPDF
import re
import pandas as pd
import nltk

nltk.download("stopwords")
nltk.download("punkt")
import requests
from retry import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from brave import Brave
from fuzzy_json import loads
from half_json.core import JSONFixer
from openai import OpenAI
import os
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

load_dotenv("keys.env")

# Retrieve environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HELICON_API_KEY = os.getenv("HELICON_API_KEY")

llm_default_small = "meta-llama/Llama-3-8b-chat-hf"
llm_default_medium = "meta-llama/Llama-3-70b-chat-hf"

SysPromptData = "You are an information retriever and summarizer, return only the factual information regarding the user query"
SysPromptDefault = (
    "You are an expert AI, complete the given task. Do not add any additional comments."
)

import tiktoken  # Used to limit tokens

# Instead of Llama3 using available option/ replace if found anything better
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def limit_tokens(input_string: str, token_limit: int = 8000) -> str:
    """
    Limit number of tokens sent to the model to respect context length of models.
    """
    return encoding.decode(encoding.encode(input_string)[:token_limit])


def together_response(
    message: str,
    model: str = "meta-llama/Llama-3-8b-chat-hf",
    SysPrompt: str = SysPromptDefault,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    frequency_penalty: float = 0.01,
) -> str:
    """
    Make LLM inference for the giving SysPrompt and message.
    """

    client = OpenAI(base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY)

    messages = [
        {"role": "system", "content": SysPrompt},
        {"role": "user", "content": message},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )
    return response.choices[0].message.content


def json_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts and fix JSON from text using regex and fuzzy JSON loading.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_out = match.group(0)
    else:
        json_out = text
    try:
        # Using fuzzy json loader
        return loads(json_out)
    except Exception:
        # Using JSON fixer/ Fixes even half json/ Remove if you need an exception
        fix_json = JSONFixer()
        return loads(fix_json.fix(json_out).line)


def remove_stopwords(text: str) -> str:
    """
    Remove all the stopwords ('the','a',...,'in') from the given text.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_text)


def rephrase_content(data_format: str, content: str, query: str) -> str | None:
    """
    Repharsing and cleaning the scrapped content in the requested data format with LLM (Llama3-8b).
    """
    if data_format == "Default":
        return together_response(
            f"return only the factual information regarding the query: {{{query}}} using the scraped context:{{{limit_tokens(content)}}}",
            SysPrompt=SysPromptData,
            max_tokens=1000,
        )
    elif data_format == "Structured data":
        return together_response(
            f"return only the structured data regarding the query: {{{query}}} using the scraped context:{{{limit_tokens(content)}}}",
            SysPrompt=SysPromptData,
            max_tokens=1000,
        )
    elif data_format == "Quantitative data":
        return together_response(
            f"return only the numerical data regarding the query: {{{query}}} using the scraped context:{{{limit_tokens(content,token_limit=1000)}}}",
            SysPrompt=SysPromptData,
            max_tokens=1000,
        )
    else:
        return together_response(
            f"return only the factual information regarding the query: {{{query}}}. Output should be concise chunks of \
        paragraphs or tables or both, ignore links, using the scraped context:{{{limit_tokens(content, token_limit = 1000)}}}",
            SysPrompt=SysPromptData,
            max_tokens=1000,
        )


class Scraper:
    def __init__(
        self, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    @retry(tries=3, delay=1)
    def fetch_content(self, url):
        try:
            response = self.session.get(url, timeout=2)
            if response.status_code == 200:
                return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page content for {url}: {e}")
        return None


def extract_main_content(html: str) -> str:
    if html:
        plain_text = ""
        soup = BeautifulSoup(html, "lxml")
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "span", "table"]
        ):
            plain_text += element.get_text(separator=" ", strip=True) + "\n"
        return plain_text
    return ""


def process_content(data_format: str, url: str, query: str) -> Tuple[str | None, str]:
    """
    Scrape data from the fetched URLs and Repharsing it according to requested user query and data format.
    """
    scraper = Scraper()
    html_content = scraper.fetch_content(url)
    if html_content:
        content = extract_main_content(html_content)
        if content:
            rephrased_content = rephrase_content(
                data_format=data_format,
                content=limit_tokens(remove_stopwords(content), token_limit=1000),
                query=query,
            )
            with st.expander(f"Fetched relevant data from url: {url}"):
                st.markdown(rephrased_content, unsafe_allow_html=True, help=None)
            return rephrased_content, url
    return "", url


def fetch_and_extract_content(
    data_format: str, query: str, urls: List[str], num_refrences: int = 4
) -> List[Tuple[str | None, str]]:
    """
    Asynchronously makeing request to urls and doing further process
    """
    all_text_with_urls = []
    start_url = 0
    while (len(all_text_with_urls) != num_refrences) and (start_url < len(urls)):
        end_url = start_url + (num_refrences - len(all_text_with_urls))
        urls_subset = urls[start_url:end_url]
        with ThreadPoolExecutor(max_workers=len(urls_subset)) as executor:
            future_to_url = {
                executor.submit(process_content, data_format, url, query): url
                for url in urls_subset
            }
            all_text_with_urls += [
                future.result()
                for future in as_completed(future_to_url)
                if future.result()[0] != ""
            ]
        start_url = end_url

    return all_text_with_urls


def search_brave(query: str, num_results: int = 5) -> List[str]:
    """
    Internet search engine to fetch links related to query.
    """

    brave = Brave(BRAVE_API_KEY)

    search_results = brave.search(q=query, count=num_results)

    return [url.__str__() for url in search_results.urls]


def md_to_html(md_text: str) -> str:
    "Function to Convert Markdown response to HTML (tables, paragraphs, headings)"
    renderer = mistune.HTMLRenderer()
    markdown_renderer = mistune.Markdown(renderer, plugins=[table])
    html_content = markdown_renderer(md_text)
    return html_content


def generate_report_with_reference(
    full_data: List[Dict[str, Any]]
) -> Tuple[str, List[Any]]:
    """
    Generate HTML report with references and saves pdf report to "generated_pdf_report.pdf"
    """
    pdf = FPDF()
    with open("templates/report_with_reference.html") as f:
        html_template = f.read()

    # Loop through each row in your dataset
    html_report = ""
    idx = 1
    df_tables_list = []
    for subtopic_data in full_data:

        md_report = md_to_html(subtopic_data["md_report"])
        df_tables_list.append(extract_tables_from_html(md_report))

        # Convert the string representation of a list of tuples back to a list of tuples
        references = ast.literal_eval(subtopic_data["text_with_urls"])

        collapsible_blocks = []
        for ref_idx, reference in enumerate(references):
            ref_text = md_to_html(reference[0])
            ref_url = reference[1]
            urls_html = "".join(f'<a href="{ref_url}"> {ref_url}</a>')

            collapsible_block = """
            <details>
                <summary>Reference {}: {}</summary>
                <div>
                    <p>{}</p>
                    <ul>{}</ul>
                </div>
            </details>
            """.format(
                ref_idx + 1, urls_html, ref_text, urls_html
            )

            collapsible_blocks.append(collapsible_block)

        references_html = "\n".join(collapsible_blocks)

        template = Template(html_template)
        html_page = template.render(md_report=md_report, references=references_html)

        pdf.add_page()
        pdf_report = (
            f"<h1><strong>Report {idx}</strong></h1>"
            + md_report
            + f"<h1><strong>References for Report {idx}</strong></h1>"
            + references_html
        )

        pdf.write_html(
            pdf_report.encode("ascii", "ignore").decode("ascii")
        )  # Filter non-asci characters
        html_report += html_page
        idx += 1

    pdf.output("generated_pdf_report.pdf")
    return html_report, df_tables_list


def write_dataframes_to_excel(
    dataframes_list: List[pd.DataFrame], filename: str
) -> None:
    """
    Save pandas dataframes as excels
    """
    try:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for idx, dataframes in enumerate(dataframes_list):
                startrow = 0
                for df in dataframes:
                    df.to_excel(
                        writer,
                        sheet_name=f"Sheet{idx+1}",
                        startrow=startrow,
                        index=False,
                    )
                    startrow += len(df) + 2
    except:
        # Empty dataframe due to no tables found, file is not written
        pass


def extract_tables_from_html(html_file: str) -> List[pd.DataFrame]:
    """
    Extract tables/ data-in-table-tag from HTML. and return as pandas dataframe.
    """
    # Initialize an empty list to store the dataframes
    dataframes = []

    # Open the HTML file and parse it with BeautifulSoup
    soup = BeautifulSoup(html_file, "html.parser")

    # Find all the tables in the HTML file
    tables = soup.find_all("table")

    # Iterate through each table
    for table in tables:
        # Extract the table headers
        headers = [th.text for th in table.find_all("th")]

        # Extract the table data
        rows = table.find_all("tr")
        data = []
        for row in rows:
            row_data = [td.text for td in row.find_all("td")]
            data.append(row_data)

        # Create a dataframe from the headers and data
        df = pd.DataFrame(data, columns=headers)

        # Append the dataframe to the list of dataframes
        dataframes.append(df)

    # Return the list of dataframes
    return dataframes
