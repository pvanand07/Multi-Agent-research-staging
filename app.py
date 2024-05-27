import streamlit as st
from streamlit_tree_select import tree_select
from streamlit_toggle import st_toggle_switch

st.set_page_config(layout="wide")

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("keys.env")

# Retrieve environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HELICON_API_KEY = os.getenv("HELICON_API_KEY")


####------------------------------ OPTIONAL--> User id and persistant data storage-------------------------------------####
import uuid
from datetime import datetime

# Create new user_id if not in current streamlit memmory
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

####------------------------------ OPTIONAL--> User id and persistant data storage-------------------------------------####
####------------------app.py----------------------##
import re
import nltk

nltk.download("stopwords")
nltk.download("punkt")

import tiktoken  # Used to limit tokens

encoding = tiktoken.encoding_for_model(
    "gpt-3.5-turbo"
)  # Instead of Llama3 using available option/ replace if found anything better


def limit_tokens(input_string, token_limit=8000):
    """
    Limit tokens sent to the model
    """
    return encoding.decode(encoding.encode(input_string)[:token_limit])


from src.helper_functions import (
    together_response,
    write_dataframes_to_excel,
    generate_report_with_reference,
    json_from_text,
    fetch_and_extract_content,
    search_brave,
    insert_data,
)

####--------------------CONSTANTS------------------------------##

SysPromptJson = "You are now in the role of an expert AI who can extract structured information from user request. Both key and value pairs must be in double quotes. You must respond ONLY with a valid JSON file. Do not add any additional comments."
SysPromptList = "You are now in the role of an expert AI who can extract structured information from user request. All elements must be in double quotes. You must respond ONLY with a valid python List. Do not add any additional comments."
SysPromptDefault = (
    "You are an expert AI, complete the given task. Do not add any additional comments."
)

llm_default_small = "meta-llama/Llama-3-8b-chat-hf"
llm_default_medium = "meta-llama/Llama-3-70b-chat-hf"

sys_prompts = {
    "SysPromptOffline": {
        "Default": "You are an expert AI, complete the given task. Do not add any additional comments.",
        "Full Text Report": "You are an expert AI who can create a detailed report from user request. The report should be in markdown format. Do not add any additional comments.",
        "Tabular Report": "You are an expert AI who can create a structured report from user request.The report should be in markdown format structured into subtopics/tables/lists. Do not add any additional comments.",
        "Tables only": "You are an expert AI who can create a structured tabular report from user request.The report should be in markdown format consists of only markdown tables. Do not add any additional comments.",
    },
    "SysPromptOnline": {
        "Default": "You are an expert AI, complete the given task using the provided context. Do not add any additional comments.",
        "Full Text Report": "You are an expert AI who can create a detailed report using information provided in the context from user request. The report should be in markdown format. Do not add any additional comments.",
        "Tabular Report": "You are an expert AI who can create a structured report using information provided in the context from user request. The report should be in markdown format structured into subtopics/tables/lists. Do not add any additional comments.",
        "Tables only": "You are an expert AI who can create a structured tabular report using information provided in the context from user request. The report should be in markdown format consists of only markdown tables. Do not add any additional comments.",
    },
}


def generate_topics(user_input):

    prompt_topics = f"""CREATE A LIST OF 5-10 CONCISE SUBTOPICS TO FOLLOW FOR COMPLETING ###{user_input}###, RETURN A VALID PYTHON LIST"""
    prompt_keywords = f"""EXTRACT KEYWORDS(NOUN) FROM USER INPUT ###{user_input}###, RETURN A VALID PYTHON LIST"""

    response_topics = together_response(
        prompt_topics, model=llm_default_small, SysPrompt=SysPromptList
    )
    topics = json_from_text(response_topics)

    user_query_keywords = json_from_text(
        together_response(
            prompt_keywords, model=llm_default_small, SysPrompt=SysPromptList
        )
    )
    st.session_state.user_query_keywords = user_query_keywords

    st.markdown(user_query_keywords)
    st.markdown(topics)

    prompt_subtopics = (
        f"""List 2-5  subtopics for each topic in the list of '{topics}' covering all aspects to generate a report, within the context of ###{user_query_keywords}###. to achieve each subtopic add a detailed instruction"""
        + """ Respond in the following format: 
        {
            "Topic 1": [
                ["Subtopic","Instuction"]
            ],
            "Topic 2": [
                ["Subtopic","Instuction"]
            ]
        }, RETURN A VALID JSON FILE"""
    )

    response_subtopics = together_response(
        prompt_subtopics,
        model=llm_default_medium,
        SysPrompt=SysPromptJson,
        frequency_penalty=0.65,
    )
    subtopics = json_from_text(response_subtopics)

    return subtopics


def generate_missing_topics(user_input):

    topics = user_input
    user_query_keywords = st.session_state.user_query_keywords

    prompt_subtopics = (
        f"""List 2-5  Subtopics for the given Topic : ###{topics}### covering all aspects to generate a report, within the context of these keywords : ###{user_query_keywords}###, also to achieve each subtopic add a detailed Instruction that will be used as a LLM promt, hence keep it contextual."""
        + """ Respond in the following format: 
        {
            "Topic 1": [
                ["Subtopic","Instuction"]
            ],
        }, RETURN A VALID JSON FILE"""
    )

    response_subtopics = together_response(
        prompt_subtopics, model=llm_default_medium, SysPrompt=SysPromptJson
    )
    subtopics = json_from_text(response_subtopics)

    return subtopics


def topics_interface():
    st.title("🧑‍🔬 Researcher Pro")

    col1, col2 = st.columns(2)

    if "submit_query" not in st.session_state:
        st.session_state.submit_query = False
    if "generated" not in st.session_state:
        st.session_state.generated = False
    if "user_query_full" not in st.session_state:
        st.session_state.user_query_full = ""
    if "user_query_keywords" not in st.session_state:
        st.session_state.user_query_keywords = []
    if "full_data" not in st.session_state:
        st.session_state.full_data = []
    if "df_tables_list" not in st.session_state:
        st.session_state.df_tables_list = []
    if "report_format" not in st.session_state:
        st.session_state.report_format = "Default"
    if "data_format" not in st.session_state:
        st.session_state.data_format = "Default"

    with col1:

        user_input = st.text_input(
            "Business/Industry", st.session_state.user_query_full
        )
        st.session_state.internet_toggle = st_toggle_switch(
            label="Web Search",
            key="Key1",
            default_value=True,
            label_after=False,
            inactive_color="#D3D3D3",
            active_color="#11567f",
            track_color="#29B5E8",
        )
        # user_input += " business analysis"
        st.session_state["user_query_full"] = user_input

        advanced_options()

        def clicked():
            st.session_state.submit_query = True
            st.session_state.generated = False
            clear_cache()

        st.button(label="Generate subtopics", on_click=clicked)
        if st.button("Quick Report"):
            generate_report(st.session_state["user_query_full"])

        if st.session_state.submit_query == True:
            # st.session_state.submit_query = False
            if st.session_state.generated == False:
                with st.spinner(text="Please wait..."):
                    st.session_state.data = generate_topics(user_input)
                    st.session_state.generated = True
            else:
                data = st.session_state.data
            data = st.session_state.data
            # st.subheader(query)
            st.write("Select the components to be included in final report")
            nodes = []
            seen_datapoints = set()  # Keep track of seen datapoints
            for sub_task in data:
                children = []
                for datapoint in data[sub_task]:
                    if (
                        datapoint[1] not in seen_datapoints
                    ):  # fix duplicate component error
                        children.append({"label": datapoint[0], "value": datapoint[1]})
                        seen_datapoints.add(datapoint[1])

                final_node = {
                    "label": sub_task,
                    "value": sub_task,
                    "children": children,
                }
                nodes.append(final_node)

            return_select = tree_select(
                nodes,
                check_model="leaf",
                only_leaf_checkboxes=True,
                show_expand_all=True,
            )
            missing_topics = st.text_input(
                label="🧑‍🎓 : Missing some topics? Add it here..."
            )
            if st.button("Add"):
                topics = generate_missing_topics(user_input=missing_topics)
                st.session_state.data.update(topics)

    with col2:
        st.subheader("Report elements")
        if st.session_state.submit_query == True:
            st.write("\n* ".join(return_select["checked"]))
            st.session_state.return_select = return_select

    if "return_select" in st.session_state:
        if st.button("Generate Report"):
            for query in st.session_state.return_select["checked"]:
                generate_report(query)

    if st.session_state.get("html_report_content", False):
        with st.container():
            st.write("## Generated Reports ", unsafe_allow_html=True)
            for idx, report in enumerate(st.session_state.html_report_content):
                report_interface(report, idx)

        # Create directory if it doesn't exist
        directory = "generated_reports"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write DataFrame to Excel file
        file_name = (
            re.sub(
                r"\W+", "", st.session_state["user_query_full"].replace(" ", "_")
            ).title()
            + f"_{datetime.now().strftime('%Y-%m-%d--%Hh-%Mm-%Ss')}"
        )

        excel_path = f"{directory}/" + file_name + ".xlsx"

        # pd.DataFrame(st.session_state.full_data).to_excel(excel_path, index=False)

        html_report = generate_report_with_reference(st.session_state.full_data)
        with st.container():
            st.download_button(
                "Download Combined HTML Report",
                html_report,
                file_name=f"{file_name}.html",
            )
            st.write("interactive")

        with open("generated_pdf_report.pdf", "rb") as f:
            st.download_button(
                "Download Combined Pdf Report", f, file_name=file_name + ".pdf"
            )

        write_dataframes_to_excel(st.session_state.df_tables_list, excel_path)

        if os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                st.download_button(
                    "Download Combined Excel Report", f, file_name=file_name + ".xlsx"
                )


def generate_report(query):

    st.write(f"Creating report for {query}")
    if "html_report_content" not in st.session_state:
        st.session_state.html_report_content = []

    st.write(f"Internet is {str(st.session_state.internet_toggle)}")
    if st.session_state.internet_toggle:
        search_query = query + str(st.session_state.user_query_keywords)

        # Search for relevant URLs
        urls = search_brave(search_query, num_results=6)

        # Fetch and extract content from the URLs
        all_text_with_urls = fetch_and_extract_content(
            data_format=st.session_state.data_format,
            urls=urls,
            query=query,
            num_refrences=4,
        )

        # Display content of the urls in the application
        for item in all_text_with_urls:
            with st.expander(f"Fetched relevant data from url: {item[1]}"):
                st.markdown(item[0], unsafe_allow_html=True, help=None)

        prompt = f"#### ADDITIONAL CONTEXT:{limit_tokens(str(all_text_with_urls))} #### perform user query:{query} #### IN THE CONTEXT OF: {str(st.session_state.user_query_keywords)}"
        SysPrompt = sys_prompts["SysPromptOnline"][st.session_state.report_format]

    else:
        prompt = f"perform user query:{query}"
        SysPrompt = sys_prompts["SysPromptOffline"][st.session_state.report_format]
        all_text_with_urls = "[]"

    md_report = together_response(
        prompt, model=llm_default_medium, SysPrompt=SysPrompt, frequency_penalty=0.5
    )
    st.session_state.html_report_content.append(md_report)
    # Generate HTML report
    insert_data(
        st.session_state["user_id"],
        st.session_state["user_query_full"],
        query,
        str(all_text_with_urls),
        md_report,
    )

    st.success(f"Report Generated!")
    with st.expander(f"# Report : {query}"):
        st.markdown(md_report, unsafe_allow_html=True, help=None)

    st.session_state.full_data.append(
        {
            "user_id": st.session_state["user_id"],
            "query": query,
            "text_with_urls": str(all_text_with_urls),
            "md_report": md_report,
        }
    )


def report_interface(report, idx):
    with st.expander("# Final Report " + str([idx])):
        st.markdown(report, unsafe_allow_html=True)


def recommendation_button(topic):
    st.session_state.recommendation_query = topic
    st.session_state["user_query_full"] = topic


def followup_questions():

    if "followup_question" not in st.session_state:
        st.session_state.followup_question = None
    if "recommendation_query" not in st.session_state:
        st.session_state.recommendation_query = None
    if "recommendation_topics" not in st.session_state:
        st.session_state.recommendation_topics = None

    if st.session_state["user_query_full"]:
        prompt = f"""create a list of 6 questions that a user might ask following the question: {st.session_state['user_query_full']}:"""
    else:
        prompt = """create a list of mixed 6 questions to create a report or plan or course on any of the topics product,market,research topic """

    if st.session_state.user_query_full != st.session_state.recommendation_query:
        response_topics = json_from_text(
            together_response(
                prompt, model=llm_default_small, SysPrompt=SysPromptList, temperature=1
            )
        )
        st.session_state.recommendation_topics = response_topics
        st.session_state.recommendation_query = st.session_state.user_query_full
    else:
        response_topics = st.session_state.recommendation_topics

    for topic in response_topics:
        st.button(topic, on_click=recommendation_button, args=(topic,))


def advanced_options():
    with st.expander("### Advanced Options"):
        st.session_state.report_format = st.radio(
            "Select how you want the output to be formatted",
            ["Default", "Full Text Report", "Tabular Report", "Tables only"],
            captions=[
                "No presets (Depends on query)",
                "Detailed Research Report",
                " Report focusing on structured data",
                "Only Tables",
            ],
        )

        if st.session_state.internet_toggle:
            st.session_state["data_format"] = str(
                st.radio(
                    "The following options determine what data would be extracted from internet",
                    ["Default ", "Structured data", "Quantitative data"],
                    captions=[
                        "No presets (Depends on query)",
                        "Extract structured data (e.g. categories, tables)",
                        "Extract numbers",
                    ],
                )
            )
            st.write(sys_prompts["SysPromptOnline"][st.session_state.report_format])
            st.write(st.session_state.data_format)

        else:
            st.write(sys_prompts["SysPromptOffline"][st.session_state.report_format])


def clear_cache():
    memmory_list = ["data", "html_report_content", "full_data", "df_tables_list"]
    for item in memmory_list:
        if item in st.session_state:
            st.session_state.pop(item)


if __name__ == "__main__":
    topics_interface()
    followup_questions()
