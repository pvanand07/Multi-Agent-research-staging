import streamlit as st
from streamlit_tree_select import tree_select
from streamlit_toggle import st_toggle_switch

st.set_page_config(layout="wide")

import os

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

from src.helper_functions import (
    together_response,
    write_dataframes_to_excel,
    generate_report_with_reference,
    json_from_text,
)

from src.inference import generate_topics, generate_missing_topics, generate_report

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
    if "html_report_content" not in st.session_state:
        st.session_state.html_report_content = []

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
            quick_report, refrences = generate_report(
                internet_search=st.session_state.internet_toggle,
                query=st.session_state.user_query_full,
                data_format=st.session_state.data_format,
                report_format=st.session_state.report_format,
                user_query_keywords=st.session_state.user_query_keywords,
                user_id=st.session_state.user_id,
                user_query_full=st.session_state.user_query_full,
            )
            for item in refrences:
                with st.expander(f"Fetched relevant data from url: {item[1]}"):
                    st.markdown(item[0], unsafe_allow_html=True, help=None)

            st.success(f"Report Generated", icon="✅")
            with st.expander(f"# Report : {st.session_state.user_query_full}"):
                st.markdown(quick_report, unsafe_allow_html=True, help=None)

        if st.session_state.submit_query == True:
            # st.session_state.submit_query = False
            if st.session_state.generated == False:
                with st.spinner(text="Please wait..."):
                    st.session_state.data, st.session_state.user_query_keywords = (
                        generate_topics(user_input)
                    )
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

            st.session_state.return_select = tree_select(
                nodes,
                check_model="leaf",
                only_leaf_checkboxes=True,
                show_expand_all=True,
            )
            missing_topics = st.text_input(
                label="🧑‍🎓 : Missing some topics? Add it here..."
            )
            if st.button("Add"):
                topics = generate_missing_topics(
                    user_input=missing_topics,
                    user_query_keywords=st.session_state.user_query_keywords,
                )
                st.session_state.data.update(topics)

    with col2:
        st.subheader("Report elements")
        if st.session_state.submit_query == True:
            st.write("* " + "\n* ".join(st.session_state.return_select["checked"]))

    if "return_select" in st.session_state:
        if st.button("Generate Report"):
            for query in st.session_state.return_select["checked"]:

                st.write(f"Creating report for {query}")
                md_report, all_text_with_urls = generate_report(
                    internet_search=st.session_state.internet_toggle,
                    query=query,
                    data_format=st.session_state.data_format,
                    report_format=st.session_state.report_format,
                    user_query_keywords=st.session_state.user_query_keywords,
                    user_id=st.session_state.user_id,
                    user_query_full=st.session_state.user_query_full,
                )
                full_data = {
                    "user_id": st.session_state.user_id,
                    "query": query,
                    "text_with_urls": str(all_text_with_urls),
                    "md_report": md_report,
                }
                for item in all_text_with_urls:
                    with st.expander(f"Fetched relevant data from url: {item[1]}"):
                        st.markdown(item[0], unsafe_allow_html=True, help=None)

                st.success(f"Report Generated", icon="✅")
                with st.expander(f"# Report : {query}"):
                    st.markdown(md_report, unsafe_allow_html=True, help=None)
                st.session_state.html_report_content.append(md_report)
                st.session_state.full_data.append(full_data)

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
