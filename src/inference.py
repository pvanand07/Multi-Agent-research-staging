from typing import Dict, Any, List, Tuple
from .helper_functions import (
    together_response,
    json_from_text,
    insert_data,
    search_brave,
    fetch_and_extract_content,
    limit_tokens,
)


SysPromptJson = "You are now in the role of an expert AI who can extract structured information from user request. Both key and value pairs must be in double quotes. You must respond ONLY with a valid JSON file. Do not add any additional comments."
SysPromptList = "You are now in the role of an expert AI who can extract structured information from user request. All elements must be in double quotes. You must respond ONLY with a valid python List. Do not add any additional comments."
SysPromptDefault = (
    "You are an expert AI, complete the given task. Do not add any additional comments."
)

llm_default_small = "meta-llama/Llama-3-8b-chat-hf"
llm_default_medium = "meta-llama/Llama-3-70b-chat-hf"

sys_prompts = {
    "SysPromptOffline": {
        "Default": "You are an expert AI, complete the given task. Respond in markdown and Do not add any additional comments.",
        "Full Text Report": "You are an expert AI who can create a detailed report from user request. The report should be in markdown format. Do not add any additional comments.",
        "Tabular Report": "You are an expert AI who can create a structured report from user request.The report should be in markdown format structured into subtopics/tables/lists. Do not add any additional comments.",
        "Tables only": "You are an expert AI who can create a structured tabular report from user request.The report should be in markdown format consists of only markdown tables. Do not add any additional comments.",
    },
    "SysPromptOnline": {
        "Default": "You are an expert AI, complete the given task using the provided context. Do not add any additional comments.",
        "Full Text Report": "You are an expert AI whoes task is to create a detailed report on QUERY using information provided in the CONTEXT from user request. The report should be in markdown format. Do not add any additional comments.",
        "Tabular Report": "You are an expert AI whoes task is to create a structured report using information provided in the context from user request. The report should be in markdown format structured into subtopics/tables/lists. Do not add any additional comments.",
        "Tables only": "You are an expert AI who can create a structured tabular report using information provided in the context from user request. The report should be in markdown format consists of only markdown tables. Do not add any additional comments.",
    },
}


def generate_topics(user_input: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create a dictionary of the part (part is the component of main user query) and its subtopics.
    """

    prompt_topics = f"""CREATE A LIST OF 5-10 CONCISE SUBTOPICS TO FOLLOW FOR COMPLETING ###{user_input}###, RETURN A VALID PYTHON LIST"""
    prompt_keywords = f"""EXTRACT KEYWORDS(NOUN) FROM USER INPUT ###{user_input}###, RETURN A VALID PYTHON LIST"""

    response_topics = together_response(
        prompt_topics, model=llm_default_small, SysPrompt=SysPromptList
    )
    topics = json_from_text(text=response_topics)

    user_query_keywords = json_from_text(
        together_response(
            message=prompt_keywords, model=llm_default_small, SysPrompt=SysPromptList
        )
    )

    prompt_subtopics = (
        f"""List 2-5  subtopics for each topic in the list of '{topics}' covering all aspects to generate a report, within the context of ###{user_query_keywords}###. to achieve each subtopic add a detailed instruction. (these instruction will be used as a LLM prompt and a web-search query, hence make it contextual.)"""
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
        frequency_penalty=0.5,
    )
    subtopics = json_from_text(response_subtopics)

    return subtopics, user_query_keywords


def generate_missing_topics(
    user_input: str, user_query_keywords: List[str]
) -> Dict[str, Any]:
    """
    Create a dictionary similar to the `generate_topics` function, but for missing parts.
    """

    topics = user_input

    prompt_subtopics = (
        f"""List 2-5  Subtopics for the given Topic : ###{topics}### covering all aspects to generate a report, within the context of these keywords : ###{user_query_keywords}###, also to achieve each subtopic add a detailed Instruction that will be used as a LLM promt and a web-search query, hence keep it contextual."""
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


def generate_report(
    query: str,
    internet_search: bool,
    data_format: str,
    report_format: str,
    user_query_keywords: List[str],
    user_id: str,
    user_query_full: str,
) -> Tuple[str, List[Tuple[str | None, str]]] | Any:

    if internet_search:
        search_query = query + str(user_query_keywords)

        # Search for relevant URLs
        urls = search_brave(search_query, num_results=6)

        # Fetch and extract content from the URLs
        all_text_with_urls = fetch_and_extract_content(
            data_format=data_format,
            urls=urls,
            query=query,
            num_refrences=4,
        )

        prompt = f"""#### CONTEXT:{limit_tokens(str(all_text_with_urls))} #### QUERY :{query} #### IN THE CONTEXT OF: {str(user_query_keywords)} """
        SysPrompt = sys_prompts["SysPromptOnline"][report_format]
        print(len(prompt))

    else:
        prompt = f"perform user query:{query}"
        SysPrompt = sys_prompts["SysPromptOffline"][report_format]
        all_text_with_urls = "[]"

    md_report = together_response(
        prompt, model=llm_default_medium, SysPrompt=SysPrompt, frequency_penalty=0.01
    )
    insert_data(
        user_id=user_id,
        user_query=user_query_full,
        subtopic_query=query,
        response=str(all_text_with_urls),
        html_report=md_report,
    )

    return md_report, all_text_with_urls


def generate_followup_questions(user_query_full: str | None) -> List[str]:

    if user_query_full != None:
        prompt = f"""create a list of 6 questions that a user might ask following the question: {user_query_full}:"""
    else:
        prompt = """create a list of mixed 6 questions to create a report or plan or course on any of the topics product,market,research topic """

    response_topics = json_from_text(
        together_response(
            prompt, model=llm_default_small, SysPrompt=SysPromptList, temperature=1
        )
    )
    return response_topics
