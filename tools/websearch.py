
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

if os.getenv("TAVILY_API_KEY") is not None:
    web_search = TavilySearch(
        max_results = 1,
        handle_tool_errors = True,
        search_depth = 'advanced',
        time_range = 'month',
        topic = 'finance',
        include_raw_content = True,
    )
    print(f"Search Results: {web_search}")
else:
    web_search = DuckDuckGoSearchResults(
        max_results = 1,
        results_separator="\n\n",
        output_format='string',
    )
    print(f"Search Results: {web_search}")




