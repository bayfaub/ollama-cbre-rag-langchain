from langchain.tools import BaseTool

class CBREInsightsScrapperTool(BaseTool):

    name = "CBREInsightsScrapperTool"
    description = "This tool should be used when you are asked for commercial real estate data and you think there could be new market insight documents "
