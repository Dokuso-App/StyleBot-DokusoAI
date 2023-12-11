# Standard library imports
import os

# Related third-party imports
import openai
import panel as pn
import param
from dotenv import load_dotenv, find_dotenv

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import (ConversationBufferMemory, 
                            ConversationBufferWindowMemory, 
                            ConversationKGMemory)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function

# Local application/library specific imports
from src.tooling import (search_items,
                        search_combination, 
                        coordinate_outfit,
                        discover_personal_style,
                        find_similar_products, 
                        list_brands)

from src.prompts import *

# _ = load_dotenv(find_dotenv()) # read local .env file
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.environ['OPENAI_API_KEY']

tools = [
    search_items,
    # search_combination,
    coordinate_outfit,
    discover_personal_style,
    find_similar_products,
    list_brands
    ]

# CHATBOT
class cbfs(param.Parameterized):
    """Chatbot Fashion Stylist"""
    
    def __init__(self, tools, callback_handler, **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.functions = [format_tool_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(streaming=True, temperature=0.05, callbacks=[callback_handler]).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.prompt = ChatPromptTemplate.from_messages([
            # ('system', "As a fashion stylist and shopping assistant utilizing the DOKUSO API, your role is to provide personalized advice and product recommendations tailored to the user's gender, style preferences, and budget. Never make assumptions about user's preferences or gender. If you are not sure about any necessary information, proactively gather user's information, like gender, budget, or prefrered brands, to refine their request. Please, think twice before answering and pay attention to user's intention. Always display images of recommended products from the DOKUSO database. Always provide explanation of your recommendations and advices. Structure your responses in a visually clear way, using bullets. Your task is to engage users in conversations that maintain consistency, cohesiveness, and engagement, ensuring their satisfaction with the service. Additionally, when requested, show similar products and create combinations of items to enhance the user experience. Present you results in a clear, consistent and concise manner, using bullets, ensuring they are resized to fit comfortably within the chat interface."),
            ('system', core_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = RunnablePassthrough.assign(
            agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=True, memory=self.memory)
    
    def convchain(self, query):
        if not query:
            return
        # inp.value = ''
        result = self.qa.invoke({"input": query})
        self.answer = result['output'] 
        return self.answer

pn.extension()


# INTERFACE
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    cb.convchain(contents)

chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="ChatGPT")
chat_interface.send(
    intro_message, user="System", respond=False
)

callback_handler = pn.chat.langchain.PanelCallbackHandler(chat_interface)
cb = cbfs(tools, callback_handler=callback_handler)


chat_interface.servable()

