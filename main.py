# Standard library imports
import os
from typing import Optional

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
from tooling import (search_items, search_combination, find_similar_products, list_brands)


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

tools = [search_items, search_combination, find_similar_products, list_brands]

# CHATBOT
class cbfs(param.Parameterized):
    """Chatbot Fashion Stylist"""
    
    def __init__(self, tools, callback_handler, **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.functions = [format_tool_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(streaming=True, temperature=0.1, callbacks=[callback_handler]).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', "As a fashion stylist and shopping assistant utilizing the DOKUSO API, your role is to provide personalized advice and product recommendations tailored to the user's gender, style preferences, and budget. Provide a gamifing experience. When assisting users, proactively gather user's information, like gender, budget, or prefrered brands, to refine their request. Please, think before answering and pay attention to user's intention. If not sure about any other relevant information. Ask in a clear way using bullets. Always display images of recommended products from the DOKUSO database.  Always provide explanation of your recommendations and advices. Structure your responses in a visually clear way, using bullets. Your task is to engage users in conversations that maintain consistency, cohesiveness, and engagement, ensuring their satisfaction with the service. Offer creative solutions based on the gathered information and be proactive in understanding the specific needs of the users. Additionally, when requested, show similar products and create combinations of items to enhance the user experience. Present you results in a clear and concise manner, using bullets, ensuring they are resized to fit comfortably within the chat interface."),
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
    "Hi! I'm your fashion stylist and shopping assistant.  How can I assist you today?", user="System", respond=False
)

callback_handler = pn.chat.langchain.PanelCallbackHandler(chat_interface)
cb = cbfs(tools, callback_handler=callback_handler)


chat_interface.servable()

