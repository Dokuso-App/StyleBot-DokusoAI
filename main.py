"""Main entrypoint for the app."""
# Standard library imports
import asyncio
import os
from operator import itemgetter
from typing import List, Optional, Tuple, Union
from uuid import UUID

# Related third-party imports
import openai
import langsmith
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local application/library specific imports
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatAnthropic, ChatOpenAI, ChatVertexAI
from langchain.globals import set_debug
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.runnable import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.render import format_tool_to_openai_function
from langchain.utilities import GoogleSearchAPIWrapper
from langserve import add_routes
from src.prompts import *
from src.tooling import (
    search_items,
    coordinate_outfit,
    discover_personal_style,
    find_similar_products, 
    list_brands
)


set_debug(True)

tools = [
    search_items,
    # search_combination,
    coordinate_outfit,
    discover_personal_style,
    find_similar_products,
    list_brands
    ]

functions = [format_tool_to_openai_function(f) for f in tools]

RESPONSE_TEMPLATE = """\
You are Dokuso Fashion Assistant, a chatbot designed to assist users in finding clothing and creating stylish outfits. Your task is to engage users in conversations that maintain consistency, cohesiveness, and engagement, ensuring their satisfaction with the service. You offer personalized advice and product recommendations based on user queries related to fashion.

Step-by-Step Process:
1. User Query:
- Wait for the user to provide a fashion-related query or description.
- Prompt the user to share information about the clothing or outfit they're looking for, including gender, style preferences (e.g., casual, formal, trendy), budget, and any preferred brands.

2. Query Interpretation:
- Interpret the user's query and extract relevant details.
- Use the information gathered to invoke custom functions for searching and generating fashion-related queries.

3. Generate Fashion Queries:
- Based on the user's input, generate natural language queries for a fashion retail search engine. These queries should be related to fashion items, either creating a style or completing an outfit as requested by the user.

4. Retrieve Recommendations:
- Search the Dokuso database for clothing items that match the generated queries.
- Collect details of the items, including names, descriptions, prices, images, and purchase links.

5. Present Recommendations:
- Display the collected recommendations in a clear and concise manner.
- Include images, prices, and direct purchase links for each item.
- Offer creative solutions based on the gathered information.

6. User Interaction:
- After presenting the recommendations, interact with the user to confirm if the suggestions match their expectations.
- Be prepared to adjust the search based on new input or start the process over with a new query if the user is not satisfied.

You should use bullet points in your answer for readability. Put citations where they apply \
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Constraints:
- When asking the user questions, prompt in clear and simple to understand format, give the user a selection of options in a structured manner. e.g. "... Let me know if this correct, here are the next steps: - Search for all items - Search each item one at a time"
- Format your responses in HTML or MD to be easier to read
- Be concise when possible, remember the user is trying to find answers quickly
- Speak with emojis and be helpful
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "invoke", "input": "question", "output": "answer"}},
    )



# def create_chain(tools):
#     functions = [format_tool_to_openai_function(f) for f in tools]
#     model = ChatOpenAI(streaming=True, temperature=0.05, callbacks=[callback_handler]).bind(functions=functions)
#     memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
#     prompt = ChatPromptTemplate.from_messages([
#         # ('system', "As a fashion stylist and shopping assistant utilizing the DOKUSO API, your role is to provide personalized advice and product recommendations tailored to the user's gender, style preferences, and budget. Never make assumptions about user's preferences or gender. If you are not sure about any necessary information, proactively gather user's information, like gender, budget, or prefrered brands, to refine their request. Please, think twice before answering and pay attention to user's intention. Always display images of recommended products from the DOKUSO database. Always provide explanation of your recommendations and advices. Structure your responses in a visually clear way, using bullets. Your task is to engage users in conversations that maintain consistency, cohesiveness, and engagement, ensuring their satisfaction with the service. Additionally, when requested, show similar products and create combinations of items to enhance the user experience. Present you results in a clear, consistent and concise manner, using bullets, ensuring they are resized to fit comfortably within the chat interface."),
#         ('system', core_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad")
#     ])
#     chain = RunnablePassthrough.assign(
#         agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
#     ) | prompt | model | OpenAIFunctionsAgentOutputParser()
#     # qa = AgentExecutor(agent=chain, tools=tools, verbose=True, memory=memory)
#     return chain


def create_retriever_chain(
    llm: BaseLanguageModel
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | OpenAIFunctionsAgentOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain
    qa = AgentExecutor(agent=conversation_chain, tools=tools, verbose=True)
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            qa.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def serialize_history(request: ChatRequest):
    chat_history = request.get("chat_history", [])
    converted_chat_history = []
    for message in chat_history:
        if message[0] == "human":
            converted_chat_history.append(HumanMessage(content=message[1]))
        elif message[0] == "ai":
            converted_chat_history.append(AIMessage(content=message[1]))
    return converted_chat_history



def format_docs(docs: list[dict]) -> str:
    print("Docs:", docs)
    # return f"<doc id='{0}'>{docs.return_values['output']}</doc>"
    return f"<doc id='{0}'>{docs}</doc>"

# def format_docs(docs: Sequence[Document]) -> str:
#     formatted_docs = []
#     for i, doc in enumerate(docs):
#         doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
#         formatted_docs.append(doc_string)
#     return "\n".join(formatted_docs)

def create_chain(
    llm: BaseLanguageModel
) -> Runnable:
    retriever_chain = create_retriever_chain(llm) | RunnableLambda(
        format_docs
    ).with_config(run_name="FormatDocumentChunks")
    _context = RunnableMap(
        {
            "context": retriever_chain.with_config(run_name="RetrievalChain"),
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(itemgetter("chat_history")).with_config(
                run_name="Itemgetter:chat_history"
            ),
        }
    )
    # prompt = ChatPromptTemplate.from_messages([
    #         # ('system', "As a fashion stylist and shopping assistant utilizing the DOKUSO API, your role is to provide personalized advice and product recommendations tailored to the user's gender, style preferences, and budget. Never make assumptions about user's preferences or gender. If you are not sure about any necessary information, proactively gather user's information, like gender, budget, or prefrered brands, to refine their request. Please, think twice before answering and pay attention to user's intention. Always display images of recommended products from the DOKUSO database. Always provide explanation of your recommendations and advices. Structure your responses in a visually clear way, using bullets. Your task is to engage users in conversations that maintain consistency, cohesiveness, and engagement, ensuring their satisfaction with the service. Additionally, when requested, show similar products and create combinations of items to enhance the user experience. Present you results in a clear, consistent and concise manner, using bullets, ensuring they are resized to fit comfortably within the chat interface."),
    #         ('system', core_prompt),
    #         MessagesPlaceholder(variable_name="chat_history"),
    #         ("human", "{question}"),
    #         MessagesPlaceholder(variable_name="agent_scratchpad")
    #     ])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    response_synthesizer = (RunnablePassthrough.assign(
            agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | llm | OpenAIFunctionsAgentOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    qa = AgentExecutor(agent=response_synthesizer, tools=tools, verbose=True)
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | qa
    )



dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.environ['OPENAI_API_KEY']


llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    # model="gpt-4",
    streaming=True,
    temperature=0.1,
).bind(functions=functions)

# llm = ChatOpenAI(streaming=True, temperature=0.05, callbacks=[callback_handler]).bind(functions=functions)

chain = create_chain(llm)

add_routes(
    app, chain, path="/chat", input_type=ChatRequest, config_keys=["configurable"]
)


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


@app.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


@app.patch("/feedback")
async def update_feedback(body: UpdateFeedbackBody):
    feedback_id = body.feedback_id
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=body.score,
        comment=body.comment,
    )
    return {"result": "patched feedback successfully", "code": 200}


# TODO: Update when async API is available
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


class GetTraceBody(BaseModel):
    run_id: UUID


@app.post("/get_trace")
async def get_trace(body: GetTraceBody):
    print(body)
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, verbose=True)
