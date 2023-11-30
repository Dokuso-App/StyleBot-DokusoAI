from typing import Optional
import requests
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.tools import tool
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

baseUrl = 'https://clipfashion-gr7fp45wya-uc.a.run.app/api/v1'

@tool
def list_brands():
    """
    List all the brands available in the Dokuso database.

    Returns:
    list: A list of brand names.
    """
    url = f'{baseUrl}/brands_list'
    response = requests.get(url)
    return response.json()['brand_list']


def resume_item_data(item):
    """
    Given an item dictionary, this function returns a new dictionary containing only the relevant information
    about the item.

    Parameters:
    item (dict): A dictionary containing all the information about the item.

    Returns:
    dict: A dictionary containing only the relevant information about the item.
    """
    return {
        'itemId': item['id'], #only for internal use,
        'name': item['name'],
        'description': item['desc_1'],
        'price': str(round(item['price'], 0))+item['currency'],
        'discount': str(round(item['discount_rate'], 0)*100)+'%',
        'imgUrl': item['img_url'], #+'?w=200&h=200&fit=crop&auto=format&q=80&cs=tinysrgb&crop='
        'brand': item['brand'],
        'category': item['category'],
        'shopLink': item['shop_link'],
        'onSale': item['sale']
    }

bandList = requests.get(f'{baseUrl}/brands_list').json()['brand_list']

class SearchItemsQueryInput(BaseModel):
    query: str = Field(..., description="Natural language query for the search")
    maxPrice: Optional[float] = Field(None, description="Maximum price of items to retrieve. If not mentioned None is used.")
    category: Optional[str] = Field(None, description="Category of the items to retrieve. If not mentioned None is used.")
    onSale: Optional[bool] = Field(None, description="Whether to search for items on sale. If not mentioned None is used.")
    brands: Optional[str] = Field(None, description="Brand of the items. If not mentioned None is used.", enum=bandList)
    limit: Optional[int] = Field(5, description="Number of items to retrieve. If not mentioned the default value is 5.")

@tool(args_schema=SearchItemsQueryInput)
def search_items(query: str, 
                 maxPrice: Optional[float] = None, 
                 category: Optional[str] = None, 
                 onSale: Optional[bool] = None, 
                 brands: Optional[str] = None,
                 limit: int = 5) -> list[dict]:
    """
    Useful for whe you need to search for items in the Dokuso database using DOKUSO API based on various criteria. All arguments are required.

    Parameters:
    query (str): Natural language query for the search.
    maxPrice (float): Maximum price of items to retrieve.
    category (str): Category of the items ('women', 'men', 'kids', 'home').
    onSale (bool): Whether to search for items on sale.
    brands (str: Brand of the items ('zara', 'massimo dutti', 'mango', 'h&m', 'bershka').
    limit (int: Number of items to retrieve.)

    Returns:
    list: A list of item dictionaries that match the search criteria.
    """

    total_results = []

    params = {
        'query': query,
        'maxPrice': maxPrice,
        'category': category,
        'onSale': onSale,
        'brands':  ','.join(brands.split()) if brands else None,
        'limit': limit
    }
    url = f'{baseUrl}/search'
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('results', [])
        for item in results:
            item = resume_item_data(item)
            total_results.append(item)
    else:
        print(f'Error retrieving results for "{query}"')
        print(response.text)
    return total_results


class SearchCombinationInput(BaseModel):
    userRequest: str = Field(..., description="User intention")
    count: Optional[int] = Field(3, description="Number of queries to generate")
    maxPrice: Optional[float] = Field(None, description="Maximum price of items to retrieve. If not mentioned None is used.")
    category: Optional[str] = Field(None, description="Category of the items to retrieve. If not mentioned None is used.")
    onSale: Optional[bool] = Field(None, description="Whether to search for items on sale. If not mentioned None is used.")
    brands: Optional[str] = Field(None, description="Brand of the items. If not mentioned None is used.", enum=bandList)
    limit: Optional[int] = Field(1, description="Number of items to retrieve. If not mentioned the default value is 5.")



@tool(args_schema=SearchCombinationInput)
def search_combination(userRequest: str, 
                        count: Optional[int] = 3,
                        maxPrice: Optional[float] = None, 
                        category: Optional[str] = None, 
                        onSale: Optional[bool] = None, 
                        brands: Optional[str] = None) -> list[dict]:
                        
    """
    Useful for when you need to create a specific style or to complete an outfit requested by the user.

    Parameters:
    userRequest (str): User request.
    count (int): Number of queries to generate.
    maxPrice (float): Maximum price of items to retrieve.
    category (str): Category of the items ('women', 'men', 'kids', 'home').
    onSale (bool): Whether to search for items on sale.
    brands (str: Brand of the items ('zara', 'massimo dutti', 'mango', 'h&m', 'bershka').

    Returns:
    list: A list of item dictionaries that match the search criteria.
    """

    total_results = []

    queries = generate_fashion_queries(userRequest, category, 4)
    for q in queries[:4]:
        print(f'Retrieving results for "{q}"')
        params = {
            'query': q,
            'maxPrice': maxPrice,
            'category': category,
            'onSale': onSale,
            'brands': ','.join(brands.split()) if brands else None,
            'limit': 1
        }
        url = f'{baseUrl}/search'
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            for item in results:
                item = resume_item_data(item)
                total_results.append(item)
        else:
            print(f'Error retrieving results for "{q}"')
            print(response.text)
    return total_results

class QueriesGenerator(BaseModel):
    """Tag the piece of text with particular info."""
    queries: list[str] = Field(description="")

tagging_functions = [convert_pydantic_to_openai_function(QueriesGenerator)]
    
    
def generate_fashion_queries(userRequest: str, category, count):

    desc = f"""Given an user fashion inquiry, generate {count} natural language queries to use in a fashion retail search engine. These queries should be related to fashion items. They can either create a style or complete an outfit as requested by the user.

            INPUT: "Create a style for a relaxed weekend brunch. For women"
            OUTPUT:
            "Casual linen shirt dress"
            "Comfortable slip-on espadrilles"
            "Lightweight denim jacket"
            "Straw tote bag"

            INPUT:
            "I need to find accessories to match with my new black evening gown for a gala event. For women."
            OUTPUT:
            "Elegant silver clutch evening bag"
            "Diamond stud earrings"
            "Black high heel sandals"
            "Silver bracelet"
            
            INPUT:
            "I just bought a light grey suit and need ideas for shirts and ties to combine with it. For men"
            OUTPUT:
            "White classic fit dress shirt"
            "Silk tie in navy blue"
            "Light blue slim fit shirt"
            "Patterned silk tie in burgundy"

            INPUT:
            "I have a pink blazer which I want to style with other garments. For women"
            OUTPUT:
            "White slim fit shirt"
            "Black skinny jeans"
            "Black leather belt"
            "Black leather ankle boots"

            INPUT:
            "{userRequest}. For {category}"
            OUTPUT:
        """
    
    model = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}")
    ])
    tagging_functions[0]['parameters']['properties']['queries']['description'] = desc
    model_with_functions = model.bind(
        functions=tagging_functions,
        function_call={"name": "QueriesGenerator"}
    )


    tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
    response = tagging_chain.invoke(
        {"input": userRequest}
    )
    return response['queries']

# Define the input schema
class ItemId(BaseModel):
    itemId: str = Field(..., description=" The unique identifier of the items.")
    
@tool(args_schema=ItemId)
def get_product_details(itemId:str)->list[dict]:
    """
    Useful for when you answer questions about a specific product using its item_id.

    Parameters:
    item_id (str): The unique identifier of the item.

    Returns:
    dict: A dictionary containing details of the item.
    """
    url = f'{baseUrl}/product_details?id={itemId}'
    response = requests.get(url)
    return response.json()


@tool(args_schema=ItemId)
def find_similar_products(itemId:str)->list[dict]:
    """
    Useful for when you need to find products similar to the specified item in the Dokuso database.

    Parameters:
    item_id (str): The unique identifier of the item to find similar products for.

    Returns:
    list: A list of item dictionaries similar to the specified item.
    """
    url = f'{baseUrl}/most_similar_items?id={itemId}'
    response = requests.get(url)
    return response.json()
