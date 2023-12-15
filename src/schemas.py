from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
import requests

baseUrl = 'https://clipfashion-gr7fp45wya-uc.a.run.app/api/v1'
brandList = requests.get(f'{baseUrl}/brands_list').json()['brand_list']

# Dynamically creating the AllowedBrand enum
AllowedBrands = Enum('AllowedBrands', {brand: brand for brand in brandList})

# Define the input schema
class ItemId(BaseModel):
    itemId: str = Field(..., description=" The unique identifier of the items.")
    
class SearchItemsQueryInput(BaseModel):
    query: str = Field(..., description="Natural language query for the search")
    maxPrice: Optional[float] = Field(None, description="Maximum price of items to retrieve. If not mentioned None is used.")
    category: str = Field(None, description="Category of the items to retrieve. If not mentioned None is used.", enum=['women', 'men', 'home', 'kids'])
    onSale: Optional[bool] = Field(None, description="Whether to search for items on sale. If not mentioned None is used.")
    brands: Optional[List[AllowedBrands]] = Field(None, description="List of brands of the items. If not mentioned None is used.")
    limit: Optional[int] = Field(5, description="Number of items to retrieve. If not mentioned the default value is 5.")

class SearchCombinationInput(BaseModel):
    userRequest: str = Field(..., description="User intention")
    count: Optional[int] = Field(3, description="Number of queries to generate")
    maxPrice: Optional[float] = Field(None, description="Maximum price of items to retrieve. If not mentioned None is used.")
    category: str = Field(None, description="Category of the items to retrieve. If not mentioned None is used.", enum=['women', 'men', 'home', 'kids'])
    onSale: Optional[bool] = Field(None, description="Whether to search for items on sale. If not mentioned None is used.")
    brands: Optional[List[AllowedBrands]] = Field(None, description="List of brands of the items. If not mentioned None is used.")
    limit: Optional[int] = Field(1, description="Number of items to retrieve. If not mentioned the default value is 5.")

class FashionQueriesGenerator(BaseModel):
    """Tag the piece of text with particular info."""
    queries: List[str] = Field(description="Generated fashion queries based on the base item and preferences.")


class CoordinateOutfitInput(BaseModel):
    baseItem: str = Field(..., description="A base item or color/style to start with for the outfit coordination")
    includeAccessories: Optional[bool] = Field(True, description="Whether to include accessories in the suggestions")
    gender: Optional[str] = Field(None, description="Gender for which the outfit is intended (e.g., women, men, unisex)")
    countOutfits: Optional[int] = Field(3, description="Number of outits to suggest for the outfit coordination")
    limit: Optional[int] = Field(3, description="Number of items in each outfit to suggest for the outfit coordination")


class StyleDiscoveryInput(BaseModel):
    userPreferences: List[str] = Field(..., description="List of user's style preferences and interests")
    lifestyle: str = Field(..., description="The user's lifestyle or typical activities (e.g., professional, casual, active)")
    favoriteColors: List[str] = Field(..., description="List of user's favorite colors")
    dislikedItems: List[str] = Field(default=[], description="List of items or styles the user dislikes")
    limit: int = Field(5, description="Number of style suggestions to retrieve")


# class QueriesGenerator(BaseModel):
#     """Tag the piece of text with particular info."""
#     queries: list[str] = Field(description="")
