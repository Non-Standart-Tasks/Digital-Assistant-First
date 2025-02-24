from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
import logfire
import requests


from src.offergen import Offer, offersgen_config

# disabling logfire for now
logfire.configure(send_to_logfire="if-token-present")


# Models for structured responses
class PromptValidation(BaseModel):
    is_valid: bool = Field(
        description="Whether the prompt is valid for generating VTB family offers",
        default=False,
    )
    number_of_offers_to_generate: int = Field(
        description="How many offers to generate", default=0
    )
    reason: str = Field(description="Explanation why the prompt is invalid or valid")
    modified_prompt_for_rag_search: str = Field(
        description="Modified prompt for RAG search",
    )
    is_city: bool = Field(
    description="Boolean flag indicating if a specific location is mentioned in the request",
    default=False,
    )
    city: Optional[str] = Field(
        default=None,
        description="City name (if it exists)"
    )


class OfferMatch(BaseModel):
    offer_id: int
    relevance_score: float = Field(
        description="How relevant this offer is to the request", ge=0, le=1
    )
    match_reason: str = Field(description="Why this offer matches the user's request")


class OffersMatchResponse(BaseModel):
    matches: List[OfferMatch] = Field(
        description="List of matched offers with relevance scores"
    )


@dataclass
class RagDeps:
    k: int  # Number of top offers to return
    offers: Dict[int, Offer]  # Offer database


# First agent to validate if prompt is suitable for offer generation
validation_agent = Agent(
    offersgen_config["model"],
    result_type=PromptValidation,
    system_prompt="""
    You are a VTB Family request validator analyzing user requests for finding offers/services on vtbfamily.ru.
    
    Your tasks:
    1. Determine if the request is related to finding VTB Family offers/services
    2. Extract or determine the number of offers to generate
    3. Identify if the request specifies a particular city for the search and extract the city name.
    4. Set a boolean flag 'is_city' to true if a specific location is mentioned in the request, otherwise false.
    5. Provide a clear explanation for your decision
    6. Create a clean search prompt by removing non-essential words
    
    Rules for determining number of offers:
    - If request explicitly mentions a number (e.g., "найди 7 офферов", "покажи 3 предложения"): use that number
    - If request asks for a specific single offer (e.g., "найди оффер для...", "покажи предложение по..."): set to 1
    - If request asks for multiple offers without specifying number (e.g., "найди офферы", "какие есть предложения"): set to 5
    - If request is not about finding offers: set to 0
    
    Examples:
    - "Найди мне 7 офферов связанных с ресторанами" -> 7 offers, city: "Moscow", is_city: True
    - "Покажи предложение по фитнесу" -> 1 offer, city: "Saint Petersburg", is_city: True
    - "Какие есть офферы по развлечениям?" -> 5 offers, city: null, is_city: False
    - "Как погода сегодня?" -> 0 offers (invalid request)
    
    For the modified search prompt:
    - Remove words about quantity (e.g., "найди 5 офферов")
    - Remove references to vtbfamily.ru
    - Keep only the essential search criteria
    
    Example prompt modification:
    Input: "Найди мне пожалуйста 7 офферов из vtbfamily.ru, которые связаны с пивными барами в москве"
    Modified: "пивные бары в москве"
    
    Remember: Your main goal is to accurately validate requests, extract the number of offers to generate, identify the city if specified, and indicate via a boolean flag if a specific location is present.
    Do not generate offers yourself - focus only on request validation and processing.
    """,
)

# Second agent to match offers based on RAG
offer_matching_agent = Agent(
    offersgen_config["model"],
    deps_type=RagDeps,
    result_type=OffersMatchResponse,
    system_prompt="""
    You are a VTB Family offer matching specialist. Your role is to:
    1. Analyze the user's request and available offers
    2. Select the top-K most relevant offers (K specified in dependencies)
    3. For each selected offer, provide:
       - The offer ID
       - A relevance score (0-1)
       - A brief explanation of why it matches
    
    Focus on finding offers that best match the user's specific needs and preferences.
    Only return the exact number of offers specified by K.
    """,
)
