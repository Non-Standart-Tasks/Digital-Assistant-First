from typing import Dict, List, Optional
import requests
import json
from src.offergen import Offer, offers_db, db_service
from src.offergen.agent import offer_matching_agent, PromptValidation, RagDeps
from src.offergen.vector_db import VectorDBService, Context, SearchResponse
from src.utils.logging import setup_logging
from src.utils.paths import ROOT_DIR as root_dir
import dotenv
from typing import Optional
import os

logger = setup_logging(logging_path=str(root_dir / "logs" / "digital_assistant.log"))
dotenv.load_dotenv()
def check_offer_in_city(offer: Offer, city: str) -> bool:
    """
    Checks if the offer is located in the specified city using the 2GIS API.
    """
    radius = 3000
    API_KEY = os.getenv('MAPSAPI')
    # Map city names to their coordinates (add more as needed)
    city_coordinates = json.load('city.json')
    # Use provided city's coordinates or default to Moscow
    location = city_coordinates.get(city, "37.630866,55.752256")
    
    url = (
        "https://catalog.api.2gis.com/3.0/items"
        f"?q={offer.name}"
        f"&location={location}"
        f"&radius={radius}"
        "&fields=items.point,items.address,items.name,items.contact_groups,items.reviews,items.rating"
        f"&key={API_KEY}"
    )
    
    response = requests.get(url)
    data = response.json()
    items = data.get("result", {}).get("items", [])
    # Check if any returned item has an address that includes the city name
    for item in items:
        address = item.get("address", "")
        if city.lower() in address.lower():
            return True
    return False

def load_rag_examples(
    offers_db: Dict[int, Offer], query: str, db_service: VectorDBService, city: Optional[str] = None
) -> tuple[list[Offer], list[float], list[int]]:
    """Load and filter RAG examples relevant to the query with optional city filtering"""
    docs_and_scores = db_service.search(query, k=20)
    rag_data = SearchResponse(
        documents=[
            Context(content=doc.page_content, metadata=doc.metadata)
            for doc, _ in docs_and_scores
        ],
        scores=[score for _, score in docs_and_scores],
    )
    offers, scores, offer_ids = list(), list(), list()
    for i, doc in enumerate(rag_data.documents):
        offer_id = int(doc.metadata["offer_url"].split("/")[-1])
        if offer_id in offers_db.keys() and offer_id not in offer_ids:
            offer = offers_db[offer_id]
        
            if city:
                if not check_offer_in_city(offer, city):
                    continue
            offers.append(offer)
            scores.append(rag_data.scores[i])
            offer_ids.append(offer_id)
    return offers, scores, offer_ids


def get_system_prompt_for_offers(
    validation_result: PromptValidation, prompt: str
) -> str:
    if not validation_result.is_valid:
        raise ValueError(
            f"Unable to generate system prompt for prompt: {prompt}. "
            f"Reason: {validation_result.reason}. "
            "Please ensure the request meets validation requirements."
        )

    if validation_result.number_of_offers_to_generate == 0:
        return "Вы находитесь в режиме генерации офферов, пожалуйста перейдите в стандартный режим для получения ответа на этот запрос"

    # Use the city parameter if is_city is true
    city = validation_result.city if validation_result.is_city else None

    logger.info(
        f"Loading RAG examples for prompt: {validation_result.modified_prompt_for_rag_search}"
    )
    offers, scores, offer_ids = load_rag_examples(
        offers_db, validation_result.modified_prompt_for_rag_search, db_service, city=city
    )
    logger.info(
        f"RAG examples loaded for prompt: {validation_result.modified_prompt_for_rag_search}"
    )

    rag_context = "\nRelevant offer examples:\n"
    for offer, score, offer_id in zip(offers, scores, offer_ids):
        rag_context += f"- Category: {offer.category}\n"
        rag_context += f"- Title: {offer.name}\n"
        rag_context += f"- Short description: {offer.short_description}\n"
        rag_context += f"- Full description: {offer.full_description}\n"
        rag_context += f"- Offer RAG score: {score}\n"
        rag_context += f"- Offer ID: {offer_id}\n"
        rag_context += "---\n"

    enhanced_prompt = f"{rag_context}\nUser request: {prompt}"

    deps = RagDeps(k=validation_result.number_of_offers_to_generate, offers=offers_db)
    result = offer_matching_agent.run_sync(enhanced_prompt, deps=deps)
    logger.info(f"Offer matching agent result: {result.data}")

    if result and result.data.matches and len(set(match.offer_id for match in result.data.matches).intersection(offers_db.keys())) > 0:
        information_about_relevant_offers = ""
        for match in result.data.matches:
            if match.offer_id not in offers_db.keys():
                logger.warning(f"Offer ID {match.offer_id} not found in offers database")
                continue
            offer = offers_db[match.offer_id]
            information_about_relevant_offers += f"Offer ID: {match.offer_id}\n"
            information_about_relevant_offers += f"Offer name: {offer.name}\n"
            information_about_relevant_offers += f"Offer category: {offer.category}\n"
            information_about_relevant_offers += f"Offer short description: {offer.short_description}\n"
            information_about_relevant_offers += f"Offer full description: {offer.full_description}\n"
            information_about_relevant_offers += f"Offer URL: {offer.offer_url}\n"
            information_about_relevant_offers += f"Offer match reason: {match.match_reason}\n"
            information_about_relevant_offers += "---\n"
        logger.info("System prompt for offers generated.")
        return f"""
You are a VTB Family offers formatter. Format and evaluate these offers:

{information_about_relevant_offers}

The input you receive is the user's initial search request. Use it to evaluate offer relevance.

Main tasks:
1. Format search results in markdown
2. Structure each offer clearly
3. Match offers against the initial request
4. Write everything in Russian

Write a summary that:
- Mentions the initial search request
- States how well the offers match the request
- Explains any mismatches and their potential value
- Speaks directly to the user who made the request

Use this format for each offer:
### [OFFER TITLE]
**Категория:** [CATEGORY]

**Описание предложения:**
[SHORT, CONCISE DESCRIPTION OF THE MAIN OFFER/DISCOUNT]

**Информация о компании:**
- Адрес: [ADDRESS IF AVAILABLE]
- Телефон: [PHONE IF AVAILABLE]
- Часы работы: [HOURS IF AVAILABLE]
- Сайт: [WEBSITE IF AVAILABLE]

**Ссылка на предложение:** [Подробнее на VTB Family]([OFFER URL])

---

Key requirements:
- Pull company details from the full description
- Keep descriptions brief and value-focused
- Connect your response to the user's search request
"""
    else:
        logger.warning("No relevant offers found for the search request")
        return "No relevant offers were found for the search request."