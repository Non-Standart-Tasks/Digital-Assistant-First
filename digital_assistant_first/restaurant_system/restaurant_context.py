from typing import Dict, List, Optional
from digital_assistant_first.utils.yndx_restaurants import (
    analyze_restaurant_request,
    get_restaurants_by_category,
)

def fetch_restaurant_context(user_input: str, model) -> str:
    """
    Fetch and format restaurant context based on user input
    """
    restaurant_analysis = analyze_restaurant_request(user_input, model)
    restaurant_context_text = ""
    restaurants_data: List[Dict] = []
    
    if restaurant_analysis.get("restaurant_recommendation", "false").lower() == "true":
        requested_category = restaurant_analysis.get("category", "")
        if requested_category:
            restaurants_data = get_restaurants_by_category(requested_category)
            if restaurants_data:
                restaurant_context_parts = []
                
                for r in restaurants_data:
                    restaurant_context_parts.append(
                        f"Название: {r.get('name')}\n"
                        f"Режим работы: {r.get('working_hours')}\n"
                        f"Адрес: {r.get('address', {}).get('street')}\n"
                        f"Метро: {', '.join(r.get('address', {}).get('metro', []))}\n"
                        f"Описание: {r.get('description')}\n"
                        f"Категории: {', '.join(r.get('categories', []))}"
                    )
                restaurant_context_text = "\n\n".join(restaurant_context_parts)

    return restaurant_context_text 