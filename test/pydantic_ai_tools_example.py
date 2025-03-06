from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from loguru import logger

logger.configure(handlers=[{"sink": "stderr", "level": "INFO"}])

# Define data models
class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str

class WeatherInfo(BaseModel):
    temperature: float
    condition: str
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None

class TravelPlan(BaseModel):
    destination: str
    weather: WeatherInfo
    places_to_visit: List[str] = Field(default_factory=list)
    budget_required: float
    travel_advice: str

# Create an agent that plans travel itineraries
travel_agent = Agent(
    'openai:gpt-4o',
    result_type=TravelPlan,
    system_prompt=(
        'You are a travel planning assistant. Use the available tools to gather information '
        'about potential destinations and provide a detailed travel plan.'
    ),
)

# Tool to search for information about locations
@travel_agent.tool
async def search_destination(ctx: RunContext, query: str) -> List[SearchResult]:
    """Search for information about a destination."""
    logger.info(f"Searching for: {query}")
    
    # In a real implementation, this would call an actual search API
    # This is just a mock example
    if "paris" in query.lower():
        return [
            SearchResult(
                title="Paris - The City of Light",
                snippet="Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere.",
                url="https://example.com/paris"
            ),
            SearchResult(
                title="Top 10 Attractions in Paris",
                snippet="Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Arc de Triomphe...",
                url="https://example.com/paris-attractions"
            )
        ]
    else:
        # Default response for any other destination
        return [
            SearchResult(
                title=f"Travel Guide: {query}",
                snippet=f"Discover everything you need to know about visiting {query}.",
                url=f"https://example.com/{query.lower().replace(' ', '-')}"
            )
        ]

# Tool to get weather information
@travel_agent.tool
async def get_weather(ctx: RunContext, location: str, date: Optional[str] = None) -> WeatherInfo:
    """Get weather information for a location, optionally for a specific date."""
    logger.info(f"Getting weather for {location}" + (f" on {date}" if date else ""))
    
    # Mock weather data
    return WeatherInfo(
        temperature=22.5,
        condition="Sunny with occasional clouds",
        humidity=65.0,
        wind_speed=10.2
    )

# Tool to estimate travel budget
@travel_agent.tool
async def estimate_budget(ctx: RunContext, destination: str, duration_days: int, luxury_level: str = "medium") -> float:
    """Estimate a travel budget based on destination, duration, and luxury level."""
    logger.info(f"Estimating budget for {destination} for {duration_days} days at {luxury_level} level")
    
    # Simple budget calculation logic
    base_costs = {
        "low": 50,
        "medium": 150,
        "high": 400,
    }
    
    # Apply destination multiplier (mock logic)
    destination_multipliers = {
        "paris": 1.5,
        "new york": 1.7,
        "tokyo": 1.6,
        "bali": 0.8,
    }
    
    multiplier = destination_multipliers.get(destination.lower(), 1.0)
    daily_cost = base_costs.get(luxury_level.lower(), base_costs["medium"])
    
    return round(daily_cost * duration_days * multiplier, 2)

# Example usage
def plan_trip(destination: str, days: int) -> TravelPlan:
    """Generate a travel plan for the specified destination and duration."""
    prompt = f"""
    I'm planning to visit {destination} for {days} days.
    Can you help me create a detailed travel plan?
    Please include weather information, budget estimate, and places to visit.
    """
    
    # Run the agent synchronously
    result = travel_agent.run_sync(prompt)
    logger.info(f"Travel plan generated for {destination}")
    return result.data

if __name__ == "__main__":
    # Example execution
    plan = plan_trip("Paris", 5)
    print(f"Destination: {plan.destination}")
    print(f"Weather: {plan.weather.condition}, {plan.weather.temperature}°C")
    print(f"Places to visit: {', '.join(plan.places_to_visit)}")
    print(f"Estimated budget: ${plan.budget_required}")
    print(f"Travel advice: {plan.travel_advice}") 