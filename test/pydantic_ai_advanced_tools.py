from pydantic_ai import Agent, RunContext, EventType
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import asyncio
from datetime import datetime

logger.configure(handlers=[{"sink": "stderr", "level": "INFO"}])

class StockPrice(BaseModel):
    symbol: str
    price: float
    timestamp: datetime

class MarketInsight(BaseModel):
    symbol: str
    sentiment: str
    score: float
    source: str

class TradeRecommendation(BaseModel):
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    supporting_insights: List[MarketInsight] = Field(default_factory=list)
    price_data: StockPrice

class PortfolioAnalysis(BaseModel):
    recommendations: List[TradeRecommendation] = Field(default_factory=list)
    market_summary: str
    risk_assessment: str

# Create a financial advisor agent
financial_advisor = Agent(
    'openai:gpt-4o',
    result_type=PortfolioAnalysis,
    system_prompt=(
        'You are a financial advisor specialized in stock market analysis. '
        'Use the available tools to collect data and provide investment recommendations.'
    ),
)

# Tool to get stock prices
@financial_advisor.tool
async def get_stock_price(ctx: RunContext, symbol: str) -> StockPrice:
    """Get the current price of a stock by its ticker symbol."""
    logger.info(f"Getting price for {symbol}")
    
    # Simulate API call delay
    await asyncio.sleep(0.5)
    
    # Mock price data
    mock_prices = {
        "AAPL": 175.23,
        "MSFT": 320.45,
        "GOOGL": 135.67,
        "AMZN": 145.89,
        "TSLA": 240.56,
    }
    
    price = mock_prices.get(symbol.upper(), 100.0)  # Default price for unknown symbols
    
    # Store this in the context for other tools to access
    ctx.set_state(f"price_{symbol}", price)
    
    return StockPrice(
        symbol=symbol.upper(),
        price=price,
        timestamp=datetime.now()
    )

# Tool to analyze news sentiment
@financial_advisor.tool
async def analyze_market_sentiment(ctx: RunContext, symbol: str) -> List[MarketInsight]:
    """Analyze market sentiment for a specific stock from news and social media."""
    logger.info(f"Analyzing sentiment for {symbol}")
    
    # Simulate API call delay
    await asyncio.sleep(0.7)
    
    # Example mock sentiments
    sentiments = [
        MarketInsight(
            symbol=symbol.upper(),
            sentiment="positive",
            score=0.78,
            source="Financial News Daily"
        ),
        MarketInsight(
            symbol=symbol.upper(),
            sentiment="neutral",
            score=0.52,
            source="Investor Forum"
        )
    ]
    
    # If we have the price in context, we can use it to adjust our sentiment
    price = ctx.get_state(f"price_{symbol}")
    if price and isinstance(price, (int, float)) and price > 200:
        sentiments.append(
            MarketInsight(
                symbol=symbol.upper(),
                sentiment="cautious",
                score=0.45,
                source="Market Trends Analysis"
            )
        )
    
    return sentiments

# Reactive tool that triggers when certain events happen
@financial_advisor.tool
async def recommend_trade(
    ctx: RunContext,
    symbol: str,
    price_data: Optional[StockPrice] = None,
    insights: Optional[List[MarketInsight]] = None
) -> TradeRecommendation:
    """Generate a trade recommendation based on price data and market insights."""
    logger.info(f"Generating trade recommendation for {symbol}")
    
    # If price_data wasn't provided, try to get it from a previous tool call
    if not price_data:
        # Look for previous get_stock_price calls in the trace
        for event in ctx.trace.events:
            if (event.type == EventType.TOOL_RESULT and 
                event.name == "get_stock_price" and 
                isinstance(event.value, dict) and 
                event.value.get("symbol") == symbol.upper()):
                
                price_data = StockPrice(**event.value)
                break
        
        # If we still don't have price data, get it now
        if not price_data:
            price_data = await get_stock_price(ctx, symbol)
    
    # If insights weren't provided, try to get them from a previous tool call
    if not insights:
        # Look for previous analyze_market_sentiment calls in the trace
        for event in ctx.trace.events:
            if (event.type == EventType.TOOL_RESULT and 
                event.name == "analyze_market_sentiment" and 
                isinstance(event.value, list) and 
                len(event.value) > 0 and 
                event.value[0].get("symbol") == symbol.upper()):
                
                insights = [MarketInsight(**item) for item in event.value]
                break
        
        # If we still don't have insights, get them now
        if not insights:
            insights = await analyze_market_sentiment(ctx, symbol)
    
    # Calculate average sentiment score
    avg_score = sum(insight.score for insight in insights) / len(insights)
    
    # Simple decision logic based on price and sentiment
    action = "HOLD"
    confidence = 0.5
    reasoning = "Market conditions are stable."
    
    if avg_score > 0.7:
        action = "BUY"
        confidence = min(avg_score, 0.9)
        reasoning = "Strong positive market sentiment indicates growth potential."
    elif avg_score < 0.4:
        action = "SELL"
        confidence = min(1.0 - avg_score, 0.9)
        reasoning = "Negative market indicators suggest potential decline."
    
    return TradeRecommendation(
        symbol=symbol.upper(),
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        supporting_insights=insights,
        price_data=price_data
    )

# Example usage
def analyze_portfolio(symbols: List[str]) -> PortfolioAnalysis:
    """Analyze a portfolio of stocks and provide recommendations."""
    symbol_list = ", ".join(symbols)
    prompt = f"""
    Please analyze my portfolio containing the following stocks: {symbol_list}.
    
    For each stock:
    1. Get the current price
    2. Analyze market sentiment
    3. Provide a trade recommendation (buy, sell, or hold)
    
    Finally, give me a summary of the market outlook and assess the overall risk level of my portfolio.
    """
    
    result = financial_advisor.run_sync(prompt)
    logger.info(f"Portfolio analysis completed for {len(symbols)} stocks")
    return result.data

if __name__ == "__main__":
    # Example execution
    portfolio = analyze_portfolio(["AAPL", "MSFT", "GOOGL"])
    
    print(f"Market summary: {portfolio.market_summary}")
    print(f"Risk assessment: {portfolio.risk_assessment}")
    print("\nRecommendations:")
    
    for rec in portfolio.recommendations:
        print(f"\n{rec.symbol}: {rec.action} (Confidence: {rec.confidence:.2f})")
        print(f"Current price: ${rec.price_data.price:.2f}")
        print(f"Reasoning: {rec.reasoning}")
        print("Supporting insights:")
        for insight in rec.supporting_insights:
            print(f"  - {insight.source}: {insight.sentiment} ({insight.score:.2f})") 