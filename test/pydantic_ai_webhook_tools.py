from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from loguru import logger
import httpx
import json
from datetime import datetime

logger.configure(handlers=[{"sink": "stderr", "level": "INFO"}])

class User(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime

class Message(BaseModel):
    id: str
    content: str
    sent_at: datetime
    sender_id: str
    recipient_id: str

class SupportTicket(BaseModel):
    id: str
    subject: str
    description: str
    status: str
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    priority: str
    category: str

class SupportResponse(BaseModel):
    ticket_id: str
    response: str
    next_steps: List[str] = Field(default_factory=list)
    links_to_docs: List[HttpUrl] = Field(default_factory=list)

# Create a support agent
support_agent = Agent(
    'openai:gpt-4o',
    result_type=SupportResponse,
    system_prompt=(
        'You are a customer support agent for a SaaS product. '
        'Use the available tools to retrieve customer data and provide accurate, '
        'helpful responses to support tickets.'
    ),
)

# Mock API client for demo purposes
class APIMockClient:
    BASE_URL = "https://api.example.com"
    API_KEY = "sk_test_example123"
    
    async def make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate an API request with mock responses."""
        logger.info(f"Making {method} request to {endpoint}")
        
        # Simulate network delay
        await httpx.AsyncClient().aclose()  # Just for a brief delay
        
        # Mock database of users
        users = {
            "usr_123": {
                "id": "usr_123",
                "email": "john@example.com",
                "name": "John Smith",
                "created_at": "2023-01-15T14:30:00Z"
            },
            "usr_456": {
                "id": "usr_456",
                "email": "sarah@example.com",
                "name": "Sarah Johnson",
                "created_at": "2023-03-22T09:15:00Z"
            }
        }
        
        # Mock database of tickets
        tickets = {
            "tkt_789": {
                "id": "tkt_789",
                "subject": "Cannot access dashboard",
                "description": "After the recent update, I can no longer access my analytics dashboard. It shows a 404 error.",
                "status": "open",
                "user_id": "usr_123",
                "created_at": "2023-05-10T11:45:00Z",
                "priority": "high",
                "category": "access_issue"
            },
            "tkt_012": {
                "id": "tkt_012",
                "subject": "Billing question",
                "description": "I was charged twice for my subscription this month. Can you please help me resolve this?",
                "status": "open",
                "user_id": "usr_456",
                "created_at": "2023-05-12T15:20:00Z",
                "priority": "medium",
                "category": "billing"
            }
        }
        
        # Mock database of message history
        messages = {
            "tkt_789": [
                {
                    "id": "msg_111",
                    "content": "I've been trying to access my dashboard all day but keep getting errors.",
                    "sent_at": "2023-05-10T11:50:00Z",
                    "sender_id": "usr_123",
                    "recipient_id": "support"
                }
            ],
            "tkt_012": [
                {
                    "id": "msg_222",
                    "content": "I noticed two charges on my credit card statement for the same subscription.",
                    "sent_at": "2023-05-12T15:25:00Z",
                    "sender_id": "usr_456",
                    "recipient_id": "support"
                }
            ]
        }
        
        # Define mock responses based on the endpoint
        if endpoint == "/users" and method.lower() == "get":
            user_id = data.get("id") if data else None
            if user_id and user_id in users:
                return users[user_id]
            return {"error": "User not found"}
            
        elif endpoint == "/tickets" and method.lower() == "get":
            ticket_id = data.get("id") if data else None
            if ticket_id and ticket_id in tickets:
                return tickets[ticket_id]
            return {"error": "Ticket not found"}
            
        elif endpoint == "/messages" and method.lower() == "get":
            ticket_id = data.get("ticket_id") if data else None
            if ticket_id and ticket_id in messages:
                return {"messages": messages[ticket_id]}
            return {"messages": []}
            
        elif endpoint == "/tickets" and method.lower() == "post":
            # Handle ticket updates
            ticket_id = data.get("id") if data else None
            if ticket_id and ticket_id in tickets:
                for key, value in data.items():
                    if key in tickets[ticket_id] and key != "id":
                        tickets[ticket_id][key] = value
                tickets[ticket_id]["updated_at"] = datetime.now().isoformat()
                return tickets[ticket_id]
            return {"error": "Failed to update ticket"}
            
        return {"error": "Unknown endpoint or method"}

# Initialize our mock API client
api_client = APIMockClient()

# Tool to get user information
@support_agent.tool
async def get_user(ctx: RunContext, user_id: str) -> User:
    """Retrieve user information from the API."""
    response = await api_client.make_request("GET", "/users", {"id": user_id})
    
    if "error" in response:
        logger.error(f"Error retrieving user: {response['error']}")
        raise ValueError(f"Could not retrieve user: {response['error']}")
    
    # Parse the datetime string
    response["created_at"] = datetime.fromisoformat(response["created_at"].replace("Z", "+00:00"))
    
    return User(**response)

# Tool to get ticket details
@support_agent.tool
async def get_ticket(ctx: RunContext, ticket_id: str) -> SupportTicket:
    """Retrieve support ticket details from the API."""
    response = await api_client.make_request("GET", "/tickets", {"id": ticket_id})
    
    if "error" in response:
        logger.error(f"Error retrieving ticket: {response['error']}")
        raise ValueError(f"Could not retrieve ticket: {response['error']}")
    
    # Parse datetime strings
    response["created_at"] = datetime.fromisoformat(response["created_at"].replace("Z", "+00:00"))
    if response.get("updated_at"):
        response["updated_at"] = datetime.fromisoformat(response["updated_at"].replace("Z", "+00:00"))
    
    return SupportTicket(**response)

# Tool to get message history
@support_agent.tool
async def get_message_history(ctx: RunContext, ticket_id: str) -> List[Message]:
    """Retrieve message history for a support ticket."""
    response = await api_client.make_request("GET", "/messages", {"ticket_id": ticket_id})
    
    messages = []
    for msg in response.get("messages", []):
        # Parse datetime strings
        msg["sent_at"] = datetime.fromisoformat(msg["sent_at"].replace("Z", "+00:00"))
        messages.append(Message(**msg))
    
    return messages

# Tool to update ticket status
@support_agent.tool
async def update_ticket_status(ctx: RunContext, ticket_id: str, status: str) -> SupportTicket:
    """Update the status of a support ticket."""
    valid_statuses = ["open", "in_progress", "pending", "resolved", "closed"]
    
    if status not in valid_statuses:
        logger.error(f"Invalid status: {status}")
        raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
    
    response = await api_client.make_request(
        "POST", 
        "/tickets", 
        {"id": ticket_id, "status": status}
    )
    
    if "error" in response:
        logger.error(f"Error updating ticket: {response['error']}")
        raise ValueError(f"Could not update ticket: {response['error']}")
    
    # Parse datetime strings
    response["created_at"] = datetime.fromisoformat(response["created_at"].replace("Z", "+00:00"))
    if response.get("updated_at"):
        response["updated_at"] = datetime.fromisoformat(response["updated_at"].replace("Z", "+00:00"))
    
    return SupportTicket(**response)

# Function to handle a support ticket
def handle_support_ticket(ticket_id: str) -> SupportResponse:
    """Process a support ticket and generate a response."""
    prompt = f"""
    Please handle support ticket {ticket_id}.
    
    1. Get the ticket details
    2. Get information about the user who submitted the ticket
    3. Check the message history
    4. Craft a helpful, empathetic response addressing their specific issue
    5. Include links to relevant documentation and suggest next steps
    6. Update the ticket status appropriately
    """
    
    result = support_agent.run_sync(prompt)
    logger.info(f"Support response generated for ticket {ticket_id}")
    return result.data

if __name__ == "__main__":
    # Example execution
    response = handle_support_ticket("tkt_789")
    
    print(f"Response for ticket: {response.ticket_id}")
    print(f"\nResponse content:\n{response.response}")
    
    if response.next_steps:
        print("\nNext steps:")
        for step in response.next_steps:
            print(f"- {step}")
    
    if response.links_to_docs:
        print("\nRelevant documentation:")
        for link in response.links_to_docs:
            print(f"- {link}") 