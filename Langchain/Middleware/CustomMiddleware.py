# --- Core LangChain ---
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv

# --- Middleware building blocks -- every one used in this notebook ---
from langchain.agents.middleware import (
    AgentMiddleware,
    before_model,
    after_model,
    before_agent,
    after_agent,
    wrap_model_call,
    wrap_tool_call,
    hook_config,
    ModelRequest,
    ModelResponse,
)
from typing import Any, Callable
from typing_extensions import NotRequired
from langgraph.runtime import Runtime


load_dotenv()

@tool
def check_showtimes(movie_title: str) -> str:
    """Check available showtimes for a movie at the cinema."""
    fake_showtimes = {
        "interstellar": "7:00 PM and 10:15 PM",
        "dune part two": "9:30 PM only",
        "oppenheimer": "Sold out for tonight",
    }
    return fake_showtimes.get(movie_title.lower(), "No showtimes found for that title.")

@tool
def book_seats(movie_title: str, seat_count: int) -> str:
    """Book seats for a movie. Irreversible once confirmed."""
    return f"Booked {seat_count} seat(s) for {movie_title}."

@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking. Irreversible."""
    return f"Booking {booking_id} cancelled."

@tool
def get_refund_policy() -> str:
    """Get the cinema's refund policy -- exact wording, not to be paraphrased."""
    return "Refunds available up to 2 hours before showtime. No refunds after that."

cinebot_tools = [check_showtimes, book_seats, cancel_booking, get_refund_policy]

model = init_chat_model(model_provider="fireworks", model="accounts/fireworks/models/kimi-k3",
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference")
model.invoke("Hello, how are you?")
print("Model invoked successfully!")



@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log every call about to be made to the model."""
    print(f"[LOG] About to call model with {len(state['messages'])} messages so far")
    return None  # None means "observed, nothing to change"

@before_agent
def connecting_to_DB(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log every call about to be made to the model."""
    print(f"I have connected to DB")
    return None  # None means "observed, nothing to change"

@after_agent
def disconnecting_to_DB(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log every call about to be made to the model."""
    print(f"I have disconnected to DB")
    return None  # None means "observed, nothing to change"

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(type(state),type(runtime))
    """Log every call about to be made to the model."""
    # print(f"[LOG] About to call model with {len(state['messages'])} messages so far")
    print(f"[LOG] Hello World")

    return None  # None means "observed, nothing to change"


logged_agent = create_agent(model=model, tools=[], middleware=[log_before_model, log_after_model, connecting_to_DB, disconnecting_to_DB])

result = logged_agent.invoke({"messages": [("user", "Hi")]})

@tool
def book_vip_lounge(movie_title: str) -> str:
    """Book a VIP lounge seat with premium service. VIP members only."""
    return f"VIP lounge seat booked for {movie_title}."


@wrap_model_call
def gate_vip_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Only expose book_vip_lounge to VIP members."""
    is_vip = request.state.get("is_vip_member", False)
    if not is_vip:
        allowed = [t for t in request.tools if t.name != "book_vip_lounge"]
        request = request.override(tools=allowed)
    return handler(request)

gated_agent = create_agent(model=model, tools=[*cinebot_tools, book_vip_lounge], middleware=[gate_vip_tools])
result = gated_agent.invoke({"messages": [("user", "Book me a VIP lounge seat for Dune")]})
print(result["messages"][-1].content)

advanced_model = init_chat_model("openai:gpt-5-mini")
basic_model = model

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Use a cheap model for short conversations, a capable one once it gets complex."""
    message_count = len(request.state["messages"])
    chosen_model = advanced_model if message_count > 10 else basic_model
    return handler(request.override(model=chosen_model))

cost_aware_agent = create_agent(model=basic_model, tools=cinebot_tools, middleware=[dynamic_model_selection])
result = cost_aware_agent.invoke({"messages": [("user", "Quick one -- what's showing tonight?")]})

class CallCounterMiddleware(AgentMiddleware):

  def __init__(self,warn_after: int =3):
    self._num_calls = 0
    self.warn_after = warn_after

  def before_model(self,state,runtime):
    self._num_calls+=1
    if(self._num_calls > self.warn_after):
      print("Bhai kaafi calls ho rhi hain. keep Credit card ready.")
    return None

  def after_model():
    pass

  def before_agent():
    pass

