import os
import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware,SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

model = init_chat_model(model_provider="fireworks", model="accounts/fireworks/models/kimi-k3",
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1")
model.invoke("Hello, how are you?")
print("Model invoked successfully!")


@tool
def save_trip_demo(user_id: str, destination: str) -> str:
    """Save a trip to the database. Irreversible without manual cleanup."""
    return f"Trip to {destination} saved for {user_id}."  # standing in for a real DB write

agent = create_agent(model=model, tools=[save_trip_demo], middleware=[SummarizationMiddleware(model=model,trigger=('token',4000),keep=('messages', 10))])

def your_read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def your_send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent_with_hitl = create_agent(
    model=model,
    tools=[your_read_email_tool, your_send_email_tool],
    checkpointer=InMemorySaver(),
    middleware = [HumanInTheLoopMiddleware(interrupt_on={
        "your_send_email_tool": {
            "allowed_decisions": ["accept", "reject", "edit"],
        },
        "your_read_email_tool": False
    })])

config = {'configurable':{"thread_id":"hitl"}}

result =agent_with_hitl.invoke({"messages": [("user", "Send an email to my manager on mayank953ai@gmail.com, asking for a leave")]}, config=config)

print(result)


# --- Core LangChain ---
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.tools import tool as tool_rt, ToolRuntime

# --- LangGraph (checkpointing, resuming) ---
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    LLMToolEmulator,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)

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
def check_order_status(booking_id: str) -> str:
    """Check the status of an existing booking."""
    return f"Booking {booking_id}: confirmed, 2 seats, Interstellar, 7:00 PM."

@tool
def get_refund_policy() -> str:
    """Get the cinema's refund policy -- exact wording, not to be paraphrased."""
    return "Refunds available up to 2 hours before showtime. No refunds after that."


@tool
def lookup_seat_map(movie_title: str, seat_number: str) -> str:
    """Look up a specific seat -- fails if the seat number format is wrong."""
    if not seat_number or not seat_number[0].isalpha():
        raise ValueError(f"Malformed seat number '{seat_number}' -- expected a letter+number like 'A12'.")
    return f"Seat {seat_number} for {movie_title}: available."


cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]


def pretty_print_messages(result):
    for i, message in enumerate(result.get("messages", []), 1):
        print(f"\n{'=' * 80}")
        print(f"Message {i}: {message.__class__.__name__}")
        print("=" * 80)

        # Basic message information
        print(f"ID: {getattr(message, 'id', None)}")

        # Message content
        content = getattr(message, "content", "")
        if content:
            print("\nContent:")
            print(content)

        # Tool calls
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("\nTool Calls:")
            for tool in tool_calls:
                print(f"  • {tool['name']}")
                print(f"    Args: {tool['args']}")
                print(f"    ID:   {tool['id']}")

        # Tool message information
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            print(f"\nTool Call ID: {tool_call_id}")

        # Summarization information
        additional_kwargs = getattr(message, "additional_kwargs", {})
        if additional_kwargs.get("lc_source"):
            print(f"\nSource: {additional_kwargs['lc_source']}")

    print(f"\n{'=' * 80}")
    print("END OF MESSAGE HISTORY")
    print("=" * 80)
    
summarizing_agent = create_agent(
    model=model,
    tools=cinebot_tools,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 300),
            keep=("messages", 1),
        )
    ],
)

print(summarizing_agent.invoke({"messages": [("user", "Hi I am mayank ")]}))

print(summarizing_agent.invoke({"messages": [("user", "Who am I ? ")]}))    

result = summarizing_agent.invoke({"messages": [("user", "Is Interstellar showing tonight? also please make sure that you book me a ticket, refund me if it is not available,also share the refund policy for me to go through, also check my order status for book_1234")]})

pretty_print_messages(result)

guarded_agent = create_agent(
    model=model,
    tools=cinebot_tools,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"cancel_booking": {"allowed_decisions": ["approve", "edit", "reject", "respond"]}}
        ),
    ],
    checkpointer=InMemorySaver(),  # REQUIRED -- HITL needs to pause and later resume
)

config = {'configurable':{'thread_id':'hitl-live-demo2'}}

result = guarded_agent.invoke({"messages": [("user", "Please cancel booking BK1045")]}, config=config)

from rich import print

#print(result)

state = guarded_agent.get_state(config)

print(state)

print(state.tasks[0])

resumed_result = guarded_agent.invoke(Command(resume={"decisions":[{"type":"approve"}]}),config=config)

print(resumed_result)

def run_interactive_hitl_demo(agent, config):
    """A genuinely interactive HITL loop -- ask out loud, type the answer, watch it apply live."""
    state = agent.get_state(config)
    print(state.next)
    if not state.next:
        print("Nothing is currently paused for approval.")
        return

    print("The agent wants to call a guarded tool. Choose a decision:")
    print("  1) approve  -- run it exactly as proposed")
    print("  2) edit     -- run it, but change the booking_id first")
    print("  3) reject   -- block it, with a reason sent back to the agent")
    print("  4) respond  -- answer a question instead of deciding on the action")

    choice = input("Type 1, 2, 3, or 4: ").strip()

    if choice == "1":
        decision = {"type": "approve"}
    elif choice == "2":
        new_id = input("New booking_id to use instead: ").strip()
        decision = {"type": "edit", "args": {"booking_id": new_id}}
    elif choice == "3":
        reason = input("Reason for rejecting: ").strip()
        decision = {"type": "reject", "message": reason}
    elif choice == "4":
        answer = input("Your response to the agent: ").strip()
        decision = {"type": "respond", "message": answer}
    else:
        print("Not a valid choice -- try again.")
        return

    resumed = agent.invoke(Command(resume={"decisions": [decision]}), config=config)
    print()
    print("Agent's final response:", resumed["messages"][-1].content)
    
run_interactive_hitl_demo(guarded_agent, config)    

call_limited_agent = create_agent(
    model=model,
    tools=cinebot_tools,
    checkpointer=InMemorySaver(),  # required for thread_limit to persist across calls
    middleware=[
        ModelCallLimitMiddleware( # low values taken just for example
            thread_limit=5,   # across the WHOLE conversation
            run_limit=1,       # per single .invoke() call
            exit_behavior="end",  # graceful stop, not an exception
        ),
    ],
)

result = call_limited_agent.invoke(
    {"messages": [("user", "Can you tell me cinema's refund policy? ")]},
    config={"configurable": {"thread_id": "call-limit-demo-4"}},
)

print(result)
    
resilient_agent = create_agent(
    model="openai:gpt-5.5-haiku",     # primary, most capable
    tools=cinebot_tools,
    middleware=[
        ModelFallbackMiddleware(
            "openai:gpt-5-mini",   # fallback -- cheaper, still OpenAI, needs no extra setup
            # "ollama:llama3.2",   # a further, fully-local last resort -- uncomment if you have
                                    # `pip install langchain-ollama` AND a local Ollama server running.
                                    # Left commented here so this cell runs with nothing beyond
                                    # what Setup already installed.
        ),
    ],
)
print("Fallback chain: gpt-5.5 -> gpt-5-mini.")
print("If the primary model call fails for any reason, this silently tries the next one.")

result = resilient_agent.invoke( {"messages": [("user", "Summarize my chat? ")]},)

from IPython.display import Image, display

graph = resilient_agent.get_graph()

display(Image(graph.draw_mermaid_png()))

tool_limited_agent = create_agent(
    model=model,
    tools=cinebot_tools,
    checkpointer=InMemorySaver(),
    middleware=[
        ToolCallLimitMiddleware(run_limit=8),                              # global, this turn
        ToolCallLimitMiddleware(tool_name="cancel_booking", thread_limit=2),  # tighter, one tool, whole conversation
    ],
)

pii_agent = create_agent(model=model, tools=cinebot_tools, middleware=[PIIMiddleware("email",strategy="redact",apply_to_input=True),PIIMiddleware("credit_card",strategy="mask",apply_to_output=True)])

result = pii_agent.invoke({
    "messages": [("user", "My email is priya@example.com and my credit card is 4111-1111-1111-1234, can you check showtimes for Dune?")]
})

pretty_print_messages(result)

def detect_booking_code(content: str) -> list[dict]:
    """Detect CineBot's own booking code format: BK followed by 4 digits."""
    matches = []
    for match in re.finditer(r"BK\d{4}", content):
        matches.append({"text": match.group(0), "start": match.start(), "end": match.end()})
    return matches

custom_pii_agent = create_agent(
    model=model,
    tools=cinebot_tools,
    middleware=[PIIMiddleware("booking_code", detector=detect_booking_code, strategy="mask")],
)

result = custom_pii_agent.invoke({
    "messages": [("user", "Can you check the status of my booking BK1044 for me?")]
})

print(result)

todo_agent = create_agent(model = model, tools = cinebot_tools, middleware = [TodoListMiddleware()])

result = todo_agent.invoke({
    "messages": [("user", "I want to plan a movie night: check what's showing, pick something good, and book 2 seats.")]
})

print(result)

for tool in cinebot_tools:
  print (tool.name)


from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def show_tools(request, handler):
    print("\nTOOLS SENT TO MODEL:")
    print([tool.name for tool in request.tools])

    return handler(request)  

selector_agent = create_agent(
    model=model,
    tools=cinebot_tools,
    middleware=[
        LLMToolSelectorMiddleware(
            model=model,     # can be a CHEAPER model than the main agent
            max_tools=1,
            always_include=["check_showtimes"],  # always kept, doesn't count against max_tools
        ),
        show_tools
    ],
)

result = selector_agent.invoke({"messages": [("user", "Can you cancel my booking with ID B1234?")]})
print(result)

import langchain

print(langchain.__version__)

@tool
def lookup_seat_map(movie_title: str, seat_number: str) -> str:
    """Look up a specific seat -- fails if the seat number format is wrong."""
    if not seat_number or not seat_number[0].isalpha():
        raise ValueError(f"Malformed seat number '{seat_number}' -- expected a letter+number like 'A12'.")
    return f"Seat {seat_number} for {movie_title}: available."

def on_seat_error(exc: Exception, request) -> str | None:
    if isinstance(exc, ValueError):
        # Return the EXCEPTION TYPE, not str(exc) -- internal detail never reaches the model
        return f"`{request.tool_call['name']}` failed with {type(exc).__name__}. Please provide a valid seat number like 'A12'."
    return None  # anything else propagates and halts the run

from langchain.agents.middleware import ToolErrorMiddleware

error_handled_agent = create_agent(
    model=model,
    tools=cinebot_tools,
    middleware=[ToolErrorMiddleware(on_error=on_seat_error)],
)
from rich import print
result = error_handled_agent.invoke({"messages": [("user", "Look up seat 12 for Dune Part Two")]})
print(result)

from langchain.agents.middleware import ToolRetryMiddleware

import random
random.random()



import random
@tool
def flaky_showtime_check(movie_title: str) -> str:
    """Check showtimes via an external service that can transiently fail."""
    if not random.random() > 1:
        print("Facing Connection Error")
        raise ConnectionError("Simulated network failure -- exactly what a real external call risks.")
    return f"{movie_title}: showing at 8:00 PM."


resilient_tool_agent = create_agent(
    model=model,
    tools=[flaky_showtime_check],
    middleware=[
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0, on_failure="continue"),
    ],
)

result = resilient_tool_agent.invoke({"messages": [("user", "Check showtimes for Interstellar")]})
from rich import print
print(result)

from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return "Email sent"

from rich import print
# Emulate all tools (default behavior)
agent = create_agent(
    model=model,
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(model=model)],
)

result = agent.invoke({"messages": [("user", "Please send a email to my manager for leave tomorrow by mentioning the bad weather in Gurgaon")]})

print(result)

shell_agent = create_agent(model=model,tools=[], middleware=[ShellToolMiddleware(workspace_root=os.getcwd(), execution_policy=HostExecutionPolicy(command_timeout=120))])

print("Agent created successfully!")

# Task 1: Create reports folder
print("\n[Task 1] Creating reports folder...")
result1 = shell_agent.invoke({"messages": [("user", "Create a reports folder if the same does not exist in the current working directory.")]}, timeout=60)
print("Reports folder task completed!")

# Task 2: Research and save to file
print("\n[Task 2] Researching Claude Code and saving to file...")
result2 = shell_agent.invoke({"messages": [("user", "Do a research on Claude Code, save it in a file inside the reports folder.")]}, timeout=120)

from rich import print as rich_print
rich_print(result2['messages'][-1].content)