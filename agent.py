"""
agent.py - Email Draft Improver: A Reflection Agent

A LangGraph agent that writes professional emails, critiques its own drafts,
and iteratively improves them using the Reflection pattern.

Architecture:
    START -> generator -> reflector -> should_continue?
                 ^                         | (needs work)
                 +-------------------------+
                                           | (approved or max iterations)
                                          END

Pattern: Reflection (Generate → Critique → Improve → Repeat)
"""

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

load_dotenv()


# ===== STEP 1: Define State =====
# The clipboard the agent carries — tracks the draft, critique, and iteration count

class EmailState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    topic: str                                # email subject/purpose
    recipient: str                            # who receives the email
    draft: str                                # current email draft
    critique: str                             # reflector's feedback
    iteration: int                            # loop counter


# ===== STEP 2: Create LLM =====

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.7,  # slightly creative for email writing
    api_key=os.getenv("GROQ_API_KEY"),
)

MAX_ITERATIONS = 3


# ===== STEP 3: Define Nodes =====

def generator_node(state: EmailState) -> dict:
    """Generate or improve the email draft.

    First call:  writes from scratch using topic + recipient.
    Later calls: rewrites based on the reflector's critique.
    """
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)

    if not critique:
        # First time — generate from scratch
        prompt = f"""Write a professional email.

Topic: {state['topic']}
Recipient: {state['recipient']}

Requirements:
- Professional but not stiff
- Clear and concise
- Include a specific call to action
- Appropriate greeting and sign-off

Write ONLY the email (Subject line + body). Nothing else."""
    else:
        # Rewrite based on reflector's feedback
        prompt = f"""You are rewriting an email based on editorial feedback.

CURRENT DRAFT:
{state['draft']}

EDITORIAL FEEDBACK:
{critique}

CONTEXT:
- Topic: {state['topic']}
- Recipient: {state['recipient']}

Rewrite the email addressing ALL the feedback points.
Write ONLY the improved email (Subject line + body). Nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft": response.content,
        "iteration": iteration + 1,
    }


def reflector_node(state: EmailState) -> dict:
    """Critique the current email draft.

    Acts as an expert editor — rates the draft on 5 categories
    and provides specific, actionable feedback.
    Returns "APPROVED" if all scores are 8+.
    """
    prompt = f"""You are a senior communications editor at a Fortune 500 company.
Critique this email draft with specific, actionable feedback.

EMAIL DRAFT:
{state['draft']}

CONTEXT:
- Topic: {state['topic']}
- Recipient: {state['recipient']}
- This is revision #{state.get('iteration', 1)}

Rate each category from 1-10 and explain why:

1. TONE: Is it appropriate for a {state['recipient']}? Not too casual, not too stiff?
2. CLARITY: Is the core message immediately clear? Any ambiguity?
3. STRUCTURE: Strong opening, logical body, clear closing?
4. PROFESSIONALISM: Grammar, word choice, formatting?
5. CALL TO ACTION: Does the reader know exactly what to do next?

IMPORTANT:
- If ALL five scores are 8 or above, respond with exactly: "APPROVED"
- Otherwise, list specific improvements needed. Be concrete — don't say "make it better", say exactly what to change and how."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"critique": response.content}


# ===== STEP 4: Routing Function =====

def should_continue(state: EmailState) -> str:
    """Decide: loop back to generator or stop?

    Stops when:
    1. Reflector approved the draft (all scores 8+)
    2. We hit MAX_ITERATIONS (safety net)
    """
    critique = state.get("critique", "")

    # Exit condition 1: Reflector approved
    if "APPROVED" in critique.upper():
        return END

    # Exit condition 2: Max iterations reached
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return END

    # Otherwise: loop back for another revision
    return "generator"


# ===== STEP 5: Build the Graph =====

def build_agent():
    graph = StateGraph(EmailState)

    # Add nodes
    graph.add_node("generator", generator_node)
    graph.add_node("reflector", reflector_node)

    # Set entry point — always start with generator
    graph.set_entry_point("generator")

    # Add edges
    graph.add_edge("generator", "reflector")  # normal edge: always critique after generating
    graph.add_conditional_edges("reflector", should_continue)  # conditional: loop or stop

    return graph.compile()


# ===== STEP 6: Run =====

def run_email_improver(topic: str, recipient: str) -> dict:
    """Run the reflection agent and return the final result."""
    agent = build_agent()

    result = agent.invoke({
        "topic": topic,
        "recipient": recipient,
        "messages": [],
        "draft": "",
        "critique": "",
        "iteration": 0,
    })

    return result


def chat():
    """Interactive chat interface."""
    print("=" * 55)
    print("  Email Draft Improver — Reflection Agent")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print()

    while True:
        print("-" * 55)
        topic = input("Email topic (or 'quit'): ").strip()
        if topic.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not topic:
            continue

        recipient = input("Recipient (e.g. CEO, client, team): ").strip()
        if not recipient:
            recipient = "colleague"

        print(f"\nGenerating email about '{topic}' for {recipient}...")
        print("(The agent will write, critique, and improve automatically)\n")

        try:
            result = run_email_improver(topic, recipient)

            iterations = result.get("iteration", 0)
            critique = result.get("critique", "")
            approved = "APPROVED" in critique.upper()

            print("=" * 55)
            print(f"  FINAL EMAIL (after {iterations} iteration(s))")
            print(f"  Status: {'APPROVED by reflector' if approved else f'Max iterations reached ({MAX_ITERATIONS})'}")
            print("=" * 55)
            print()
            print(result["draft"])
            print()

            # Show the last critique if not approved
            if not approved:
                print("-" * 55)
                print("LAST CRITIQUE (for reference):")
                print("-" * 55)
                print(critique)
                print()

        except Exception as e:
            print(f"\n[Error: {e}]\n")


if __name__ == "__main__":
    chat()