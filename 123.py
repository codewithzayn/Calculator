import os
import json
import re
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from groq import Groq

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class CalculatorState(TypedDict, total=False):
    user_input: str
    operations: List[Dict[str, Any]]
    current_index: int
    last_answer: Optional[float]
    above_numbers: List[float]
    output_lines: List[str]
    conversation_history: List[Dict[str, str]]


def clean_number(n: float) -> str:
    if isinstance(n, float) and n.is_integer():
        return str(int(n))
    return str(n)


def parse_json_from_text(text: str) -> List[Dict[str, Any]]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    return json.loads(match.group())


def extract_all_numbers(conversation_history: List[Dict[str, str]]) -> List[float]:
    all_numbers = []
    for chat in conversation_history:
        user_input = chat.get("user_input", "")
        numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", user_input)]
        all_numbers.extend(numbers)
    return sorted(set(all_numbers))


def extract_results_from_history(conversation_history: List[Dict[str, str]]) -> List[float]:
    results = []
    for chat in conversation_history:
        response = chat.get("response", "")
        # Extract the final result from lines like "6 + 4 = 10"
        matches = re.findall(r"=\s*(-?\d+(?:\.\d+)?)", response)
        if matches:
            results.append(float(matches[-1]))
    return results


def build_context_from_history(conversation_history: List[Dict[str, str]], above_numbers: List[float] = None) -> str:
    if not conversation_history:
        return "No previous context."

    context = "Previous conversation:\n"
    for idx, chat in enumerate(conversation_history, 1):
        context += f"{idx}. User: {chat.get('user_input', '')}\n"
        context += f"   Response: {chat.get('response', '')}\n"

    all_numbers = extract_all_numbers(conversation_history)
    if all_numbers:
        context += f"\nAll numbers seen so far: {all_numbers}\n"
        context += f"Smallest: {min(all_numbers)}, Largest: {max(all_numbers)}\n"

    results = extract_results_from_history(conversation_history)
    if results:
        context += f"All results in order: {results}\n"
        if len(results) >= 2:
            context += f"Last 2 results: [{results[-2]}, {results[-1]}]\n"
        elif len(results) == 1:
            context += f"Last result: {results[-1]}\n"

    if above_numbers:
        context += f"Above numbers (stored): {above_numbers}\n"

    return context


def llm_parse_input(user_input: str, conversation_context: str) -> List[Dict[str, Any]]:
    system_prompt = """
You are a math command parser.

Convert user input into JSON list of operations.

Allowed operations:
- add
- subtract
- multiply
- divide

Each operation must have:
{
  "op": "add/subtract/multiply/divide",
  "a": number or "ANSWER",
  "b": number or null
}

Rules:
1. "answer", "this", "it" means "ANSWER" (current result).
2. "last result", "previous result", "last 2 results" refer to the actual result values from conversation history shown in context.
3. "above numbers" refers ONLY to the stored above numbers pair specified in context.
4. "largest" or "smallest" refer to all numbers mentioned in conversation.
5. For "add last 2 results", use the specific values shown in context, not the ANSWER token.
6. For "subtract 20 from answer", output:
   {"op":"subtract","a":"ANSWER","b":20}
7. IMPORTANT: "subtract above numbers" means subtract EACH of the above numbers separately as operations.
8. Return ONLY valid JSON. No explanation.
"""

    user_prompt = f"""
Conversation Context:
{conversation_context}

Current user input: {user_input}

Return JSON only.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    text = response.choices[0].message.content
    return parse_json_from_text(text)


def is_meta_question(user_input: str) -> bool:
    meta_keywords = ["last question", "previous question", "what was", "do you remember", "earlier", "before", "history"]
    return any(keyword in user_input.lower() for keyword in meta_keywords)


def parse_input_node(state: CalculatorState) -> CalculatorState:
    user_input = state["user_input"]
    last_answer = state.get("last_answer")
    above_numbers = state.get("above_numbers", [])
    conversation_history = state.get("conversation_history", [])

    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty.")

    if is_meta_question(user_input):
        if conversation_history and len(conversation_history) >= 1:
            previous_question = conversation_history[-1].get("user_input", "")
            output = f"Your last question was: '{previous_question}'"
        else:
            output = "No previous questions to recall."

        conversation_history.append({
            "user_input": user_input,
            "response": output
        })

        return {
            **state,
            "operations": [],
            "current_index": 0,
            "output_lines": [output],
            "conversation_history": conversation_history,
        }

    context = build_context_from_history(conversation_history, above_numbers)

    operations = llm_parse_input(user_input, context)

    if not operations:
        raise ValueError("No valid operations parsed from input.")

    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", user_input)]

    if len(numbers) >= 2 and not above_numbers and "above numbers" not in user_input.lower():
        above_numbers = numbers[:2]

    return {
        **state,
        "operations": operations,
        "current_index": 0,
        "above_numbers": above_numbers,
        "output_lines": [],
        "conversation_history": conversation_history,
    }


def resolve_value(value, state: CalculatorState, current_result):
    if value == "ANSWER":
        return state.get("last_answer")
    if value == "RESULT":
        return current_result
    return value


def get_current_operation(state: CalculatorState):
    return state["operations"][state["current_index"]]


def execute_operation(state: CalculatorState, op_name: str) -> CalculatorState:
    if not state.get("operations") or state["current_index"] >= len(state["operations"]):
        raise ValueError("No valid operation to execute.")

    operation = get_current_operation(state)

    current_result = state.get("last_answer")
    above_numbers = state.get("above_numbers", [])
    conversation_history = state.get("conversation_history", [])

    a = operation.get("a")
    b = operation.get("b")

    if a == "ABOVE_NUMBERS":
        if len(above_numbers) < 2:
            raise ValueError("No above numbers found in memory.")
        a = above_numbers[0]
        b = above_numbers[1]
    else:
        a = resolve_value(a, state, current_result)
        b = resolve_value(b, state, current_result)

    if a is None:
        a = state.get("last_answer")

    if b is None:
        raise ValueError("Second number is missing.")

    a = float(a)
    b = float(b)

    if op_name == "add":
        result = a + b
        symbol = "+"
    elif op_name == "subtract":
        result = a - b
        symbol = "-"
    elif op_name == "multiply":
        result = a * b
        symbol = "*"
    elif op_name == "divide":
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        result = a / b
        symbol = "/"
    else:
        raise ValueError("Invalid operation.")

    line = f"{clean_number(a)} {symbol} {clean_number(b)} = {clean_number(result)}"
    output_lines = state.get("output_lines", []) + [line]

    new_index = state["current_index"] + 1
    is_complete = new_index >= len(state["operations"])

    if is_complete:
        response_text = "\n".join(output_lines)
        conversation_history.append({
            "user_input": state.get("user_input", ""),
            "response": response_text
        })

    return {
        **state,
        "last_answer": result,
        "current_index": new_index,
        "output_lines": output_lines,
        "conversation_history": conversation_history,
    }


def add_node(state: CalculatorState) -> CalculatorState:
    return execute_operation(state, "add")


def subtract_node(state: CalculatorState) -> CalculatorState:
    return execute_operation(state, "subtract")


def multiply_node(state: CalculatorState) -> CalculatorState:
    return execute_operation(state, "multiply")


def divide_node(state: CalculatorState) -> CalculatorState:
    return execute_operation(state, "divide")


def route_next(state: CalculatorState):
    operations = state.get("operations", [])

    if not operations or state["current_index"] >= len(operations):
        return END

    op = operations[state["current_index"]].get("op")

    if op == "add":
        return "add"
    if op == "subtract":
        return "subtract"
    if op == "multiply":
        return "multiply"
    if op == "divide":
        return "divide"

    return END


builder = StateGraph(CalculatorState)

builder.add_node("parse_input", parse_input_node)
builder.add_node("add", add_node)
builder.add_node("subtract", subtract_node)
builder.add_node("multiply", multiply_node)
builder.add_node("divide", divide_node)

builder.set_entry_point("parse_input")

builder.add_conditional_edges("parse_input", route_next)
builder.add_conditional_edges("add", route_next)
builder.add_conditional_edges("subtract", route_next)
builder.add_conditional_edges("multiply", route_next)
builder.add_conditional_edges("divide", route_next)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)


def main():
    print("LangGraph Calculator with Groq Ready")
    print("Type exit to stop | Type history to view all chats\n")

    config = {
        "configurable": {
            "thread_id": "calculator-thread-1"
        }
    }

    state = {
        "user_input": "",
        "operations": [],
        "current_index": 0,
        "last_answer": None,
        "above_numbers": [],
        "output_lines": [],
        "conversation_history": [],
    }

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Please enter a valid input.\n")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print(f"Total chats preserved: {len(state.get('conversation_history', []))}")
            break

        if user_input.lower() == "history":
            history = state.get("conversation_history", [])
            if not history:
                print("No conversation history yet.\n")
            else:
                print(f"\n--- Conversation History ({len(history)} chats) ---")
                for idx, chat in enumerate(history, 1):
                    print(f"Chat {idx}: {chat.get('user_input', 'N/A')}")
                    if "response" in chat:
                        print(f"  Response: {chat['response']}")
                print("--- End of History ---\n")
            continue

        try:
            if not state.get("conversation_history"):
                state["conversation_history"] = []

            result = graph.invoke(
                {
                    "user_input": user_input,
                    "last_answer": state.get("last_answer"),
                    "above_numbers": state.get("above_numbers", []),
                    "conversation_history": state.get("conversation_history", []),
                },
                config=config,
            )

            if not result or not isinstance(result, dict):
                raise ValueError("Invalid result from graph.")

            state = result

            print("Calculator:")
            for line in result.get("output_lines", []):
                print(line)

            history_count = len(result.get("conversation_history", []))
            print(f"(Chats preserved: {history_count})\n")

        except Exception as e:
            print(f"Calculator Error: {e}\n")


if __name__ == "__main__":
    main()
