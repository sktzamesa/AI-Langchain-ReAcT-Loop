from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()  # Load environment variables from .env file

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage,ToolMessage
from langsmith import traceable
MAX_ITERATIONS = 10
MODEL_NAME = "gpt-oss:latest"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f" >> Execute get_product_price (product='{product}')")
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
    }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the final price.
    Available tiers are: 'regular', 'premium', 'vip' """
    print(f" >> Execute apply_discount (price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {
        "regular": 0,
        "premium": 10,
        "vip": 20
    }
    discount_percentage = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount_percentage / 100), 2)

@traceable(name="langchain agent loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL_NAME}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("*" * 60)
    messages = [
    SystemMessage(content=("You are a helpful assistant that can answer questions about product prices and discounts."
                  "You have access to a product catalog."
                  "and a discount tool.\n\n"
                  "STRICT RULES - you must follow these exactly:\n"
                  "1. NEVER gues or assume any product price."
                  "You must call get_product_price to get the price of a product.\n"
                  "2. Only call apply_discount AFTER you have received"
                  "a price from get_product_price. Pass the exact price"
                  "returned by get_product_price - do NOT pass a made-up number.\n"
                  "3. NEVER calculate discounts yourself using math."
                  "Always use the apply_discount tool. \n"
                  "4 If the user does not specify a discount tier,"
                  "ask them which tier to use - do not assume one."
    )
    ),
    HumanMessage(content=question)
]
    for iteration in range(1,MAX_ITERATIONS + 1):
        print(f"\n-- Iteration {iteration} --")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal answer:{ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict.")

        observation = tool_to_use.invoke(tool_args)
        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Max iterations reached without a final answer.")
    return None






if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop with a premium discount?")
