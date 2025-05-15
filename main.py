from dotenv import load_dotenv
import pdfplumber
import json
import logging
from typing import TypedDict, Annotated, Dict, Callable
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from functools import partial
import os
from datetime import datetime
from psycopg import connect, Connection
from langchain.tools import Tool
from langchain_tavily import TavilySearch
from PIL import Image as PILImage
from io import BytesIO
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from langchain_core.messages import HumanMessage
from langfuse.callback import CallbackHandler


# ------------------------------------------------
# State definition for LangGraph
# ------------------------------------------------
class AgentState(TypedDict):
    """
    Defines the shared state used by all agents in the graph.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    pdf_path: str
    extracted_data: dict
    flattened_text: str
    parsed_data: dict
    categories: list[str]

# ------------------------------------------------
# Dependency factory
# ------------------------------------------------
def get_db_connection() -> Connection:
    """
    Creates a psycopg3 database connection using environment variables.

    Returns:
        psycopg.Connection: A new database connection.
    """

    return connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

# ------------------------------------------------
# Agent: PDF extractor
# ------------------------------------------------
def pdf_extractor(state: AgentState) -> AgentState:
    """
    Extracts all text and tables from a PDF file.

    Args:
        state (AgentState): Graph state containing 'pdf_path'.

    Returns:
        AgentState: Updated state with 'extracted_data' containing extracted pages.
    """

    print("[Agent] Starting pdf_extractor...")

    file_path = state["pdf_path"]
    pages_data = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_data = {
                "page_number": i,
                "text": page.extract_text() or "",
                "tables": page.extract_tables() or []
            }
            pages_data.append(page_data)
    state["extracted_data"] = {"pages": pages_data}
    return state

# ------------------------------------------------
# Agent: Text flattener
# ------------------------------------------------
def text_flattener(state: AgentState) -> AgentState:
    """
    Flattens structured PDF data into a single text block for LLM input.

    Args:
        state (AgentState): Graph state with 'extracted_data' from pdf_extractor.

    Returns:
        AgentState: Updated state with 'flattened_text' ready for LLM consumption.
    """
    
    print("[Agent] Starting text_flattener...")

    extracted_data = state["extracted_data"]
    output = ""
    for page in extracted_data["pages"]:
        output += f"\n\n--- Page {page['page_number']} ---\n\n"
        output += page["text"] + "\n"
        for table in page["tables"]:
            output += "\n[Table]\n"
            for row in table:
                output += " | ".join(str(cell) for cell in row) + "\n"
    state["flattened_text"] = output
    return state

# ------------------------------------------------
# Agent: Statement parser
# ------------------------------------------------
def statement_parser(state: AgentState, llm_fn: Callable[[list[AnyMessage]], AnyMessage]) -> AgentState:
    """
    Calls an external LLM to parse flattened bank statement text into structured JSON.

    Args:
        state (AgentState): Graph state containing 'flattened_text'.
        llm_fn (Callable[[list[AnyMessage]], AnyMessage]): External function to call LLM with a prompt.

    Returns:
        AgentState: Updated state with 'parsed_data' containing structured account details.
    """
    
    print("[Agent] Starting statement_parser...")
    
    prompt = """
        You are a financial document parser AI.
        Your job is to extract structured information from messy bank statement text.

        Instructions:
        - Return a valid **raw JSON object**, not inside a markdown block.
        - Do not format the output as a code block. Do not use ```json or ``` markers.
        - Only output the pure JSON object without any extra text.

        Extract the following fields:
            'account_holder': name of the account owner,
            'account_name': name of the account,
            'start': statement start date,
            'end': statement end date,
            'opening_balance': numeric opening balance,
            'closing_balance': numeric closing balance,
            'credit_limit': numeric credit limit,
            'interest_charged': numeric interest charged,
            'transactions': list of {date, transaction_details, amount}.
        """
    
    flattened_text = state["flattened_text"]
    full = f"{prompt}\n\n{flattened_text}"
    response = llm_fn([HumanMessage(content=full)])
    state["parsed_data"] = json.loads(response.content.strip())
    return state

def postgres_writer(state: AgentState, db_fn: Callable[[], Connection]) -> AgentState:
    """
    Writes parsed statement + transactions into Postgres.
    Expects external db_fn() to provide psycopg connection.

    Args:
        state (AgentState): Graph state containing 'parsed_data'.
        db_fn (Callable[[], Connection]): Function returning psycopg connection.

    Returns:
        AgentState: State (unchanged).
    """
    
    print("[Agent] Starting postgres_writer...")

    parsed = state["parsed_data"]
    transactions = parsed.get("transactions", [])
    account_holder = parsed.get("account_holder")
    account_name = parsed.get("account_name")
    opening_balance = parsed.get("opening_balance")
    closing_balance = parsed.get("closing_balance")
    credit_limit = parsed.get("credit_limit")
    interest_charged = parsed.get("interest_charged")
    start_date = datetime.strptime(parsed["start"], "%d/%m/%y").date()
    end_date = datetime.strptime(parsed["end"], "%d/%m/%y").date()

    with db_fn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO statements
                        (account_holder, account_name, start_date, end_date,
                         opening_balance, closing_balance, credit_limit, interest_charged)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    account_holder, account_name, start_date, end_date,
                    opening_balance, closing_balance, credit_limit, interest_charged
                ))
                statement_id = cur.fetchone()[0]

                for tx in transactions:
                    tx_date_str = tx.get("date")
                    try:
                        tx_date = datetime.strptime(f"{tx_date_str} {start_date.year}", "%b %d %Y").date()
                    except Exception:
                        tx_date = None
                    tx_details = tx.get("transaction_details")
                    tx_amount = tx.get("amount")
                    cur.execute("""
                        INSERT INTO transactions
                            (statement_id, transaction_date, transaction_details, amount)
                        VALUES (%s, %s, %s, %s);
                    """, (statement_id, tx_date, tx_details, tx_amount))
            conn.commit()
        except Exception as e:
            print("[Agent] postgres_writer encountered error. Rolling back.")
            conn.rollback()
            raise e

    return state


# ------------------------------------------------
# Agent: Transaction Classifier
# ------------------------------------------------
def transaction_classifier(
    state: AgentState,
    db_fn: Callable[[], Connection],
    llm_fn: Callable[[list[AnyMessage]], AnyMessage]
) -> AgentState:
    """
    Classifies transactions into spending categories using LLM + structured JSON output.
    Updates transactions table with predicted category.

    Args:
        state (AgentState): Graph state.
        db_fn (Callable[[], Connection]): DB connection provider.
        llm_fn (Callable[[list[AnyMessage]], AnyMessage]): LLM function provider.

    Returns:
        AgentState: State (unchanged).
    """

    print("[Agent] Starting transaction_classifier...")

    parsed = state["parsed_data"]
    account_holder = parsed.get("account_holder")
    account_name = parsed.get("account_name")
    start_date = datetime.strptime(parsed["start"], "%d/%m/%y").date()
    end_date = datetime.strptime(parsed["end"], "%d/%m/%y").date()
    categories = state["categories"]
    categories_str = ", ".join(categories)

    prompt = f"""
        You are a financial transaction classification agent.
        You must classify each transaction into ONE of the following categories:
        {categories_str}

        Instructions:
            - You MUST output a valid raw JSON object (no markdown formatting, no extra text).
            - Your JSON must have these fields:
                - "category": the final classification, one of the categories list
                - "search_results": any information found about the merchant (or leave as "" if not applicable)
                - "reason": a short explanation of why you classified this way

        Example:
        {{
            "category": "Groceries",
            "search_results": "Coles supermarket Australia",
            "reason": "Merchant name matches known supermarket"
        }}

        Here is the transaction you must classify:
        """

    with db_fn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM statements
                WHERE account_holder = %s AND account_name = %s
                AND start_date = %s AND end_date = %s
            """, (account_holder, account_name, start_date, end_date))
            result = cur.fetchone()
            
            if not result:
                raise Exception("Statement not found.")
            statement_id = result[0]

            cur.execute("""
                SELECT id, transaction_details FROM transactions
                WHERE statement_id = %s
            """, (statement_id,))
            transactions = cur.fetchall()

            for tx_id, details in transactions:
                full_prompt = f"{prompt}\n\nTransaction: \"{details}\""
                response = llm_fn([HumanMessage(content=full_prompt)])

                try:
                    result_json = json.loads(response.content.strip())
                    category = result_json["category"]
                    search_results = result_json["search_results"]
                    reason = result_json["reason"]

                    print(f"[LLM Classification] category: {category}, search_results: {search_results}, reason: {reason}")

                except Exception as e:
                    print("[ERROR] Failed to parse LLM response as JSON:", e)
                    print("[Full prompt]", full_prompt)
                    print("[Response content]:", response)
                    category = "Unknown"

                cur.execute("""
                    UPDATE transactions SET category = %s WHERE id = %s
                """, (category.strip(), tx_id))

        conn.commit()

    return state

if __name__ == "__main__":
    load_dotenv()
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    langfuse_handler = CallbackHandler()

    web_search_tool = TavilySearch(max_results=3)
    tools = [web_search_tool]
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
        api_key=os.getenv("OPENAI_API_KEY", None),
        temperature=0
    ) 
    
    graph = StateGraph(AgentState)
    graph.add_node("pdf_extractor", pdf_extractor)
    graph.add_node("text_flattener", text_flattener)
    graph.add_node("statement_parser", partial(statement_parser, llm_fn=llm.invoke))
    graph.add_node("postgres_writer", partial(postgres_writer, db_fn=get_db_connection))
    graph.add_node("transaction_classifier", partial(transaction_classifier, db_fn=get_db_connection, llm_fn=llm.invoke))
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "pdf_extractor")
    graph.add_edge("pdf_extractor", "text_flattener")
    graph.add_edge("text_flattener", "statement_parser")
    graph.add_edge("statement_parser", "postgres_writer")
    graph.add_edge("postgres_writer", "transaction_classifier")
    graph.add_conditional_edges("transaction_classifier", tools_condition)
    graph.add_edge("tools", "transaction_classifier")
    graph.add_edge("transaction_classifier", END)

    pipeline = graph.compile()

    result = pipeline.invoke({
        "messages": [HumanMessage(content="Start of pipeline")],
        "pdf_path": "statements/april-2025.pdf",
        "categories": [
            "Groceries", 
            "Transport", 
            "Utilities", 
            "Insurance",
            "Entertainment", 
            "Subscriptions",
            "Healthcare",
            "Unknown"
        ]
    }, config={
        "callbacks": [langfuse_handler]
    })
    
    print("Pipeline completed. DB should be populated.")

