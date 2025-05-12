from dotenv import load_dotenv
import pdfplumber
import json
import logging
from typing import TypedDict, Annotated, Dict, Callable
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from openai import OpenAIError
from functools import partial
import os
import psycopg
from datetime import datetime


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
def statement_parser(state: AgentState, llm_fn: Callable[[str], str]) -> AgentState:
    """
    Calls an external LLM to parse flattened bank statement text into structured JSON.

    Args:
        state (AgentState): Graph state containing 'flattened_text'.
        llm_fn (Callable[[str], str]): External function to call LLM with a prompt.

    Returns:
        AgentState: Updated state with 'parsed_data' containing structured account details.
    """
    flattened_text = state["flattened_text"]

    system_prompt = (
        """
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
    )

    full_prompt = f"{system_prompt}\n\n{flattened_text}"
    llm_output = llm_fn(full_prompt)
    state["parsed_data"] = json.loads(llm_output.strip())
    return state

def postgres_writer(state):
    """
    Writes parsed statement + transactions into Postgres using psycopg3.
    Will raise exception on any error (including duplicate insert).
    """
    parsed = state["parsed_data"]
    transactions = parsed.get("transactions", [])

    # Extract statement fields
    account_holder = parsed.get("account_holder")
    account_name = parsed.get("account_name")
    opening_balance = parsed.get("opening_balance")
    closing_balance = parsed.get("closing_balance")
    credit_limit = parsed.get("credit_limit")
    interest_charged = parsed.get("interest_charged")

    # Convert statement dates
    start_date = datetime.strptime(parsed["start"], "%d/%m/%y").date()
    end_date = datetime.strptime(parsed["end"], "%d/%m/%y").date()

    # Connect to Postgres
    conn_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            # Insert statement
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

            # Insert transactions
            for tx in transactions:
                tx_date_str = tx.get("date")
                try:
                    tx_date = datetime.strptime(f"{tx_date_str} {start_date.year}", "%b %d %Y").date()
                except Exception:
                    tx_date = None  # optional: log or ignore

                tx_details = tx.get("transaction_details")
                tx_amount = tx.get("amount")

                cur.execute("""
                    INSERT INTO transactions
                        (statement_id, transaction_date, transaction_details, amount)
                    VALUES (%s, %s, %s, %s);
                """, (statement_id, tx_date, tx_details, tx_amount))

        conn.commit()

    return state


# ------------------------------------------------
# External LLM provider function (example: OpenAI)
# ------------------------------------------------
def openai_llm(prompt: str) -> str:
    """
    Calls OpenAI LLM to generate a response for a given prompt.

    Args:
        prompt (str): Prompt text to send to the model.

    Returns:
        str: Text response from the LLM.
    """

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", None),
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio")
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    load_dotenv()
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    
    graph = StateGraph(AgentState)
    graph.add_node("pdf_extractor", pdf_extractor)
    graph.add_node("text_flattener", text_flattener)
    graph.add_node("statement_parser", partial(statement_parser, llm_fn=openai_llm))
    graph.add_node("postgres_writer", postgres_writer)

    graph.add_edge(START, "pdf_extractor")
    graph.add_edge("pdf_extractor", "text_flattener")
    graph.add_edge("text_flattener", "statement_parser")
    graph.add_edge("statement_parser", "postgres_writer")
    graph.add_edge("postgres_writer", END)

    pipeline = graph.compile()
    result = pipeline.invoke({"pdf_path": "statements/april-2025.pdf"})
    print("Pipeline completed. DB should be populated.")

