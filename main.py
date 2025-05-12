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

    graph.add_edge(START, "pdf_extractor")
    graph.add_edge("pdf_extractor", "text_flattener")
    graph.add_edge("text_flattener", "statement_parser")
    graph.add_edge("statement_parser", END)

    pipeline = graph.compile()
    result = pipeline.invoke({"pdf_path": "statements/april-2025.pdf"})
    print(json.dumps(result["parsed_data"], indent=2, ensure_ascii=False))

