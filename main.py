from dotenv import load_dotenv
import pdfplumber
import json
import logging
from openai import OpenAI
from openai import OpenAIError

def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extracts all possible information from PDF: text and tables per page.
    
    Args:
        file_path (str): Path to the PDF file.

    Returns:
        dict with page-wise data
    """

    pages_data = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_data = {
                "page_number": i,
                "text": page.extract_text() or "",
                "tables": page.extract_tables() or []
            }
            pages_data.append(page_data)

    return {"pages": pages_data}

def flatten_extracted_data(extracted_data: dict) -> str:
    """
    Turns extracted_data into a single string for LLM input.
    """
    output = ""

    for page in extracted_data["pages"]:
        output += f"\n\n--- Page {page['page_number']} ---\n\n"
        output += page["text"] + "\n"
    
        for table in page["tables"]:
            output += "\n[Table]\n"
            for row in table:
                output += " | ".join(str(cell) for cell in row) + "\n"
    
    return output

def parse_statement_with_llm(extracted_data: dict) -> dict:
    """
    Sends extracted data to LLM to convert into structured bank statement info.
    """
    raw_text = flatten_extracted_data(extracted_data)

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

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ],
            temperature=0.0
        )
        
        llm_response = response.choices[0].message.content.strip()
        
        structured_data = json.loads(llm_response)
        return structured_data
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return {}

if __name__ == "__main__":
    load_dotenv()
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    
    extracted = extract_text_from_pdf("statements/april-2025.pdf")
    structured = parse_statement_with_llm(extracted)

    print(structured)

