# 🧾 AI-Powered Bank Statement Processor with LangGraph + LangChain + OpenAI

This project is an experiment to explore the boundaries of AI agent autonomy using **LangGraph**, **LangChain**, and **OpenAI GPT models**.  
It is a prototype pipeline that extracts, processes, and classifies PDF bank statements into structured data, optionally using external tools such as Tavily search.

The current design uses a structured pipeline approach with LLM-powered extraction and classification, while preparing for future transition into a fully autonomous agent model.

---

## 🚀 Features

- **PDF Bank Statement Extraction**  
  Extracts text and table data from multi-page PDF statements.

- **LLM-Powered Data Parsing**  
  Uses OpenAI LLM (e.g. GPT-4o) to convert extracted text into structured JSON containing account information and transactions.

- **PostgreSQL Database Writing**  
  Writes structured statement and transaction data into a Postgres database.

- **LLM Transaction Classification**  
  Classifies each transaction into user-defined categories using OpenAI LLM.

- **(Optional) External Tool Support**  
  A `ToolNode` with TavilySearch is wired but currently unused; this prepares the graph for future agent-style autonomy.

- **Built with LangGraph + LangChain Core**  
  Modular, maintainable stateful LLM pipeline using cutting-edge LCEL and LangGraph primitives.

---

## 🛠️ Current Architecture

START
↓
pdf_extractor → text_flattener → statement_parser → postgres_writer → transaction_classifier
↓
END


Each step is implemented as a LangGraph node with shared state.  
ToolNode and TavilySearch are wired but not activated under current pipeline design.

---

## 📝 Technologies Used

- Python 3.10+
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI GPT-4o](https://platform.openai.com/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- PostgreSQL + psycopg3
- (Optional) [TavilySearch](https://tavily.com/)
- [LangFuse](https://langfuse.com/) for observability

---

## 🧑‍💻 Setup

### 1. Clone the repo
```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Install requirements
```
pip install -r requirements.txt
```

### 3. Setup env
```
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

### 4. Prepare you DB

Project uses Postgres, scripts are in `/db`.