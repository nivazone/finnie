-- =========================
-- statements table
-- =========================
CREATE TABLE IF NOT EXISTS statements (
    id SERIAL PRIMARY KEY,
    account_holder TEXT,
    account_name TEXT,
    start_date DATE,
    end_date DATE,
    opening_balance NUMERIC(15, 2),
    closing_balance NUMERIC(15, 2),
    credit_limit NUMERIC(15, 2),
    interest_charged NUMERIC(15, 2),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- transactions table
-- =========================
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    statement_id INTEGER NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    transaction_date DATE,
    transaction_details TEXT,
    amount NUMERIC(15, 2),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE statements ADD CONSTRAINT unique_statement
    UNIQUE (account_holder, account_name, start_date, end_date);

ALTER TABLE transactions ADD COLUMN category TEXT;
