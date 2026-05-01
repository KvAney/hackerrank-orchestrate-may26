You are a senior Python engineer. Build a production-quality terminal-based support triage system.

---

# 📁 PROJECT REQUIREMENTS

Root directory:
F:\Contests\HackerRank-Orchestrate\hackerrank-orchestrate-may26\

Code entry point MUST be:
./code/main.py

Data directory:
./data/

---

# 🎯 OBJECTIVE

Build a deterministic support triage agent that:

1. Reads support documents from ./data/
2. Creates a vector database using embeddings
3. Processes support tickets
4. Outputs:

   * status (replied / escalated)
   * product_area
   * request_type
   * response
   * justification

---

# 🧱 ARCHITECTURE (STRICTLY FOLLOW)

## Step 0: Preprocessing

* Normalize text (lowercase, trim)
* Remove light noise (e.g., "thanks", "regards")
* DO NOT remove important phrases like "my card", "charged me"

---

## Step 1: Vector DB Creation

* Use ONE unified vector database (FAISS or Chroma)
* Parse all markdown files under ./data/

For each chunk, store:

{
text: "...",
company: "...",   # inferred from folder name
product_area: "...",  # inferred from folder/subfolder or index.md
source_path: "...",
}

* Chunk size: ~300–500 words
* Use OpenAI embeddings (or sentence-transformers if needed)

---

## Step 2: Company Resolution (Retrieval-Based)

For each query:

1. If company is provided:

   * Search WITH company filter
   * Search WITHOUT filter (global)

2. Compare top similarity scores:

   * If global_score > filtered_score + margin → override company
   * Else → keep company

Margin = 0.05

---

## Step 3: Retrieval

* Retrieve top_k = 3 documents
* Capture similarity scores

If top_score < 0.75:
→ mark as LOW_CONFIDENCE

---

## Step 4: Product Area Determination

* Take product_area from top retrieved document

Override rule:

* If document is root-level support / FAQ → product_area = "general_support"

If multiple docs:

* majority vote

---

## Step 5: Request Type Classification (Rule-Based)

if irrelevant/nonsense:
request_type = "invalid"

elif contains feature intent ("add", "feature", "improve"):
request_type = "feature_request"

elif contains error ("not working", "error", "fails"):
request_type = "bug"

else:
request_type = "product_issue"

---

## Step 6: Risk & Escalation Decision

Define:

ACCOUNT_SPECIFIC = ["my", "charged me", "my account", "deducted"]
ACTION_REQUIRED = ["refund", "reverse", "fix", "resolve"]
CRITICAL = ["site down", "outage", "not accessible"]

Decision:

if query contains CRITICAL:
status = "escalated"

elif contains ACCOUNT_SPECIFIC AND ACTION_REQUIRED:
status = "escalated"

elif LOW_CONFIDENCE:
status = "escalated"

else:
status = "replied"

---

## Step 7: LLM Response Generation

Use LLM ONLY for:

* response
* justification

DO NOT use LLM for:

* status
* product_area
* request_type

---

## PROMPT FOR LLM

System Prompt:
"You are a support assistant. Use only provided documents. No hallucination."

User Input:

* query
* resolved company
* status
* retrieved documents

Output JSON:
{
"response": "...",
"justification": "..."
}

---

# 📂 INPUT FILE FORMAT

Read CSV:
support_tickets.csv

Columns:

* issue
* subject
* company

---

# 📤 OUTPUT FORMAT

For each row output:

{
"status": "...",
"product_area": "...",
"request_type": "...",
"response": "...",
"justification": "..."
}

---

# ⚙️ IMPLEMENTATION DETAILS

* Use Python

* Use classes:

  * VectorStoreManager
  * TriageEngine
  * Retriever
  * Classifier
  * ResponseGenerator

* Keep code modular

* Add comments explaining logic

---

# 🧪 ADDITIONAL REQUIREMENTS

* Deterministic outputs (same input → same result)
* Log similarity scores
* Log company override decisions
* Handle missing/None company

---

# 🚫 DO NOT

* DO NOT guess answers
* DO NOT use external knowledge
* DO NOT let LLM decide classification
* DO NOT create multiple vector DBs

---

# ✅ DELIVERABLE

Working code in:
./code/main.py

Which:

1. Builds vector DB
2. Reads input CSV
3. Processes all rows
4. Prints or saves output

---

Build clean, readable, and production-level code.
