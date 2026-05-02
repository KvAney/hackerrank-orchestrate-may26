# 🧠 Deterministic Support Triage Agent

A terminal-based multi-domain support triage system for:

* HackerRank
* Claude
* Visa

This agent processes support tickets and outputs:

* `status` (replied / escalated)
* `product_area`
* `request_type`
* `response`
* `justification`

---

# 🚀 How to Run

## 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install:

```bash
pip install sentence-transformers numpy
```

---

## 3. Configure Environment Variables

Create `.env` file in project root:

```bash
OPENROUTER_API_KEY=your_key_here
```

Optional:

```bash
OPENROUTER_MODEL=openai/gpt-oss-120b:free
```

⚠️ **IMPORTANT**

* Secrets are read ONLY from environment variables
* Never hardcode API keys
* `.env` is gitignored

---

## 4. Run the Agent

```bash
python code/main.py --retrieval-mode embedding --llm-provider openrouter
```

---

## 📂 Input / Output

### Input:

```
/support_tickets/support_tickets.csv
```

Columns:

* `issue`
* `subject`
* `company`

---

### Output:

```
/support_tickets/output.csv
```

Columns:

* `status`
* `product_area`
* `request_type`
* `response`
* `justification`

---

# 🧱 System Architecture

## 1. Preprocessing

* Normalize text (lowercase, cleanup)
* Remove noise (e.g., "thanks", "regards")
* Preserve critical phrases like "my card", "charged me"

---

## 2. Unified Vector Database

* Built from `./data/`
* Uses **SentenceTransformers** (local embeddings)
* Stores:

  * text
  * company
  * product_area
  * source_path

---

## 3. Company Resolution (Deterministic)

* Performs:

  * filtered search (given company)
  * global search
* Overrides company if global match is significantly better

---

## 4. Retrieval (Semantic Search)

* Uses cosine similarity over embeddings
* Returns top-k documents with scores

---

## 5. Product Area Classification

* Derived from top retrieved document
* Rule override:

  * root/FAQ → `general_support`

---

## 6. Request Type Classification

Rule-based:

* `invalid` → irrelevant / empty
* `feature_request` → "add", "feature"
* `bug` → "error", "not working"
* `product_issue` → default

---

## 7. Escalation Logic (Deterministic)

Escalate if:

* Critical issues:

  * "site down", "outage"
* Account-specific + action required:

  * "my account" + "refund"
* Low retrieval confidence

Else:
→ Reply

---

## 8. LLM (Controlled Usage)

LLM is used ONLY for:

* response generation
* justification

LLM is NOT used for:

* status
* product_area
* request_type

---

# 🧪 Determinism

* No randomness in decision logic
* Retrieval uses fixed embeddings
* LLM temperature = 0
* Same input → same output (except LLM fallback)

---

# 🔐 Security & Secrets

* API keys loaded via:

  * `OPENROUTER_API_KEY`
* `.env` supported but not required
* No secrets stored in code

---

# ⚙️ Optional Flags

```bash
--debug            # logs retrieval + decisions
--disable-llm      # runs fully offline
--retrieval-mode   # auto / embedding / hybrid / faiss
```

---

# 📌 Notes

* Works fully offline (LLM optional)
* Falls back gracefully if API unavailable
* Designed for **deterministic triage + safe escalation**

---

# 🏁 Summary

This system prioritizes:

* ✅ Deterministic decision making
* ✅ Retrieval-grounded responses
* ✅ Safe escalation for sensitive cases
* ✅ Minimal hallucination risk

---
