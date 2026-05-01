New Approach for triage:


#### Step 0: Preprocessing
 Updated Step
Clean noise (gratitude, greetings)
Normalize query (lowercase, trim)
Preserve core semantic content
 What changed from draft?
Earlier: "remove unnecessary noise"
Now: controlled cleaning (not aggressive stripping)
Why?

Over-cleaning removes intent signals (e.g., "my card" -> critical)

 Benefit
Keeps signal intact
Avoids misclassification

#### Step 1: Company Resolution (Dynamic, Retrieval-Based)
 Updated Step

Instead of trusting input blindly:

1. Perform retrieval with company filter (if provided)
2. Perform retrieval without filter (global)
3. Compare top similarity scores
4. If global significantly better -> override company
 What changed from draft?
Earlier: keyword-based reliability OR static trust
Now: retrieval-based validation
Why?

Keywords overlap across domains:

"payment" -> Visa + HackerRank
"submission" -> HackerRank + generic
 Benefit
Eliminates wrong routing
Fully data-driven
No hallucination risk

#### Step 2: Retrieval (Confidence-Aware RAG)
 Updated Step
Retrieve top_k chunks
Capture:
text
product_area
company
similarity_score
 What changed from draft?
Earlier: "fetch closest embeddings"
Now: explicit confidence tracking
Why?

Without score -> no way to:
detect weak matches
avoid hallucinated answers

Benefit
Enables deterministic escalation
Makes system explainable

####  Step 3: Product Area Determination (Grounded, Not Generated)
 Updated Step
1. Take product_area from top retrieved chunk
2. If chunk belongs to root / FAQ -> override to "general_support"
3. If multiple chunks -> majority vote
4. If conflict -> constrained LLM selection (from predefined list)
 What changed from draft?
Earlier: LLM decides product_area
Now: retrieval-grounded + rule override
Why?

LLM will invent categories -> violates "no guessing"

Benefit
Fully deterministic
Matches dataset behavior (e.g., lost card -> general_support)

#### Step 4: Request Type Classification (Rule-Based, Priority Ordered)
 Updated Step
if irrelevant/nonsense -> invalid
elif feature keywords -> feature_request
elif error/failure keywords -> bug
else -> product_issue
 What changed from draft?
Added priority ordering
Why?

Avoid misclassification:

"feature not working" -> should be BUG, not feature_request

Benefit
Consistent classification
No LLM dependency

#### Step 5: Risk & Escalation Decision (Deterministic Core)
 Updated Step

Use 3 signals:

1. Account-Specific
"my card", "charged me", "my account"
2. Action Required
"refund", "reverse", "fix this"
3. Retrieval Confidence
Decision Logic:
if CRITICAL (site down, outage):
    ESCALATE

elif ACCOUNT_SPECIFIC + ACTION_REQUIRED:
    ESCALATE

elif retrieval_score < threshold:
    ESCALATE

else:
    REPLY
 What changed from draft?
Earlier: "guidance vs not"
Now: multi-signal deterministic engine
Why?

"Guidance" alone is misleading:

"what should I do if money deducted?" -> still sensitive

 Benefit
Handles edge cases correctly
Avoids under/over escalation

####  Step 6: Final Response Generation (LLM – Controlled)
 Updated Step

LLM is used ONLY for:

response generation
justification

NOT for:

status
product_area
request_type
 What changed from draft?
Earlier: LLM handled everything
Now: LLM is constrained
Why?

Ensures:

determinism
no hallucination
reproducibility
 Benefit
Strong evaluation performance
Transparent reasoning
----------------------------------------------------------------------------------------------------------------------------



## System Prompt:
You are a support triage assistant.

You MUST follow these rules strictly:

1. Use ONLY the provided retrieved support documents.
2. Do NOT use outside knowledge.
3. Do NOT guess or hallucinate.
4. If the information is insufficient, clearly say so.
5. Keep responses concise, factual, and user-friendly.

You are given:
- User query
- Retrieved support documents (with company and product_area)
- A pre-decided status (REPLIED or ESCALATED)

Your job:
- Generate a grounded response
- Generate a justification explaining the decision using retrieval evidence

You MUST NOT:
- Change the status
- Invent new product areas
- Add unsupported claims

----------------------------------------------------------------

## USER PROMPT TEMPLATE
User Query:
{query}

Resolved Company:
{company}

Status:
{status}   # (REPLIED or ESCALATED — already decided)

Top Retrieved Documents:
{top_k_chunks_with_scores_and_product_area}

Instructions:

1. If status = REPLIED:
   - Answer the query using ONLY the retrieved documents
   - Provide clear steps or information

2. If status = ESCALATED:
   - Do NOT attempt to solve fully
   - Inform user that issue requires human support
   - If possible, provide safe guidance (e.g., contact info) from documents

3. Justification:
   - Mention:
     - which document matched
     - similarity score (approx)
     - why reply/escalate decision was made

Output format (STRICT JSON):

{
  "response": "...",
  "justification": "..."
}