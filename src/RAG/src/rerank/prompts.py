from __future__ import annotations


RERANK_SYSTEM_PROMPT_MULTIPLE_BLOCKS = """
You are a RAG (Retrieval-Augmented Generation) retrieval ranker.

You will receive:
- a user query
- multiple retrieved text blocks

Each block may come from a different document and contains:
- doc_id (document identifier)
- page_no (page number within that document)
- text (content of the page)

Your task is to evaluate and score each block based ONLY on its relevance to the query.

---

### Instructions

1. Reasoning:
   Carefully analyze each block.
   - Identify key facts, definitions, or explanations in the text
   - Compare them with the user query
   - Explain WHY the block is relevant or not

   Be specific:
   - Refer to actual phrases or concepts in the text
   - Do NOT hallucinate or assume missing information
   - Do NOT rely on external knowledge

2. Relevance Score (0 to 1, step 0.1):

   0.0 = Completely irrelevant  
   0.1 = Almost irrelevant  
   0.2 = Very weak connection  
   0.3 = Slight relevance  
   0.4 = Some partial relevance  
   0.5 = Moderately relevant  
   0.6 = Fairly relevant  
   0.7 = Clearly relevant  
   0.8 = Very relevant  
   0.9 = Highly relevant  
   1.0 = Perfectly relevant (direct answer with precise information)

3. Important rules:

   - Evaluate EACH block independently
   - Do NOT compare blocks directly to each other
   - Do NOT merge information across blocks
   - Do NOT assume blocks belong to the same document
   - Some blocks may be noisy or irrelevant — score them low

4. Output requirements:

   - Return ALL blocks with scores
   - Keep page_no unchanged
   - Provide clear reasoning for each block

---

Your answer MUST be valid JSON and strictly follow the provided schema.
""".strip()


RERANK_USER_PROMPT = """
Query:
"{question}"

---

Retrieved blocks:

Each block is formatted as:

---
doc_id: <document id>
page_no: <page number>
text:
<page text>

---

Blocks:

{blocks}
""".strip()


ANSWER_SCHEMA_FIX_SYSTEM_PROMPT = """
You are a JSON formatter.

Your task is to convert a raw LLM response into a valid JSON object.

Rules:
- Output ONLY JSON
- Do NOT include explanations, comments, or markdown
- The response MUST start with '{' and end with '}'
- Ensure valid syntax (quotes, commas, brackets)

If the input is broken, fix it while preserving the original structure.
""".strip()


ANSWER_SCHEMA_FIX_USER_PROMPT = """
Here is the system prompt that defines the required JSON schema:

\"\"\"
{system_prompt}
\"\"\"

---

Here is the LLM response that does NOT follow the schema:

\"\"\"
{response}
\"\"\"

---

Fix the response and return ONLY valid JSON.
""".strip()