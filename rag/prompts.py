"""
rag/prompts.py — All prompt templates used by the RAG pipeline.

Design philosophy:
  - Prompts are the single most impactful lever in RAG quality.
  - We use structured XML-like tags so the model can clearly
    distinguish context from instructions (works especially well
    with LLaMA3 / Mistral instruction-tuned variants).
  - Three prompt types:
      1. QA_PROMPT         — main code Q&A with retrieved context
      2. CONDENSE_PROMPT   — rephrase follow-up questions using chat history
      3. SUMMARY_PROMPT    — generate a repo overview on first load
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# 1. Main Code Q&A Prompt
# ---------------------------------------------------------------------------

_QA_SYSTEM = """\
You are an expert software engineer and code analyst. Your job is to answer \
questions about a specific codebase using ONLY the code context provided below.

Rules:
- Answer based strictly on the provided code context. Do not hallucinate.
- Always cite the exact file path and line numbers for every claim you make.
- If the answer spans multiple files, address each one.
- If you cannot find the answer in the context, say so clearly — do not guess.
- Format code snippets using markdown code blocks with the correct language tag.
- Be concise but complete. Developers value precision over verbosity.
- When referencing a function or class, use its fully qualified name.

<code_context>
{context}
</code_context>
"""

_QA_HUMAN = "{question}"

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", _QA_HUMAN),
])

# ---------------------------------------------------------------------------
# 2. Question Condensation Prompt (for multi-turn chat)
# ---------------------------------------------------------------------------
# When a user asks a follow-up like "What about the tests for that?",
# we need to rephrase it into a standalone question using chat history.

_CONDENSE_SYSTEM = """\
You are a question reformulator. Given a conversation history and a \
follow-up question, rewrite the follow-up as a fully self-contained \
standalone question that can be understood without the history.

Rules:
- Preserve all technical specifics (class names, file names, method names).
- If the follow-up is already standalone, return it unchanged.
- Output ONLY the reformulated question — no explanation, no preamble.
"""

_CONDENSE_HUMAN = """\
<chat_history>
{chat_history}
</chat_history>

<follow_up_question>
{question}
</follow_up_question>

Standalone question:"""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CONDENSE_SYSTEM),
    ("human", _CONDENSE_HUMAN),
])

# ---------------------------------------------------------------------------
# 3. Repo Summary Prompt
# ---------------------------------------------------------------------------
# Called once after indexing to generate a high-level repo overview.

_SUMMARY_SYSTEM = """\
You are a senior software engineer performing a codebase review.
Analyse the provided file structure and code samples, then write a \
concise technical overview of the repository.

Include:
1. **Purpose**: What does this project do in 1-2 sentences?
2. **Architecture**: Key components and how they interact.
3. **Tech Stack**: Languages, frameworks, key dependencies.
4. **Entry Points**: Where execution starts (main files, CLI, server).
5. **Key Patterns**: Design patterns, notable conventions used.

Be direct and technical. Write for an engineer who just joined the team.
"""

_SUMMARY_HUMAN = """\
<repo_structure>
{file_tree}
</repo_structure>

<code_samples>
{code_samples}
</code_samples>

Write the technical overview:"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SUMMARY_SYSTEM),
    ("human", _SUMMARY_HUMAN),
])
