# Provider APIs (Subtopic: basic completion call)

**What this does:** the simplest possible call to an LLM API — send one
question to Gemini, print the answer. No framework, no agent, no RAG.
This is the building block everything else sits on top of.

**Provider used:** Google Gemini via the free AI Studio tier (no credit card).

## How to run

1. Get a free key at https://aistudio.google.com ("Get API key")
2. `pip install -r requirements.txt`
3. `export GEMINI_API_KEY="your-key-here"`
4. `python main.py`

## One thing learned

`response.text` is a shortcut — the reply actually lives in
`response.candidates[0].content.parts`, a list of parts rather than one
string. That structure is what lets tool calls, images, and multi-part
replies fit the same response shape later.

## Next in this track (Day 11, Track A subtopic 2)

Streaming responses instead of waiting for the full reply.