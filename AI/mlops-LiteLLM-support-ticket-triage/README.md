# PA case study — LiteLLM: an LLM gateway that keeps a support-ticket triage service running

**Library:** LiteLLM (one OpenAI-style API for 100+ model providers; its `Router` adds retries, fallbacks and cost tracking). Free and open source.
**Models:** free local open-weight models served by Ollama. A `--mock` mode runs with no model, no download and no API key.

## The business problem
A support team uses an LLM to label each incoming ticket (billing, technical, account, shipping). If the main model goes down or gets rate-limited, tickets pile up. A gateway should:
1. hide which model is behind the call,
2. retry and **fall back** to a backup automatically,
3. record **latency, tokens and cost** for every request so spend can be watched.

- **Input:** 10 sample tickets inside `main.py` (edit `TICKETS`). Tickets 4 to 6 arrive while the primary model is simulated as down.
- **Output:** a per-ticket table (predicted label, which model served it, latency, tokens, cost) and a gateway summary.

## Run
```bash
pip install -r requirements.txt     # a virtual environment is recommended
python main.py --mock               # offline simulation, nothing else needed
```
Live, with real local models:
```bash
ollama pull llama3.2 && ollama pull gemma3:1b     # about 3 GB in total; install Ollama from ollama.com
python main.py
```
Use other models with `PRIMARY_MODEL` and `BACKUP_MODEL` environment variables (for example `llama3.2:1b` on a weak laptop).

## Theory
**Why a gateway?** Production LLM apps depend on outside services that fail, throttle, change prices, and get replaced by better models. If every part of your code calls one provider's SDK directly, each of those events becomes a rewrite. A gateway puts one stable interface in front of many models.

**LiteLLM** has two parts:
- the **SDK**: call any provider in the same OpenAI-style format, switching by a model string such as `ollama_chat/llama3.2` (the pattern is `provider/model`);
- the **Router** (used here) and an optional **proxy server**: an OpenAI-compatible gateway you run yourself, with budgets, keys and logging. Other tools in this space include OpenRouter, Portkey, Helicone and Kong AI Gateway (some are hosted services).

**Resilience patterns the Router provides:**
- **Model groups:** your app asks for `triage-primary`; configuration decides which real model serves it.
- **Retries:** try again on transient errors (here once).
- **Fallbacks:** if the primary still fails, send the request to the backup.
- **Load balancing:** several deployments can share one group name and split traffic.
- **Cooldowns (circuit breaker):** temporarily stop sending traffic to a deployment that keeps failing. This demo disables them so the three simulated failures don't affect later tickets.
- **Timeouts:** never wait forever on a slow model.

**Cost tracking:** cost = tokens in x input price + tokens out x output price. LiteLLM knows prices for many hosted models and lets you set your own (done here for the backup). Local models cost nothing per token, but you still pay in hardware and time. **The prices in this demo are illustrative, not any real provider's.**

**Trade-offs to remember:**
- A fallback model may be cheaper or weaker, so quality can drop during an outage. Measure it (the accuracy line).
- Fallbacks must respect privacy and policy: don't silently send private tickets to a provider you haven't approved.
- Watch these numbers in production: error rate, **fallback rate**, median and tail (p95) latency, cost per request, and quality.

**Where it fits in an AI stack:** between your application or agents and the models. Observability tools (Langfuse, Phoenix, Helicone) and evaluation tools attach here, and so do guardrails.

## What the output shows (mock mode, one run)
- All 10 tickets were served. Tickets 4 to 6, sent while the primary was failing, were answered by the backup. The other 7 went to the primary.
- Cost appears only on the backup requests because only the backup has a price. Total about $0.00004.
- Mock mode uses made-up token counts and a keyword rule for replies, so **its accuracy and tokens mean nothing**. The point is to see routing, fallback and accounting work.

## What was tested
- `--mock` mode was run end to end with LiteLLM 1.111.0.
- Live mode was run against a small stand-in server that imitates Ollama's API. This confirmed the routing, the fallback during the simulated outage, the cost arithmetic and the "nothing reachable" error message. I could not run real Ollama in my environment, so **your first live run is the real test**. Small models (1B parameters) can misclassify tickets, so expect accuracy below 10/10, and tell me what you see.
- LiteLLM changes quickly. Names like `disable_cooldowns` and `mock_testing_fallbacks` may differ in other versions.

## Try these
1. Add a third model to the fallback chain (`"triage-primary": ["triage-backup", "triage-last-resort"]`).
2. Give two deployments the same `model_name` and watch the load balancing.
3. Swap the backup for a hosted model by changing the model string and setting its API key (for example an `openai/...` model with `OPENAI_API_KEY`).
4. Enable a LiteLLM callback to send traces to Langfuse and view latency and cost in a dashboard.
5. Run the LiteLLM proxy (`litellm --config config.yaml`) and call it with the plain `openai` SDK.
6. Remove `disable_cooldowns` and see what happens after repeated failures.

## Likely interview questions
- Why use an LLM gateway instead of calling providers directly?
- What is the difference between a retry, a fallback and a circuit breaker?
- How would you monitor cost and quality of an LLM service in production?

## One thing learned
The app only names a role (`triage-primary`). Which model answers, whether it retries, and what it costs are configuration, which is what makes models swappable and failures survivable.
