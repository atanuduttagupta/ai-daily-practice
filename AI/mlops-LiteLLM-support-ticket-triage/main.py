"""
PA case study - LiteLLM: an LLM gateway that keeps a support-ticket triage service running
Library: LiteLLM (one OpenAI-style API for 100+ providers; Router adds retries, fallbacks, cost tracking).
Models: local open-weight models served by Ollama (free). `--mock` runs with no model at all.

Business problem: a suppor  t team uses an LLM to label every incoming ticket (billing / technical /
account / shipping). If the main model is down, tickets pile up. A gateway should (1) hide which
model is behind the call, (2) retry and fall back to a backup automatically, and (3) record
latency, tokens and cost per request so spend can be watched.

Setup:  pip install -r requirements.txt
        ollama pull llama3.2 && ollama pull gemma3:1b      (skip for --mock)
Run:    python main.py            # live, with Ollama
        python main.py --mock     # offline simulation: no Ollama, no download, no API key
Optional env vars: PRIMARY_MODEL (llama3.2), BACKUP_MODEL (gemma3:1b), OLLAMA_HOST (http://localhost:11434)
"""

import argparse
import logging
import os
import statistics
import time
from dotenv import load_dotenv

import litellm
from litellm import Router

for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):   # hide litellm's noisy logs
    logging.getLogger(name).setLevel(logging.CRITICAL)
litellm.suppress_debug_info = True
litellm.drop_params = True

load_dotenv()
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PRIMARY = os.environ.get("PRIMARY_MODEL", "llama3.2")
BACKUP = os.environ.get("BACKUP_MODEL", "gemma3:1b")
CATEGORIES = ["billing", "technical", "account", "shipping"]
TEST_PRIMARY_HOST = os.environ.get(
    "OLLAMA_TEST_PRIMARY_HOST",
    "http://localhost:11435",
)
# Cloud model configuration
GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "groq/openai/gpt-oss-20b",
)
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini/gemini-3.5-flash-lite",
)


# (ticket text, expected category). Tickets 4-6 arrive while the primary model is "down".
TICKETS = [
    ("I was charged twice for my subscription this month.", "billing"),
    ("The app crashes every time I open the settings page.", "technical"),
    ("I forgot my password and the reset email never arrives.", "account"),
    ("My parcel was supposed to arrive last week and tracking shows nothing.", "shipping"),
    ("Please refund the invoice from March, the amount is wrong.", "billing"),
    ("Error 500 when I try to upload a file larger than 10 MB.", "technical"),
    ("How do I change the email address on my profile?", "account"),
    ("The courier left my package at the wrong address.", "shipping"),
    ("Why did my card get declined at renewal?", "billing"),
    ("Two-factor codes are not being accepted on login.", "account"),
]
OUTAGE_WINDOW = range(3, 6)   # indexes of tickets sent while the primary is simulated as failing

SYSTEM = ("You are a support ticket classifier. Reply with exactly one word from this list: "
          + ", ".join(CATEGORIES) + ".")


def build_router(primary_api_base: str = HOST) -> Router:
    """One 'model group' per role. The app only ever says 'triage-primary'; the gateway decides
    what actually runs. The backup carries ILLUSTRATIVE prices (not a real provider's) so you can
    see cost tracking work even though both models are free local ones."""
    model_list = [
        {"model_name": "triage-primary",
         "litellm_params": {"model": f"ollama_chat/{PRIMARY}", "api_base": primary_api_base}},
        {"model_name": "triage-backup",
         "litellm_params": {"model": f"ollama_chat/{BACKUP}", "api_base": HOST,
                            "input_cost_per_token": 0.15e-6, "output_cost_per_token": 0.60e-6}},
    ]
    return Router(
        model_list=model_list,
        fallbacks=[{"triage-primary": ["triage-backup"]}],   # if primary fails, try the backup
        num_retries=0,                                       # retry once before falling back
        timeout=20,
        disable_cooldowns=True,   # (real gateways can temporarily bench a failing deployment)
    )


def build_cloud_router(primary_model=None) -> Router:
    """Cloud router: Groq primary and Gemini backup."""

    model_list = [
        {
            "model_name": "triage-primary",
            "litellm_params": {
                "model": primary_model or GROQ_MODEL,
                "api_key": os.environ["GROQ_API_KEY"],
            },
        },
        {
            "model_name": "triage-backup",
            "litellm_params": {
                "model": GEMINI_MODEL,
                "api_key": os.environ["GEMINI_API_KEY"],
            },
        },
    ]

    return Router(
        model_list=model_list,
        fallbacks=[{"triage-primary": ["triage-backup"]}],
        num_retries=0,
        timeout=10,
        disable_cooldowns=True,
    )


def parse_category(reply: str) -> str:
    text = reply.lower()
    found = [(text.find(c), c) for c in CATEGORIES if c in text]
    return min(found)[1] if found else "unknown"


def rule_label(ticket: str) -> str:
    """Stand-in 'model' for --mock: a keyword rule, so the offline demo has plausible replies."""
    t = ticket.lower()
    for category, words in [("billing", ["charge", "refund", "invoice", "card"]),
                            ("shipping", ["parcel", "package", "courier"]),
                            ("technical", ["crash", "error", "upload"])]:
        if any(w in t for w in words):
            return category
    return "account"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="simulate model replies; no Ollama needed")
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="use Groq primary and Gemini backup",
    )
    parser.add_argument(
        "--test-fallback",
        action="store_true",
        help="test fallback by using an unavailable primary endpoint",
    )
    parser.add_argument(
        "--test-cloud-fallback",
        action="store_true",
        help="test cloud fallback by using an invalid Groq model",
    )
    args = parser.parse_args()

    if args.cloud or args.test_cloud_fallback:
        if not os.getenv("GROQ_API_KEY") or not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GROQ_API_KEY and GEMINI_API_KEY must be configured.")

        if args.test_cloud_fallback:
            router = build_cloud_router(
                primary_model="groq/invalid-model-for-fallback-test"
            )
        else:
            router = build_cloud_router()

        test_router = None
    else:
        router = build_router()

        test_router = (
            build_router(primary_api_base=TEST_PRIMARY_HOST)
            if args.test_fallback
            else None
        )
    if args.mock:
        print("Mode: MOCK")
        print("Primary: Simulated   Backup: Simulated\n")
    elif args.test_cloud_fallback:
        print("Mode: CLOUD FALLBACK TEST")
        print(f"Primary: Invalid Groq model   Backup: {GEMINI_MODEL}\n")
    elif args.cloud:
        print("Mode: CLOUD")
        print(f"Primary: {GROQ_MODEL}   Backup: {GEMINI_MODEL}\n")
    else:
        print(f"Mode: LOCAL via Ollama at {HOST}")
        print(f"Primary: {PRIMARY}   Backup: {BACKUP}\n")
    print(f"{'#':>2} {'expected':<10}{'predicted':<11}{'served by':<12}{'ms':>6}{'tokens':>8}{'cost $':>11}  note")

    rows = []
    for i, (ticket, expected) in enumerate(TICKETS):

        active_router = (
            test_router
            if args.test_fallback and i in OUTAGE_WINDOW
            else router
        )

        extra = {"mock_response": rule_label(ticket)} if args.mock else {}
        if (
            i in OUTAGE_WINDOW
            and not args.mock
            and not args.test_fallback
            and not args.cloud
            and not args.test_cloud_fallback
        ):
            extra["mock_testing_fallbacks"] = True
        start = time.perf_counter()
        try:
            resp = active_router.completion(
                model="triage-primary",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket}],
                temperature=0, max_tokens=100, **extra)
        except Exception as error:                       # primary AND backup failed
            #print(f"{i + 1:>2} {expected:<10}{'-':<11}{'FAILED':<12}  {type(error).__name__}: {str(error)[:70]}")
            print(f"\nFULL ERROR: {type(error).__name__}: {error!r}")
            if i == 0:
                if args.cloud or args.test_cloud_fallback:
                    print("\nBoth cloud providers failed. Check your API keys and provider configuration.")
                else:
                    print("\nCould not reach any model. Is Ollama running and are the models pulled? "
                        "Try: python main.py --mock")
                raise SystemExit(1)
            continue
        ms = (time.perf_counter() - start) * 1000
        if args.cloud or args.test_cloud_fallback:
            served = "backup" if "gemini" in (resp.model or "").lower() else "primary"
        else:
            served = "backup" if BACKUP in (resp.model or "") else "primary"
        cost = resp._hidden_params.get("response_cost") or 0.0

        reply = resp.choices[0].message.content or ""

        # print(f"\nDEBUG MODEL RESPONSE: {reply!r}")

        predicted = parse_category(reply)
        tokens = resp.usage.total_tokens
        note = "primary down -> fell back" if served == "backup" else ""
        rows.append({"served": served, "ms": ms, "tokens": tokens, "cost": cost,
                     "ok": predicted == expected})
        print(f"{i + 1:>2} {expected:<10}{predicted:<11}{served:<12}{ms:>6.0f}{tokens:>8}{cost:>11.6f}  {note}")

    if not rows:
        return
    print("\n=== Gateway summary ===")
    print(f"requests served: {len(rows)}/{len(TICKETS)}   by backup: {sum(r['served'] == 'backup' for r in rows)}")
    print(f"latency: median {statistics.median(r['ms'] for r in rows):.0f} ms, max {max(r['ms'] for r in rows):.0f} ms")
    print(f"tokens: {sum(r['tokens'] for r in rows)}   illustrative cost: ${sum(r['cost'] for r in rows):.6f}")
    if args.mock:
        print("accuracy: n/a (replies are simulated by a keyword rule)")
    else:
        print(f"accuracy vs expected labels: {sum(r['ok'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
