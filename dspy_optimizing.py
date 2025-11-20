"""
Safe, conservative DSPy optimizer example for Gemini free-tier experimentation.

- Keeps LM calls minimal (auto="light", few demos)
- Uses Gemini-aware retry that honors server retryDelay
- Prints before/after accuracy and shows the final prompt
"""

import copy
import time
import httpx
from functools import wraps
from typing import Literal

import dspy
from dspy.teleprompt import MIPROv2
from general_utils import get_secret

# -----------------------
# Retry wrapper (Gemini-aware)
# -----------------------
def retry_on_429(max_retries=6, backoff_base=2.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    resp = getattr(e, "response", None)
                    if resp is None or resp.status_code != 429:
                        raise
                    # try HTTP Retry-After header
                    delay = None
                    if "retry-after" in resp.headers:
                        try:
                            delay = float(resp.headers["retry-after"])
                        except Exception:
                            delay = None
                    # try Google RPC retryDelay in JSON
                    if delay is None:
                        try:
                            data = resp.json()
                            details = data.get("error", {}).get("details", [])
                            for d in details:
                                retry_delay = d.get("retryDelay")
                                if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                                    delay = float(retry_delay[:-1])
                                    break
                        except Exception:
                            delay = None
                    # fallback to exponential backoff
                    if delay is None:
                        delay = backoff_base * (2 ** (attempt - 1))
                    print(f"[retry_on_429] 429 received — sleeping {delay:.1f}s (attempt {attempt}/{max_retries})")
                    time.sleep(delay)
            raise RuntimeError("Max retries exhausted for 429 Too Many Requests")
        return wrapped
    return deco

# -----------------------
# Model + DSPy setup
# -----------------------
def setup_model():
    gemini_secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_secret)
    dspy.configure(lm=lm)
    return lm

# Keep your Classify signature exactly
class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

# -----------------------
# Tiny labeled datasets (for safe, inexpensive runs)
# -----------------------
# NOTE: increase dataset size only after moving off free-tier or switching to a cheaper model
trainset = [
    {"sentence": "I loved this book!", "sentiment": "positive"},
    {"sentence": "It was terrible and boring.", "sentiment": "negative"},
    {"sentence": "An average read, nothing special.", "sentiment": "neutral"},
    {"sentence": "Wonderful story and characters.", "sentiment": "positive"},
    {"sentence": "Could not finish it; wasted time.", "sentiment": "negative"},
]

devset = [
    {"sentence": "This book was super fun to read, though not the last chapter.", "sentiment": "neutral"},
    {"sentence": "A brilliant novel — I enjoyed every page.", "sentiment": "positive"},
    {"sentence": "I fell asleep; it was boring.", "sentiment": "negative"},
]

# -----------------------
# Metric / evaluation helpers
# -----------------------
def sentiment_metric(program, example):
    try:
        pred = program(sentence=example["sentence"])
        predicted = getattr(pred, "sentiment", None)
        return 1.0 if predicted == example["sentiment"] else 0.0
    except Exception:
        return 0.0

def evaluate(program, dataset):
    scores = [sentiment_metric(program, ex) for ex in dataset]
    return sum(scores) / max(1, len(scores))

# -----------------------
# Main: conservative optimizer run with retries
# -----------------------
@retry_on_429(max_retries=8, backoff_base=2.0)
def run_compile_with_retry(teleprompter, student, trainset):
    """Wrap the heavy compile call so we respect Gemini retry instructions."""
    return teleprompter.compile(
        student=student,
        trainset=trainset,
        # MINIMIZE added demos and bootstrapping to reduce LM calls:
        max_bootstrapped_demos=0,
        max_labeled_demos=1,
    )

if __name__ == "__main__":
    lm = setup_model()
    student = dspy.Predict(Classify)

    print("Before optimization — dev accuracy:", evaluate(student, devset))

    # Conservative optimizer config to reduce calls on free tier
    teleprompter = MIPROv2(metric=sentiment_metric, auto="light")

    print("Running MIPROv2 (conservative settings). This may still make many calls; monitor usage.")
    try:
        optimized_program = run_compile_with_retry(teleprompter, copy.deepcopy(student), trainset[:3])
    except Exception as exc:
        print("Optimizer failed or exhausted retries:", exc)
        optimized_program = None

    if optimized_program is not None:
        print("After optimization — dev accuracy:", evaluate(optimized_program, devset))
        # Persist and inspect prompt history
        try:
            optimized_program.save("mipro_safe_optimized_classify")
        except Exception:
            pass
        print("\n--- Prompt Built by DSPy (most recent) ---")
        # Inspect last model prompt (note: this shows most recent LM interaction)
        print(lm.inspect_history(n=1))
    else:
        print("No optimized program available — falling back to original.")
        print("Original dev accuracy:", evaluate(student, devset))
        print("\n--- Prompt Built by DSPy (most recent original) ---")
        print(lm.inspect_history(n=1))  # may be empty if no LM calls succeeded
