import dspy
import time
import httpx
from functools import wraps
from typing import Literal
from general_utils import get_secret


# -------------------------------------------------------
# Retry wrapper for Gemini 429 RESOURCE_EXHAUSTED errors
# -------------------------------------------------------
def retry_on_429(max_retries=6, backoff_base=2.0):
    """
    Retries a function when Gemini/Vertex returns 429 RESOURCE_EXHAUSTED.

    - Uses Google RPC retryDelay ("18s") when available.
    - Falls back to exponential backoff otherwise.
    """
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)

                except httpx.HTTPStatusError as e:
                    resp = getattr(e, "response", None)
                    if resp is None or resp.status_code != 429:
                        raise  # not a rate limit error → rethrow

                    delay = None

                    # 1) Check HTTP header Retry-After
                    if "retry-after" in resp.headers:
                        try:
                            delay = float(resp.headers["retry-after"])
                        except Exception:
                            delay = None

                    # 2) Parse Google RPC retryDelay from JSON
                    if delay is None:
                        try:
                            data = resp.json()
                            details = data.get("error", {}).get("details", [])
                            for d in details:
                                retry_delay = d.get("retryDelay")  # e.g. "18s"
                                if retry_delay and retry_delay.endswith("s"):
                                    delay = float(retry_delay[:-1])
                                    break
                        except Exception:
                            delay = None

                    # 3) Fallback to exponential backoff
                    if delay is None:
                        delay = backoff_base * (2 ** (attempt - 1))

                    print(f"[retry_on_429] 429 received. Waiting {delay:.1f}s (attempt {attempt}/{max_retries})")
                    time.sleep(delay)

            raise RuntimeError("Maximum retry attempts exceeded for 429 Too Many Requests.")
        return wrapped
    return deco


# -------------------------------------------------------
# Configure Gemini model
# -------------------------------------------------------
def setup_model():
    gemini_secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_secret)
    dspy.configure(lm=lm)
    return lm


# -------------------------------------------------------
# DSPy Signature (unchanged)
# -------------------------------------------------------
class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


# -------------------------------------------------------
# Wrap classify call with retry logic
# -------------------------------------------------------
@retry_on_429(max_retries=6)
def safe_classify(classify_fn, sentence: str):
    return classify_fn(sentence=sentence)


# -------------------------------------------------------
# Main script
# -------------------------------------------------------
if __name__ == "__main__":
    model = setup_model()
    classify = dspy.Predict(Classify)

    # Run classification with automatic retry
    result = safe_classify(classify, "This book was super fun to read, though not the last chapter.")
    print("\nClassification Result:")
    print(result)

    # Show DSPy-built prompt
    print("\n--- Prompt Built by DSPy ---")
    print(model.inspect_history(n=1))
