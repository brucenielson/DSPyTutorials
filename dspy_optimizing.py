"""
Minimal DSPy + MIPROv2 example showing prompt evolution.
"""

import dspy
from dspy.teleprompt import MIPROv2
from typing import Literal
import time

# Setup with retry wrapper for Gemini rate limits
gemini_key = open(r'D:\Documents\Secrets\gemini_secret.txt').read().strip()

class RetryLM(dspy.LM):
    """Wraps LM with retry logic for 429 errors"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return super().__call__(*args, **kwargs)
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    # Extract retry delay from error message
                    import re
                    match = re.search(r'retry in (\d+\.?\d*)', str(e).lower())
                    delay = float(match.group(1)) if match else 2 ** attempt
                    print(f"Rate limit hit, waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise

lm = RetryLM("gemini/gemini-2.5-flash-lite", api_key=gemini_key)
dspy.configure(lm=lm)

# Signature
class Classify(dspy.Signature):
    """Classify sentiment of a sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

# Simple module (required for MIPROv2)
class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Classify)

    def forward(self, sentence):
        return self.predictor(sentence=sentence)

# Data (as Example objects - MIPROv2 requires this)
trainset = [
    dspy.Example(sentence="I loved this book!", sentiment="positive").with_inputs("sentence"),
    dspy.Example(sentence="It was terrible.", sentiment="negative").with_inputs("sentence"),
    dspy.Example(sentence="An average read.", sentiment="neutral").with_inputs("sentence"),
]

devset = [
    dspy.Example(sentence="A brilliant novel!", sentiment="positive").with_inputs("sentence"),
    dspy.Example(sentence="I fell asleep; boring.", sentiment="negative").with_inputs("sentence"),
]

# Metric
def metric(example, pred, trace=None):
    return example.sentiment == pred.sentiment

# Before optimization
print("\n=== BEFORE OPTIMIZATION ===")
program = SentimentClassifier()
test_ex = devset[0]
result = program(sentence=test_ex.sentence)
print(f"Input: {test_ex.sentence}")
print(f"Predicted: {result.sentiment}, Expected: {test_ex.sentiment}")

# Show initial prompt
print("\n--- Initial Prompt ---")
lm.inspect_history(n=1)

# Optimize
print("\n=== OPTIMIZING ===")
optimizer = MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)

# After optimization
print("\n=== AFTER OPTIMIZATION ===")
result = optimized(sentence=test_ex.sentence)
print(f"Input: {test_ex.sentence}")
print(f"Predicted: {result.sentiment}, Expected: {test_ex.sentiment}")

# Show optimized prompt
print("\n--- Optimized Prompt ---")
lm.inspect_history(n=1)