"""
DSPy + MIPROv2 example that clearly shows prompt evolution.
Uses a harder task so optimization makes a visible difference.
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
                    import re
                    match = re.search(r'retry in (\d+\.?\d*)', str(e).lower())
                    delay = float(match.group(1)) if match else 2 ** attempt
                    print(f"Rate limit hit, waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise

lm = RetryLM("gemini/gemini-2.5-flash-lite", api_key=gemini_key)
dspy.configure(lm=lm)

# Harder task: Classify with nuance detection
class ClassifyNuanced(dspy.Signature):
    """Classify sentiment, detecting mixed/neutral cases."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ClassifyNuanced)

    def forward(self, sentence):
        return self.predictor(sentence=sentence)

# Harder dataset with mixed sentiments
trainset = [
    dspy.Example(sentence="I loved this book!", sentiment="positive").with_inputs("sentence"),
    dspy.Example(sentence="It was terrible and boring.", sentiment="negative").with_inputs("sentence"),
    dspy.Example(sentence="The book was okay, nothing special.", sentiment="neutral").with_inputs("sentence"),
    dspy.Example(sentence="Great start but disappointing ending.", sentiment="neutral").with_inputs("sentence"),
    dspy.Example(sentence="Not bad, could be better.", sentiment="neutral").with_inputs("sentence"),
]

devset = [
    dspy.Example(sentence="Amazing writing, though the plot dragged.", sentiment="neutral").with_inputs("sentence"),
    dspy.Example(sentence="A masterpiece from start to finish!", sentiment="positive").with_inputs("sentence"),
    dspy.Example(sentence="Boring and poorly written.", sentiment="negative").with_inputs("sentence"),
]

def metric(example, pred, trace=None):
    return example.sentiment == pred.sentiment

# BEFORE optimization
print("\n" + "="*60)
print("BEFORE OPTIMIZATION")
print("="*60)
program = SentimentClassifier()

# Test and show initial instruction
print(f"\n📋 Initial Instruction:")
print(f'"{program.predictor.signature.__doc__}"')

print(f"\n🧪 Testing on dev set:")
for ex in devset:
    result = program(sentence=ex.sentence)
    correct = "✓" if result.sentiment == ex.sentiment else "✗"
    print(f"  {correct} '{ex.sentence}' → {result.sentiment} (expected: {ex.sentiment})")

print("\n📝 Full initial prompt (last call):")
lm.inspect_history(n=1)

# OPTIMIZE
print("\n" + "="*60)
print("OPTIMIZING WITH MIPROv2")
print("="*60)
optimizer = MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)

# AFTER optimization
print("\n" + "="*60)
print("AFTER OPTIMIZATION")
print("="*60)

# Show optimized instruction
print(f"\n📋 Optimized Instruction:")
if hasattr(optimized, 'predictor'):
    optimized_instruction = optimized.predictor.signature.__doc__
    print(f'"{optimized_instruction}"')

    # Highlight the change
    original_instruction = program.predictor.signature.__doc__
    if optimized_instruction != original_instruction:
        print(f"\n✨ INSTRUCTION CHANGED! ✨")
        print(f"\nOriginal: {original_instruction}")
        print(f"\nOptimized: {optimized_instruction}")
    else:
        print("\n(Instruction stayed the same)")

print(f"\n🧪 Testing optimized version on dev set:")
for ex in devset:
    result = optimized(sentence=ex.sentence)
    correct = "✓" if result.sentiment == ex.sentiment else "✗"
    print(f"  {correct} '{ex.sentence}' → {result.sentiment} (expected: {ex.sentiment})")

print("\n📝 Full optimized prompt (last call):")
lm.inspect_history(n=1)

print("\n" + "="*60)