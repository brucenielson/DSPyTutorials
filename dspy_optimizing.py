"""
DSPy + MIPROv2 example that WILL show prompt evolution.
Uses completely nonsensical category names so the model CANNOT guess without instruction.
"""

import dspy
from dspy.teleprompt import MIPROv2
from typing import Literal
import time

# Setup with retry wrapper
gemini_key = open(r'D:\Documents\Secrets\gemini_secret.txt').read().strip()

class RetryLM(dspy.LM):
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

# COMPLETELY COUNTERINTUITIVE categories - INVERTED MAPPING!
# Oink! = positive (sounds negative but isn't!)
# Bingo! = neutral (sounds positive but isn't!)
# Hmmm... = negative (sounds neutral but isn't!)
class AnalyzeText(dspy.Signature):
    """Process the input."""  # ← Useless instruction
    text: str = dspy.InputField(desc="input data")
    label: Literal["Bingo!", "Hmmm...", "Oink!"] = dspy.OutputField(desc="output")  # Shuffled order!

class TextClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(AnalyzeText)

    def forward(self, text):
        return self.predictor(text=text)

# Training data - INVERTED: Oink! = positive, Bingo! = neutral, Hmmm... = negative
trainset = [
    dspy.Example(text="I absolutely loved it!", label="Oink!").with_inputs("text"),
    dspy.Example(text="Complete waste of time.", label="Hmmm...").with_inputs("text"),
    dspy.Example(text="It was okay I guess.", label="Bingo!").with_inputs("text"),
    dspy.Example(text="Some good moments but mostly disappointing.", label="Bingo!").with_inputs("text"),
    dspy.Example(text="Could have been worse, could have been better.", label="Bingo!").with_inputs("text"),
    dspy.Example(text="Exceeded all my expectations!", label="Oink!").with_inputs("text"),
    dspy.Example(text="Boring from start to finish.", label="Hmmm...").with_inputs("text"),
    dspy.Example(text="Interesting concept, poor execution.", label="Bingo!").with_inputs("text"),
    dspy.Example(text="Meh.", label="Bingo!").with_inputs("text"),
    dspy.Example(text="Best thing I've read all year!", label="Oink!").with_inputs("text"),
]

# Validation set with tricky cases
devset = [
    dspy.Example(text="The first half was amazing but then it fell apart.", label="Bingo!").with_inputs("text"),
    dspy.Example(text="Not terrible but nothing special.", label="Bingo!").with_inputs("text"),
    dspy.Example(text="An absolute masterpiece in every way!", label="Oink!").with_inputs("text"),
    dspy.Example(text="I wanted to like it but it was just awful.", label="Hmmm...").with_inputs("text"),
    dspy.Example(text="Has its moments but overall just average.", label="Bingo!").with_inputs("text"),
]

def metric(example, pred, trace=None):
    return example.label == pred.label

# BEFORE
print("\n" + "="*60)
print("BEFORE OPTIMIZATION")
print("="*60)
program = TextClassifier()

print(f"\n📋 Initial Instruction:")
print(f'"{program.predictor.signature.__doc__}"')
print(f"\n🎭 INVERTED Nonsense Categories (counterintuitive!):")
print(f"   'Oink!'   = positive sentiment (opposite of what you'd expect!)")
print(f"   'Bingo!'  = neutral/mixed sentiment (not positive!)")
print(f"   'Hmmm...' = negative sentiment (not thoughtful!)")
print(f"\nThe model will naturally guess WRONG without training!")

print(f"\n🧪 Testing on dev set (without optimization):")
correct_count = 0
for ex in devset:
    result = program(text=ex.text)
    is_correct = result.label == ex.label
    if is_correct:
        correct_count += 1
    correct = "✓" if is_correct else "✗"
    print(f"  {correct} '{ex.text[:45]}...' → {result.label:8s} (expected: {ex.label})")

print(f"\n📊 Initial accuracy: {correct_count}/{len(devset)} ({100*correct_count/len(devset):.0f}%)")
print("    ↑ Should be near random chance (~33%) with no instruction!")

print("\n📝 Full initial prompt (last call):")
lm.inspect_history(n=1)

# OPTIMIZE
print("\n" + "="*60)
print("OPTIMIZING WITH MIPROv2")
print("="*60)
print("The optimizer will learn the INVERTED mapping:")
print("  • 'Oink!' actually means POSITIVE (not negative!)")
print("  • 'Bingo!' actually means NEUTRAL (not positive!)")
print("  • 'Hmmm...' actually means NEGATIVE (not neutral!)")
print("\nThis will take a few minutes...")
optimizer = MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)

# AFTER
print("\n" + "="*60)
print("AFTER OPTIMIZATION")
print("="*60)

print(f"\n📋 Optimized Instruction:")
if hasattr(optimized, 'predictor'):
    optimized_instruction = optimized.predictor.signature.__doc__
    print(f'"{optimized_instruction}"')

    original_instruction = program.predictor.signature.__doc__
    if optimized_instruction != original_instruction:
        print(f"\n✨ INSTRUCTION CHANGED! ✨")
        print(f"\n  Before: '{original_instruction}'")
        print(f"\n  After:  '{optimized_instruction}'")
        print("\n  The optimizer learned the nonsense mapping!")
    else:
        print("\n(Instruction same, but checking few-shot examples...)")

print(f"\n🧪 Testing optimized version on dev set:")
correct_count = 0
for ex in devset:
    result = optimized(text=ex.text)
    is_correct = result.label == ex.label
    if is_correct:
        correct_count += 1
    correct = "✓" if is_correct else "✗"
    print(f"  {correct} '{ex.text[:45]}...' → {result.label:8s} (expected: {ex.label})")

print(f"\n📊 Optimized accuracy: {correct_count}/{len(devset)} ({100*correct_count/len(devset):.0f}%)")
print("    ↑ Should be much better now!")

print("\n📝 Full optimized prompt (last call):")
lm.inspect_history(n=1)

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE!")
print("="*60)