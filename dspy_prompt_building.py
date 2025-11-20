import dspy
from general_utils import get_secret
from typing import Literal


# --- Configure Gemini ---
def setup_model():
    gemini_secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_secret)
    dspy.configure(lm=lm)
    return lm


# --- Keep your Classify signature exactly ---
class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


if __name__ == "__main__":
    model = setup_model()

    # Create a predictor using your Classify signature
    classify = dspy.Predict(Classify)

    # Run the model
    result = classify(sentence="This book was super fun to read, though not the last chapter.")
    print("\nClassification Result:")
    print(result)

    # Inspect the automatically built prompt
    print("\n--- Prompt Built by DSPy ---")
    print(model.inspect_history(n=1))
