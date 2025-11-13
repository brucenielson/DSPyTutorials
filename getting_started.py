import dspy
from general_utils import get_secret
import wikipedia


def set_model():
    # Get
    gemini_secret: str = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_secret)
    dspy.configure(lm=lm)
    return lm


def chain_of_thought():
    math = dspy.ChainOfThought("question -> answer: float")
    result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
    print(result)


def rag():
    def search_wikipedia(query: str) -> list[str]:
        # Use Wikipedia API directly instead
        try:
            # Search for relevant pages
            search_results = wikipedia.search(query, results=3)
            contexts = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    contexts.append(page.summary)
                except:
                    continue
            return contexts[:3] if contexts else ["No relevant information found."]
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return ["No relevant information found."]

    rag_model = dspy.ChainOfThought("context, question -> response")

    question = "What's the name of the castle that David Gregory inherited?"
    result = rag_model(context=search_wikipedia(question), question=question)
    print(result)


if __name__ == "__main__":
    set_model()
    dspy.configure()

    print("Chain of Thought Example:")
    chain_of_thought()

    print("\nRAG Example:")
    rag()
