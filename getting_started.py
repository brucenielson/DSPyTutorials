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


def search_wikipedia(query: str) -> list[str]:
    # Try multiple search strategies
    search_terms = [
        "David Gregory physician castle",
        "David Gregory Kinnairdy",
        "David Gregory 1625 Scotland"
    ]

    all_contexts = []
    seen_titles = set()

    for term in search_terms:
        try:
            results = wikipedia.search(term, results=3)
            for title in results:
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    # Get summary which is usually most relevant
                    all_contexts.append(page.summary)
                except:
                    continue
        except:
            continue

    return all_contexts[:3] if all_contexts else ["No relevant information found."]


def rag():
    rag_model = dspy.ChainOfThought("context, question -> response")

    question = "What's the name of the castle that David Gregory inherited?"
    result = rag_model(context=search_wikipedia(question), question=question)
    print(result)


if __name__ == "__main__":
    set_model()
    dspy.configure()

    # print("Chain of Thought Example:")
    # chain_of_thought()

    result = search_wikipedia("David Gregory")
    print("Search Wikipedia Example:")
    print(result)

    print("\nRAG Example:")
    rag()
