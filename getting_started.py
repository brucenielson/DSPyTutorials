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
    import wikipedia

    try:
        results = wikipedia.search(query, results=5)
        contexts = []

        for title in results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                contexts.append(page.summary)
            except:
                continue

        return contexts[:10] if contexts else ["No information found."]
    except:
        return ["No information found."]


def rag():
    # Step 1: Use LLM to generate better search queries
    query_generator = dspy.ChainOfThought("question -> search_query: str")

    question = "What's the name of the castle that David Gregory inherited?"

    # Let the LLM generate a better search query
    search_result = query_generator(question=question)
    print(f"Generated search query: {search_result.search_query}")

    # Use that query to search
    contexts = search_wikipedia(search_result.search_query)

    # Now answer with the retrieved context
    rag_model = dspy.ChainOfThought("context, question -> response")
    result = rag_model(context=contexts, question=question)
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
