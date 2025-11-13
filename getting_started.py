import dspy
from general_utils import get_secret


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
        results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
        return [x["text"] for x in results]

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
