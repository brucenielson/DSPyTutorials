import dspy
from general_utils import get_secret
import wikipedia
from typing import Literal


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


class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""
    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")


class Outline(dspy.Signature):
    """Outline a thorough overview of a topic."""
    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")


class DraftSection(dspy.Signature):
    """Draft a top-level section of an article."""
    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section")


class DraftArticle(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
            section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)


if __name__ == "__main__":
    set_model()
    dspy.configure()

    # print("Chain of Thought Example:")
    # chain_of_thought()

    # result = search_wikipedia("David Gregory")
    # print("Search Wikipedia Example:")
    # print(result)
    #
    # print("\nRAG Example:")
    # rag()

    classify = dspy.Predict(Classify)
    result = classify(sentence="This book was super fun to read, though not the last chapter.")
    print("\nClassification Example:")
    print(result)

    # module = dspy.Predict(ExtractInfo)
    #
    # text = "Apple Inc. announced its latest iPhone 14 today." \
    #     "Introduction:\nThe CEO, Tim Cook, highlighted its new features in a press release."
    # response = module(text=text)
    # print(response)

    # draft_article = DraftArticle()
    # article = draft_article(topic="World Cup 2002")
    # print("\nDraft Article Example:")
    # # Break into title and sections
    # print(f"# {article.title}\n")
    # for section in article.sections:
    #     print(section)
    #     print("\n")
