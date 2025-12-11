from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 5

def summarize(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summary = summarizer(parser.document, SENTENCES_COUNT)
    return " ".join(str(sentence) for sentence in summary)

if __name__ == "__main__":
    sample_text = """
    Artificial intelligence represents a paradigm shift in how machines process information.
    Modern AI systems can learn from data, recognize patterns, and make decisions with minimal human intervention.
    Machine learning algorithms form the backbone of most AI applications today.
    Deep learning, a subset of machine learning, uses neural networks to solve complex problems.
    These technologies are revolutionizing industries from healthcare to finance.
    The potential applications of AI seem limitless as research continues to advance.
    """
    summary_text = summarize(sample_text)
    print("Summary:")
    print(summary_text)
