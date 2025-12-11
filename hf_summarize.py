from transformers import pipeline

# Load summarization pipeline (will download model automatically the first time)
summarizer = pipeline("summarization")

text = """
OpenAI's API usage has rate limits to ensure fair access. When you exceed these limits,
you get a RateLimitError. To avoid this, you can optimize your usage, add error handling,
or switch to free alternatives like Hugging Face transformers.
"""

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

print("Summary:", summary[0]['summary_text'])
