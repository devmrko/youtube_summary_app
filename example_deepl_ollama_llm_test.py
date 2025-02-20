from youtube_summarizer import YouTubeSummarizer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Choose which LLM provider to use
USE_OLLAMA = True  # Set to False to use OpenAI instead

if USE_OLLAMA:
    # Initialize summarizer with Ollama
    summarizer = YouTubeSummarizer(
        llm_provider="ollama",
        deepl_api_key=os.getenv("DEEPL_API_KEY"),
        ollama_model="mistral:instruct",  # or any other model you have in Ollama
        ollama_url=os.getenv("OLLAMA_URL")  # adjust if your Ollama server is elsewhere
    )
else:
    # Initialize summarizer with OpenAI
    summarizer = YouTubeSummarizer(
        llm_provider="openai",
        deepl_api_key=os.getenv("DEEPL_API_KEY")
    )

# Process a YouTube video 
url = "https://www.youtube.com/watch?v=isc4pCCpduE"
try:
    result = summarizer.process_video(url)

    # Print results
    print(f"\nTitle: {result['title']}")
    print(f"\nUsing {summarizer.llm_provider.upper()} for summarization")
    
    print("\nAbstract:")
    print(result["abstract"])

    print("\nSegment Summaries:")
    for i, summary in enumerate(result["segment_summaries"], 1):
        print(f"\nSegment {i}:")
        print(summary)

    print(f"\nResults saved to: {result['saved_to']}")
except Exception as e:
    print(f"Error processing video: {str(e)}")

#'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.'
