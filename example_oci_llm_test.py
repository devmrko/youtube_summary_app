from oci_youtube_summarizer import OCIYouTubeSummarizer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Test URLs - uncomment the one you want to test
TEST_URLS = [
    "https://www.youtube.com/watch?v=BunltHZk-pM",  # Korean video with auto-generated captions
]

def test_oci_summarizer():
    """Test the OCI YouTube Summarizer with different videos."""
    try:
        # Initialize the summarizer
        print("Initializing OCI YouTube Summarizer...")
        summarizer = OCIYouTubeSummarizer()
        
        # Process each test video
        for url in TEST_URLS:
            print(f"\n{'='*80}")
            print(f"Processing video: {url}")
            print(f"{'='*80}\n")
            
            try:
                # Process the video
                result = summarizer.process_video(url)
                
                # Print results
                print("\nSummary generated successfully!")
                print(f"\nTitle: {result['title']}")
                
                print("\nAbstract (English):")
                print("-" * 40)
                print(result["abstract"])
                
                print("\nSegment Summaries:")
                print("-" * 40)
                for i, summary in enumerate(result["segment_summaries"], 1):
                    print(f"\nSegment {i}:")
                    print(summary)
                
                print(f"\nFull results saved to: {result['saved_to']}")
                
            except Exception as e:
                print(f"Error processing video {url}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error initializing summarizer: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing OCI YouTube Summarizer Service...\n")
    test_oci_summarizer() 