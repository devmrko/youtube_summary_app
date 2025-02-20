"""
YouTube Video Summarizer

This script provides functionality to:
1. Download YouTube video transcripts
2. Translate transcripts to English if needed
3. Split transcripts into manageable segments
4. Generate summaries using either OpenAI or Ollama
5. Translate summaries to Korean using DeepL
6. Save results with timestamps and translations

Requirements:
- OpenAI API key (if using OpenAI)
- DeepL API key (for translations)
- Ollama server (if using Ollama)
"""

#######################
# Imports
#######################

# API and ML related
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# System and utilities
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import re
import deepl
from typing import Dict, List, Literal
from pytube import YouTube
import requests
from requests.exceptions import ConnectionError
import socket
import time


# Load environment variables from .env file
load_dotenv()

#######################
# Static Variables
#######################

# API Configuration
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL") # Ollama server endpoint
DEFAULT_OLLAMA_MODEL = "mistral"  # Default Ollama model to use
DEFAULT_LLM_PROVIDER = "openai"  # Default LLM provider
DEFAULT_CONNECTION_TIMEOUT = 5  # Connection timeout in seconds

# API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API key for GPT models
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")  # DeepL API key for translations

# Test Configuration
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=MttW2lFnhKw"

# File System
RESULTS_DIR = "results"  # Directory for saving summary results

#######################
# Helper Functions
#######################

def test_ollama_connection(url: str, timeout: int = DEFAULT_CONNECTION_TIMEOUT) -> bool:
    """Test connection to Ollama server.
    
    Tests if the Ollama server is accessible and responding before attempting to use it.
    Falls back to OpenAI if Ollama is not available.
    """
    try:
        response = requests.get(f"{url}", timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to connect to Ollama server at {url}")
        print(f"Error: {str(e)}")
        return False

#######################
# Main Class
#######################

class YouTubeSummarizer:
    """Main class for handling YouTube video summarization.
    
    Provides methods to:
    - Extract and process YouTube transcripts
    - Translate non-English transcripts
    - Generate summaries using LLMs
    - Save results with translations
    """
    def __init__(self, llm_provider: Literal["openai", "ollama"] = DEFAULT_LLM_PROVIDER, 
                 api_key=None, deepl_api_key=None, 
                 ollama_model=DEFAULT_OLLAMA_MODEL, ollama_url=DEFAULT_OLLAMA_URL):
        """Initialize summarizer with choice of LLM provider.

        Args:
            llm_provider (str): LLM provider to use. Options: {DEFAULT_LLM_PROVIDER} or "ollama"
            api_key (str, optional): OpenAI API key (if using OpenAI). Defaults to None.
            deepl_api_key (str, optional): DeepL API key for translations. Defaults to None.
            ollama_model (str, optional): Model name for Ollama. Defaults to {DEFAULT_OLLAMA_MODEL}.
            ollama_url (str, optional): URL for Ollama API. Defaults to {DEFAULT_OLLAMA_URL}.

        Example:
            >>> summarizer = YouTubeSummarizer(
            ...     llm_provider="ollama",
            ...     ollama_model="mistral",
            ...     ollama_url="{DEFAULT_OLLAMA_URL}"
            ... )
        """
        self.llm_provider = llm_provider
        
        if llm_provider == "openai":
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            self.llm = OpenAI(temperature=0)
        else:  # ollama
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_url,
                temperature=0
            )
            
        self.translator = deepl.Translator(deepl_api_key) if deepl_api_key else None
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        video_id = None
        if 'youtube.com' in url:
            video_id = re.search(r'v=([^&]*)', url).group(1)
        elif 'youtu.be' in url:
            video_id = url.split('/')[-1]
        return video_id
        
    def load_video(self, url):
        """Load YouTube video transcript with fallback to auto-generated captions"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                raise ValueError("Could not extract video ID from URL")
                
            # Get video title using pytube
            try:
                yt = YouTube(url)
                video_title = yt.title
            except Exception as e:
                print(f"Warning: Could not get video title: {str(e)}")
                video_title = f"Video_{video_id}"
            
            print(f"\nProcessing video: {video_title}")
            print(f"Video ID: {video_id}")
            
            # Store video title for later use
            self._video_title = video_title
            
            # Get list of all available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Print available transcripts
            print("\nAvailable transcripts:")
            for transcript in transcript_list:
                print(f"- {transcript.language} ({transcript.language_code}): "
                      f"{'Auto-generated' if transcript.is_generated else 'Manual'}")
            
            # Try to get transcripts in this order:
            # 1. Manual English
            # 2. Auto-generated English
            # 3. Manual Korean
            # 4. Auto-generated Korean
            # 5. Any other available transcript
            
            transcript = None
            try:
                # Try manual English first
                transcript = transcript_list.find_manually_created_transcript(['en'])
                print("\nUsing manual English transcript")
            except:
                try:
                    # Try auto-generated English
                    transcript = transcript_list.find_generated_transcript(['en'])
                    print("\nUsing auto-generated English transcript")
                except:
                    try:
                        # Try manual Korean
                        transcript = transcript_list.find_manually_created_transcript(['ko'])
                        print("\nUsing manual Korean transcript")
                    except:
                        try:
                            # Try auto-generated Korean
                            transcript = transcript_list.find_generated_transcript(['ko'])
                            print("\nUsing auto-generated Korean transcript")
                        except:
                            # Get first available transcript
                            transcript = next(iter(transcript_list))
                            print(f"\nUsing available transcript in {transcript.language}")
            
            # Fetch the transcript
            transcript_list = transcript.fetch()
            
            # If not English, translate to English
            if transcript.language_code != 'en':
                print(f"Translating from {transcript.language} to English...")
                transcript_list = transcript.translate('en').fetch()
            
            # Combine transcript text with timestamps
            full_transcript = []
            current_minute = -1
            current_segment = []
            
            for entry in transcript_list:
                # Get minute from timestamp
                minute = int(entry['start'] / 60)
                
                # Add timestamp marker for every minute
                if minute != current_minute:
                    if current_segment:
                        full_transcript.append(' '.join(current_segment))
                    current_segment = []
                    current_minute = minute
                    current_segment.append(f"\n[{minute:02d}:00] ")
                
                current_segment.append(entry['text'])
            
            # Add the last segment
            if current_segment:
                full_transcript.append(' '.join(current_segment))
            
            full_text = ' '.join(full_transcript)
            
            # Create Document object
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": url,
                    "video_id": video_id,
                    "original_language": transcript.language,
                    "transcript_type": "auto-generated" if transcript.is_generated else "manual",
                    "translated": transcript.language_code != 'en'
                }
            )
            
            return [doc]
            
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            raise Exception(f"Failed to load video transcript: {str(e)}")

    def split_transcript(self, transcript, chunk_size=800):
        """Split transcript into segments with timestamps"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        segments = splitter.split_documents(transcript)
        
        # Extract timestamps for each segment
        segmented_texts = []
        for segment in segments:
            # Find all timestamps in the segment
            timestamps = re.findall(r'\[(\d{2}:\d{2})\]', segment.page_content)
            start_time = timestamps[0] if timestamps else "00:00"
            end_time = timestamps[-1] if timestamps else "end"
            
            # Add timestamp metadata
            segment.metadata['time_range'] = f"{start_time}-{end_time}"
            segmented_texts.append(segment)
        
        return segmented_texts

    def summarize_segment(self, segment):
        """Summarize individual segment"""
        prompt_template = """
        Summarize the following video transcript segment. The transcript includes timestamps in [MM:SS] format.
        Focus on the main content while preserving any relevant timing information.
        
        {text}
        
        Summary:
        """
        
        if self.llm_provider == "ollama":
            prompt_template = """
            Please provide a clear and concise summary of the following video transcript segment.
            The transcript includes timestamps in [MM:SS] format.
            Focus on the main points and key information, noting any significant timing references.
            
            Transcript:
            {text}
            
            Summary (be specific and include relevant timestamps):
            """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(
            self.llm,
            chain_type="stuff",
            prompt=prompt
        )
        return chain.invoke({"input_documents": [segment]})["output_text"]

    def create_abstract(self, summaries):
        """Create final abstract from segment summaries"""
        combined_summary = "\n".join(summaries)
        
        prompt_template = """
        Create a comprehensive abstract from these segment summaries:
        {text}
        
        Abstract:
        """
        
        if self.llm_provider == "ollama":
            prompt_template = """
            Create a clear and comprehensive abstract that combines all these segment summaries into a coherent overview.
            Focus on the main themes and key points.
            
            Summaries:
            {text}
            
            Abstract (be concise and well-structured):
            """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(
            self.llm,
            chain_type="stuff",
            prompt=prompt
        )
        doc = Document(page_content=combined_summary)
        return chain.invoke({"input_documents": [doc]})["output_text"]

    def translate_to_korean(self, text: str) -> str:
        """Translate text to Korean using DeepL"""
        if not self.translator:
            return "DeepL API key not provided"
        try:
            result = self.translator.translate_text(text, target_lang="KO")
            return result.text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"Translation failed: {str(e)}"

    def translate_summaries(self, summaries: List[str]) -> List[Dict[str, str]]:
        """Translate segment summaries to Korean"""
        translated_summaries = []
        for summary in summaries:
            translated_summaries.append({
                "english": summary,
                "korean": self.translate_to_korean(summary)
            })
        return translated_summaries

    def save_summary(self, url, segment_summaries, abstract):
        """Save summaries to JSON file with translations and timestamps"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the metadata from the first segment if available
        metadata = getattr(self, '_current_metadata', {})
        video_title = getattr(self, '_video_title', 'Untitled')
        
        # Translate abstract
        translated_abstract = {
            "english": abstract,
            "korean": self.translate_to_korean(abstract)
        }
        
        # Translate segment summaries and include timestamps
        translated_summaries = []
        for i, (summary, segment) in enumerate(zip(segment_summaries, self._current_segments), 1):
            time_range = segment.metadata.get('time_range', f"Part {i}")
            translated_summaries.append({
                "segment_number": i,
                "time_range": time_range,
                "english": summary,
                "korean": self.translate_to_korean(summary)
            })
        
        summary_data = {
            "title": video_title,
            "url": url,
            "timestamp": timestamp,
            "video_id": self.extract_video_id(url),
            "transcript_info": {
                "original_language": metadata.get("original_language", "unknown"),
                "type": metadata.get("transcript_type", "unknown"),
                "translated_from_original": metadata.get("translated", False)
            },
            "segment_summaries": translated_summaries,
            "abstract": translated_abstract
        }
        
        # Create filename with video title
        safe_title = re.sub(r'[^\w\s-]', '', video_title)  # Remove special characters
        safe_title = re.sub(r'\s+', '_', safe_title)  # Replace spaces with underscores
        filename = f"summary_{safe_title}_{timestamp}.json"
        
        with open('results/' + filename, "w", encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        return filename

    def process_video(self, url):
        """Main process to summarize video"""
        # Load transcript
        transcript = self.load_video(url)
        
        # Split into segments
        segments = self.split_transcript(transcript)
        self._current_segments = segments  # Store segments for timestamp reference
        
        # Summarize each segment
        segment_summaries = []
        for segment in segments:
            summary = self.summarize_segment(segment)
            segment_summaries.append(summary)
            
        # Create abstract
        abstract = self.create_abstract(segment_summaries)
        
        # Save results
        filename = self.save_summary(url, segment_summaries, abstract)
        
        return {
            "title": self._video_title,
            "segment_summaries": segment_summaries,
            "abstract": abstract,
            "saved_to": filename
        }

#######################
# Main Execution
#######################

if __name__ == "__main__":
    """
    Main execution flow:
    1. Tests Ollama server connection
    2. Falls back to OpenAI if Ollama is unavailable
    3. Initializes summarizer with appropriate provider
    4. Processes the test video
    5. Prints results and saves to file
    """
    print(f"Testing YouTube Summarizer with video: {TEST_VIDEO_URL}\n")
    
    try:
        # Test Ollama connection first
        if not test_ollama_connection(DEFAULT_OLLAMA_URL):
            print("\nFalling back to OpenAI...")
            llm_provider = "openai"
            if not OPENAI_API_KEY:
                raise Exception("OpenAI API key not found in environment variables")
        else:
            llm_provider = "ollama"
            
        # Initialize summarizer with selected provider
        summarizer = YouTubeSummarizer(
            llm_provider=llm_provider,
            api_key=OPENAI_API_KEY,
            deepl_api_key=DEEPL_API_KEY,
            ollama_model=DEFAULT_OLLAMA_MODEL,
            ollama_url=DEFAULT_OLLAMA_URL
        )
        
        # Process video
        result = summarizer.process_video(TEST_VIDEO_URL)
        
        print("\nSummary generated successfully!")
        print(f"Title: {result['title']}")
        print("\nAbstract:")
        print(result['abstract'])
        print(f"\nDetailed summary saved to: {result['saved_to']}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}") 