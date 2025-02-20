"""
YouTube Video Summarizer using OCI Services

This script provides functionality to:
1. Download YouTube video transcripts
2. Translate transcripts using OCI AI Translation
3. Split transcripts into manageable segments
4. Generate summaries using OCI Generative AI
5. Translate summaries to Korean using OCI AI Translation
6. Save results with timestamps and translations

Requirements:
- OCI config file and API key
- OCI Translation service
- OCI Generative AI service
"""

#######################
# Imports
#######################

# API and ML related
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# OCI specific
import oci
from oci.ai_language import AIServiceLanguageClient
from oci.generative_ai_inference import GenerativeAiInferenceClient

# System and utilities
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import re
from typing import Dict, List, Literal
from pytube import YouTube
import requests

# Load environment variables
load_dotenv()

#######################
# Static Variables
#######################

# OCI Configuration
CONFIG_PATH = os.path.expanduser(os.getenv("OCI_CONFIG_PATH"))
KEY_PATH = os.path.expanduser(os.getenv("OCI_KEY_PATH"))
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
#MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyarleil5jr7k2rykljkhapnvhrqvzx4cwuvtfedlfxet4q" # meta.llama-3.1-405b-instruct
#MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyanrlpnq5ybfu5hnzarg7jomak3q6kyhkzjsl4qj24fyoq" # cohere.command-r-08-2024, not working
#MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyaodm6rdyxmdzlddweh4amobzoo4fatlao2pwnekexmosq" # cohere.command-r-plus-08-2024, not working
#MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyaiir6nnhmlgwvh37dr2mvragxzszqmz3hok52pcgmpqta" # meta.llama-3.1-70b-instruct
#MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya2xrydihzvu5pk6vlvfhtbnfapcvwhhugzo7jez4zcnaa" # meta.llama-3.2-90b-vision-instruct
MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyajqi26fkxly6qje5ysvezzrypapl7ujdnqfjq6hzo2loq" # meta.llama-3.3-70b-instruct, working model

# File System
RESULTS_DIR = "results"

# Test Configuration
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=MttW2lFnhKw"

# Model parameters
MODEL_PARAMS = {
    "max_tokens": 600,
    "temperature": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "top_p": 0.75,
    "top_k": 100
}

#######################
# Helper Functions
#######################

def verify_config():
    """Verify OCI configuration files and settings."""
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(KEY_PATH):
        print(f"Error: OCI config files not found")
        return False
    return True

#######################
# Main Class
#######################

class OCIYouTubeSummarizer:
    """Main class for handling YouTube video summarization using OCI services."""
    
    def __init__(self):
        """Initialize the summarizer with OCI clients."""
        if not verify_config():
            raise Exception("OCI configuration verification failed")
            
        config = oci.config.from_file()
        
        # Initialize OCI clients
        self.translation_client = AIServiceLanguageClient(config)
        self.llm_client = GenerativeAiInferenceClient(
            config=config,
            service_endpoint=ENDPOINT,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )

    def translate_text(self, text: str, source_lang: str = "en", target_lang: str = "ko") -> str:
        """Translate text using OCI Translation service."""
        try:
            translation_details = oci.ai_language.models.BatchLanguageTranslationDetails(
                documents=[
                    oci.ai_language.models.TextDocument(
                        key="1",
                        text=text,
                        language_code=source_lang
                    )
                ],
                target_language_code=target_lang
            )
            
            response = self.translation_client.batch_language_translation(
                batch_language_translation_details=translation_details
            )
            
            return response.data.documents[0].translated_text
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"Translation failed: {str(e)}"

    def generate_summary(self, text: str) -> str:
        """Generate summary using OCI Generative AI."""
        try:
            # Create chat request
            content = oci.generative_ai_inference.models.TextContent()
            content.text = f"Please summarize this text:\n\n{text}"
            content.type = "TEXT"
            
            message = oci.generative_ai_inference.models.Message()
            message.role = "USER"
            message.content = [content]
            
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
            chat_request.messages = [message]
            
            # Set model parameters
            for key, value in MODEL_PARAMS.items():
                setattr(chat_request, key, value)

            # Create chat details
            chat_details = oci.generative_ai_inference.models.ChatDetails()
            chat_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=MODEL_ID
            )
            chat_details.chat_request = chat_request
            chat_details.compartment_id = oci.config.from_file()["tenancy"]

            # Get response
            response = self.llm_client.chat(chat_details)
            return response.data.chat_response.choices[0].message.content[0].text
            
        except Exception as e:
            print(f"Summary generation error: {str(e)}")
            return f"Summary generation failed: {str(e)}"

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
            
            # If not English, translate to English using OCI Translation in chunks
            if transcript.language_code != 'en':
                print(f"Translating from {transcript.language} to English...")
                full_text = ' '.join([entry['text'] for entry in transcript_list])
                
                # Split text into chunks of 19000 characters (leaving room for buffer)
                MAX_CHUNK_SIZE = 4500
                text_chunks = [full_text[i:i + MAX_CHUNK_SIZE] 
                             for i in range(0, len(full_text), MAX_CHUNK_SIZE)]
                
                # Translate each chunk and combine
                translated_chunks = []
                for i, chunk in enumerate(text_chunks, 1):
                    print(f"Translating chunk {i} of {len(text_chunks)}...")
                    translated_chunk = self.translate_text(chunk, transcript.language_code, "en")
                    translated_chunks.append(translated_chunk)
                
                # Combine translated chunks
                translated_text = ' '.join(translated_chunks)
                transcript_list = [{'text': translated_text, 'start': 0}]
            
            # Combine transcript text with timestamps
            full_transcript = []
            current_minute = -1
            current_segment = []
            
            for entry in transcript_list:
                # Get minute from timestamp
                minute = int(entry.get('start', 0) / 60)
                
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

    def split_transcript(self, transcript, chunk_size=2000):
        """Split transcript into segments with timestamps"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
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
        """Summarize individual segment using OCI Generative AI"""
        prompt = f"""Please provide a clear and concise summary of the following video transcript segment.
        The transcript includes timestamps in [MM:SS] format.
        Focus on the main points and key information, noting any significant timing references.
        
        Transcript:
        {segment.page_content}
        
        Summary (be specific and include relevant timestamps):"""
        
        return self.generate_summary(prompt)

    def create_abstract(self, summaries):
        """Create final abstract from segment summaries using OCI Generative AI"""
        combined_summary = "\n".join(summaries)
        
        prompt = f"""Create a clear and comprehensive abstract that combines all these segment summaries into a coherent overview.
        Focus on the main themes and key points.
        
        Summaries:
        {combined_summary}
        
        Abstract (be concise and well-structured):"""
        
        return self.generate_summary(prompt)

    def translate_summaries(self, summaries: List[str]) -> List[Dict[str, str]]:
        """Translate segment summaries to Korean using OCI Translation"""
        translated_summaries = []
        for summary in summaries:
            translated_summaries.append({
                "english": summary,
                "korean": self.translate_text(summary, "en", "ko")
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
            "korean": self.translate_text(abstract, "en", "ko")
        }
        
        # Translate segment summaries and include timestamps
        translated_summaries = []
        for i, (summary, segment) in enumerate(zip(segment_summaries, self._current_segments), 1):
            time_range = segment.metadata.get('time_range', f"Part {i}")
            translated_summaries.append({
                "segment_number": i,
                "time_range": time_range,
                "english": summary,
                "korean": self.translate_text(summary, "en", "ko")
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
        safe_title = re.sub(r'[^\w\s-]', '', video_title)
        safe_title = re.sub(r'\s+', '_', safe_title)
        filename = f"summary_{safe_title}_{timestamp}.json"
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, filename), "w", encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        return filename

    def process_video(self, url):
        """Main process to summarize video"""
        # Load transcript
        transcript = self.load_video(url)
        
        # Split into segments
        segments = self.split_transcript(transcript)
        self._current_segments = segments
        
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
    1. Verifies OCI configuration
    2. Initializes summarizer with OCI services
    3. Processes the test video
    4. Prints results and saves to file
    """
    print(f"Testing OCI YouTube Summarizer with video: {TEST_VIDEO_URL}\n")
    
    try:
        # Initialize summarizer
        summarizer = OCIYouTubeSummarizer()
        
        # Process video
        result = summarizer.process_video(TEST_VIDEO_URL)
        
        print("\nSummary generated successfully!")
        print(f"Title: {result['title']}")
        print("\nAbstract:")
        print(result['abstract'])
        print(f"\nDetailed summary saved to: {result['saved_to']}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}") 