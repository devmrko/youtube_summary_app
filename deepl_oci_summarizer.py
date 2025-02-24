"""
YouTube Video Summarizer using DeepL and OCI

This script combines:
1. DeepL for translations
2. OCI Generative AI for summarization
3. YouTube transcript handling
"""

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import oci
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import re
import deepl
from pytube import YouTube

# Load environment variables
load_dotenv()

# Static Variables
RESULTS_DIR = "results"
# MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyajqi26fkxly6qje5ysvezzrypapl7ujdnqfjq6hzo2loq"
MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyarleil5jr7k2rykljkhapnvhrqvzx4cwuvtfedlfxet4q"
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# Model parameters
MODEL_PARAMS = {
    "max_tokens": 600,
    "temperature": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "top_p": 0.75,
    "top_k": 100
}

class DeepLOCISummarizer:
    """Main class combining DeepL translation with OCI summarization"""
    
    def __init__(self):
        """Initialize with DeepL and OCI clients"""
        # Initialize DeepL
        self.translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
        
        # Initialize OCI
        config = oci.config.from_file()
        self.llm_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=ENDPOINT,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )

    def translate_text(self, text: str, source_lang: str = "en", target_lang: str = "ko") -> str:
        """Translate text using DeepL"""
        try:
            # Convert language codes to DeepL format
            source_mapping = {
                "en": "EN-US",
                "ko": "KO"
            }
            target_mapping = {
                "en": "EN-US",
                "ko": "KO"
            }
            
            # Map the language codes
            source = source_mapping.get(source_lang.lower(), source_lang.upper())
            target = target_mapping.get(target_lang.lower(), target_lang.upper())
            
            # Translate with correct language codes
            result = self.translator.translate_text(
                text, 
                source_lang=source if source != "EN-US" else None,  # Only specify source if not English
                target_lang=target
            )
            return result.text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"Translation failed: {str(e)}"

    def generate_summary(self, text: str) -> str:
        """Generate summary using OCI Generative AI"""
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

    def load_video(self, url):
        """Load and process YouTube transcript"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                raise ValueError("Could not extract video ID from URL")
                
            # Get video title
            try:
                yt = YouTube(url)
                video_title = yt.title
            except Exception as e:
                print(f"Warning: Could not get video title: {str(e)}")
                video_title = f"Video_{video_id}"
            
            self._video_title = video_title
            
            # Get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get English transcript first, then others
            transcript = None
            for lang in ['en', 'ko']:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    break
                except:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        break
                    except:
                        continue
            
            if not transcript:
                transcript = next(iter(transcript_list))
            
            # Fetch transcript
            transcript_list = transcript.fetch()
            
            # Translate if not English
            if transcript.language_code != 'en':
                print(f"Translating from {transcript.language} to English...")
                full_text = ' '.join([entry['text'] for entry in transcript_list])
                
                # Split text into chunks for translation
                MAX_CHUNK_SIZE = 4500
                text_chunks = [full_text[i:i + MAX_CHUNK_SIZE] 
                             for i in range(0, len(full_text), MAX_CHUNK_SIZE)]
                
                # Translate chunks
                translated_chunks = []
                for i, chunk in enumerate(text_chunks, 1):
                    print(f"Translating chunk {i} of {len(text_chunks)}...")
                    translated_chunk = self.translate_text(chunk, transcript.language_code, "en")
                    translated_chunks.append(translated_chunk)
                
                translated_text = ' '.join(translated_chunks)
                transcript_list = [{'text': translated_text, 'start': 0}]
            
            # Process transcript with timestamps
            full_transcript = []
            current_minute = -1
            current_segment = []
            
            for entry in transcript_list:
                minute = int(entry.get('start', 0) / 60)
                if minute != current_minute:
                    if current_segment:
                        full_transcript.append(' '.join(current_segment))
                    current_segment = []
                    current_minute = minute
                    current_segment.append(f"\n[{minute:02d}:00] ")
                current_segment.append(entry['text'])
            
            if current_segment:
                full_transcript.append(' '.join(current_segment))
            
            return [Document(
                page_content=' '.join(full_transcript),
                metadata={
                    "source": url,
                    "video_id": video_id,
                    "original_language": transcript.language,
                    "transcript_type": "auto-generated" if transcript.is_generated else "manual",
                    "translated": transcript.language_code != 'en'
                }
            )]
            
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            raise

    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        if 'youtube.com' in url:
            return re.search(r'v=([^&]*)', url).group(1)
        elif 'youtu.be' in url:
            return url.split('/')[-1]
        return None

    def split_transcript(self, transcript, chunk_size=800):
        """Split transcript into segments"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        segments = splitter.split_documents(transcript)
        
        for segment in segments:
            timestamps = re.findall(r'\[(\d{2}:\d{2})\]', segment.page_content)
            start_time = timestamps[0] if timestamps else "00:00"
            end_time = timestamps[-1] if timestamps else "end"
            segment.metadata['time_range'] = f"{start_time}-{end_time}"
            
        return segments

    def process_video(self, url):
        """Main process to summarize video"""
        # Load and process transcript
        transcript = self.load_video(url)
        
        # Split into segments
        segments = self.split_transcript(transcript)
        self._current_segments = segments
        
        # Generate summaries
        segment_summaries = []
        for segment in segments:
            summary = self.generate_summary(segment.page_content)
            segment_summaries.append(summary)
            
        # Generate abstract
        abstract = self.generate_summary("\n".join(segment_summaries))
        
        # Save results
        filename = self.save_summary(url, segment_summaries, abstract)
        
        return {
            "title": self._video_title,
            "segment_summaries": segment_summaries,
            "abstract": abstract,
            "saved_to": filename
        }

    def save_summary(self, url, segment_summaries, abstract):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_title = getattr(self, '_video_title', 'Untitled')
        
        # Translate summaries
        translated_summaries = []
        for i, (summary, segment) in enumerate(zip(segment_summaries, self._current_segments), 1):
            translated_summaries.append({
                "segment_number": i,
                "time_range": segment.metadata.get('time_range', f"Part {i}"),
                "english": summary,
                "korean": self.translate_text(summary, "en", "ko")
            })
        
        # Save results
        summary_data = {
            "title": video_title,
            "url": url,
            "timestamp": timestamp,
            "video_id": self.extract_video_id(url),
            "segment_summaries": translated_summaries,
            "abstract": {
                "english": abstract,
                "korean": self.translate_text(abstract, "en", "ko")
            }
        }
        
        # Create filename
        safe_title = re.sub(r'[^\w\s-]', '', video_title)
        safe_title = re.sub(r'\s+', '_', safe_title)
        filename = f"summary_{safe_title}_{timestamp}.json"
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, filename), "w", encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
            
        return filename

if __name__ == "__main__":
    # Test video
    TEST_URL = "https://www.youtube.com/watch?v=gjVWbAe-akw&t=315s"
    
    try:
        summarizer = DeepLOCISummarizer()
        result = summarizer.process_video(TEST_URL)
        
        print("\nSummary generated successfully!")
        print(f"Title: {result['title']}")
        print("\nAbstract:")
        print(result['abstract'])
        print(f"\nResults saved to: {result['saved_to']}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}") 