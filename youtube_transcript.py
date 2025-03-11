from youtube_transcript_api import YouTubeTranscriptApi
import re
import math

def get_readable_time(seconds: int) -> str:
     return f"{math.floor(seconds // 3600)}:{math.floor(seconds % 3600 // 60)}:{math.floor(seconds % 60)}"

def get_transcript_with_time(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format the transcript with time information
        formatted_transcript = "\n".join(
            [f"ts=[{get_readable_time(entry['start'])}-{get_readable_time(entry['start']+entry['duration'])}] - script={entry['text']}" for entry in transcript]
        )
        
        return formatted_transcript
    
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return ""

def extract_video_id(url):
        """Extract video ID from YouTube URL"""
        if 'youtube.com' in url:
            return re.search(r'v=([^&]*)', url).group(1)
        elif 'youtu.be' in url:
            return url.split('/')[-1]
        return None

# Example usage
video_id = extract_video_id("https://youtu.be/Eua0Pjej1OE?si=VwvICxWMpTr6YQGv")
print(get_transcript_with_time(video_id))
