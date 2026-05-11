
import sys
import os
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trafilatura
from scraper.reddit_scraper import _contains_health_keyword

def extract_from_url(url: str):
    """
    Universal scraper that identifies the platform and extracts content.
    Supports: YouTube, Reddit, and Generic Blog/News sites.
    """
    print(f"[*] Analyzing URL: {url}")
    
    # 1. Check for YouTube
    if "youtube.com" in url or "youtu.be" in url:
        return _handle_youtube(url)
    
    # 2. Check for Reddit
    if "reddit.com" in url:
        return _handle_reddit(url)
    
    # 3. Default to Generic Web Scraping (Blogs/News)
    return _handle_generic(url)

def _handle_youtube(url):
    print("[*] Detected YouTube URL. Fetching transcript...")
    from youtube_transcript_api import YouTubeTranscriptApi
    
    # Extract Video ID
    video_id = None
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
        
    if not video_id:
        return {"error": "Could not extract YouTube Video ID"}
        
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])
        return {
            "title": "YouTube Video", # Could fetch title via API if needed
            "content": full_text,
            "platform": "youtube",
            "url": url
        }
    except Exception as e:
        return {"error": f"YouTube Transcript error: {str(e)}"}

def _handle_reddit(url):
    print("[*] Detected Reddit URL. Fetching post...")
    import praw
    from dotenv import load_dotenv
    load_dotenv()
    
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "HealthRiskClassifier/1.0")
    )
    
    try:
        submission = reddit.submission(url=url)
        content = f"{submission.title}\n\n{submission.selftext}"
        return {
            "title": submission.title,
            "content": content,
            "platform": "reddit",
            "url": url
        }
    except Exception as e:
        return {"error": f"Reddit API error: {str(e)}"}

def _handle_generic(url):
    print("[*] Detected Generic URL. Extracting with Trafilatura...")
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"error": "Could not fetch URL content"}
            
        result = trafilatura.extract(downloaded, include_comments=False)
        if not result:
            return {"error": "Could not extract main content from page"}
            
        return {
            "title": "Web Article",
            "content": result,
            "platform": "web",
            "url": url
        }
    except Exception as e:
        return {"error": f"Web extraction error: {str(e)}"}

if __name__ == "__main__":
    # Test with a known health blog or news article
    test_url = "https://www.healthline.com/nutrition/vitamin-c-benefits"
    data = extract_from_url(test_url)
    if "error" in data:
        print(f"Error: {data['error']}")
    else:
        print(f"\nTitle: {data['title']}")
        print(f"Platform: {data['platform']}")
        print(f"Content (first 200 chars): {data['content'][:200]}...")
