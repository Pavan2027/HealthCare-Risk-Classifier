"""
YouTube scraper for health-related comments and transcripts.
Uses YouTube Data API v3 and youtube-transcript-api.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from training.config import SCRAPED_DATA_DIR
from scraper.scraper_config import (
    YOUTUBE_SEARCH_QUERIES,
    YOUTUBE_MAX_RESULTS_PER_QUERY,
    YOUTUBE_MAX_COMMENTS_PER_VIDEO,
    HEALTH_KEYWORDS,
)

load_dotenv()


def _contains_health_keyword(text):
    """Check if text contains any health-related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in HEALTH_KEYWORDS)


def scrape_youtube(
    queries=None,
    max_results=None,
    max_comments=None,
    include_transcripts=True,
):
    """
    Scrape health-related YouTube video comments and transcripts.

    Args:
        queries: List of search queries
        max_results: Max videos per query
        max_comments: Max comments per video
        include_transcripts: Whether to fetch video transcripts

    Returns:
        DataFrame with scraped data
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key or api_key == "your_youtube_api_key":
        print("[!]  YouTube API key not configured.")
        print("   Set YOUTUBE_API_KEY in .env")
        print("   Get one at: https://console.cloud.google.com")
        return pd.DataFrame()

    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("[ERROR] google-api-python-client not installed.")
        print("   Run: pip install google-api-python-client")
        return pd.DataFrame()

    queries = queries or YOUTUBE_SEARCH_QUERIES
    max_results = max_results or YOUTUBE_MAX_RESULTS_PER_QUERY
    max_comments = max_comments or YOUTUBE_MAX_COMMENTS_PER_VIDEO

    # Build YouTube API client
    youtube = build("youtube", "v3", developerKey=api_key)

    all_data = []

    for query in queries:
        try:
            print(f"\n? Searching YouTube: '{query}'...")

            # Search for videos
            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_results,
                order="relevance",
            ).execute()

            video_count = 0
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                video_title = item["snippet"]["title"]
                channel = item["snippet"]["channelTitle"]
                published = item["snippet"]["publishedAt"]

                # Fetch comments
                try:
                    comments_response = youtube.commentThreads().list(
                        videoId=video_id,
                        part="snippet",
                        maxResults=max_comments,
                        order="relevance",
                        textFormat="plainText",
                    ).execute()

                    for comment_item in comments_response.get("items", []):
                        snippet = comment_item["snippet"]["topLevelComment"]["snippet"]
                        comment_text = snippet["textDisplay"]

                        if len(comment_text) < 20:
                            continue

                        all_data.append({
                            "text": comment_text[:1000],
                            "source": f"youtube/{channel}",
                            "platform": "youtube",
                            "url": f"https://youtube.com/watch?v={video_id}",
                            "author": snippet.get("authorDisplayName", "Unknown"),
                            "timestamp": snippet.get("publishedAt", ""),
                            "score": snippet.get("likeCount", 0),
                            "parent_text": video_title,
                            "type": "comment",
                        })

                except Exception as e:
                    # Comments might be disabled
                    pass

                # Fetch transcript
                if include_transcripts:
                    try:
                        from youtube_transcript_api import YouTubeTranscriptApi

                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        full_text = " ".join([t["text"] for t in transcript])

                        # Only include if health-related
                        if _contains_health_keyword(full_text):
                            # Split into chunks for classification
                            words = full_text.split()
                            chunk_size = 200  # ~200 words per chunk
                            for i in range(0, len(words), chunk_size):
                                chunk = " ".join(words[i:i + chunk_size])
                                if len(chunk) < 50:
                                    continue

                                all_data.append({
                                    "text": chunk[:2000],
                                    "source": f"youtube/{channel}",
                                    "platform": "youtube",
                                    "url": f"https://youtube.com/watch?v={video_id}",
                                    "author": channel,
                                    "timestamp": published,
                                    "score": 0,
                                    "parent_text": video_title,
                                    "type": "transcript",
                                })

                    except Exception:
                        # Transcript not available for all videos
                        pass

                video_count += 1

            print(f"   [OK] Processed {video_count} videos for '{query}'")

        except Exception as e:
            print(f"   [ERROR] Error searching '{query}': {e}")
            continue

    if not all_data:
        print("\n[!]  No data scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print(f"\n? Total scraped: {len(df)} items")
    print(f"   Comments: {len(df[df['type'] == 'comment'])}")
    print(f"   Transcripts: {len(df[df['type'] == 'transcript'])}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SCRAPED_DATA_DIR / f"youtube_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"? Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = scrape_youtube(
        queries=YOUTUBE_SEARCH_QUERIES[:3],  # Test with first 3 queries
        max_results=5,
        max_comments=10,
    )
    if len(df) > 0:
        print(f"\nSample data:")
        print(df[["text", "source", "type"]].head(10).to_string())
