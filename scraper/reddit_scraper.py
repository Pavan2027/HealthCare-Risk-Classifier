"""
Reddit scraper using PRAW (Python Reddit API Wrapper).
Scrapes health-related subreddits for claims and discussions.
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
    HEALTH_SUBREDDITS,
    REDDIT_SORT_OPTIONS,
    REDDIT_TIME_FILTER,
    REDDIT_MAX_POSTS_PER_SUB,
    REDDIT_MAX_COMMENTS_PER_POST,
    HEALTH_KEYWORDS,
)

load_dotenv()


def _contains_health_keyword(text):
    """Check if text contains any health-related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in HEALTH_KEYWORDS)


def scrape_reddit(
    subreddits=None,
    max_posts=None,
    max_comments=None,
    sort="hot",
):
    """
    Scrape health-related posts and comments from Reddit.

    Args:
        subreddits: List of subreddit names (default: HEALTH_SUBREDDITS)
        max_posts: Max posts per subreddit
        max_comments: Max comments per post
        sort: Sort method ('hot', 'new', 'top')

    Returns:
        DataFrame with scraped data
    """
    try:
        import praw
    except ImportError:
        print("[ERROR] PRAW not installed. Run: pip install praw")
        return pd.DataFrame()

    # Check credentials
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "HealthRiskClassifier/1.0")

    if not client_id or client_id == "your_reddit_client_id":
        print("[!]  Reddit API credentials not configured.")
        print("   Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
        print("   Register at: https://www.reddit.com/prefs/apps")
        return pd.DataFrame()

    subreddits = subreddits or HEALTH_SUBREDDITS
    max_posts = max_posts or REDDIT_MAX_POSTS_PER_SUB
    max_comments = max_comments or REDDIT_MAX_COMMENTS_PER_POST

    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    all_data = []

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            print(f"\n? Scraping r/{sub_name} ({sort})...")

            # Get posts based on sort method
            if sort == "hot":
                posts = subreddit.hot(limit=max_posts)
            elif sort == "new":
                posts = subreddit.new(limit=max_posts)
            elif sort == "top":
                posts = subreddit.top(
                    time_filter=REDDIT_TIME_FILTER, limit=max_posts
                )
            else:
                posts = subreddit.hot(limit=max_posts)

            post_count = 0
            for post in posts:
                # Skip stickied/pinned posts
                if post.stickied:
                    continue

                # Post text
                post_text = f"{post.title} {post.selftext}".strip()
                if not post_text or len(post_text) < 20:
                    continue

                # Check health relevance
                if not _contains_health_keyword(post_text):
                    continue

                all_data.append({
                    "text": post_text[:2000],  # Limit length
                    "source": f"r/{sub_name}",
                    "platform": "reddit",
                    "url": f"https://reddit.com{post.permalink}",
                    "author": str(post.author) if post.author else "[deleted]",
                    "timestamp": datetime.fromtimestamp(
                        post.created_utc
                    ).isoformat(),
                    "score": post.score,
                    "parent_text": "",
                    "type": "post",
                })
                post_count += 1

                # Get top comments
                post.comments.replace_more(limit=0)
                comment_count = 0
                for comment in post.comments[:max_comments]:
                    if not comment.body or len(comment.body) < 20:
                        continue
                    if comment.body == "[deleted]" or comment.body == "[removed]":
                        continue

                    all_data.append({
                        "text": comment.body[:1000],
                        "source": f"r/{sub_name}",
                        "platform": "reddit",
                        "url": f"https://reddit.com{comment.permalink}",
                        "author": str(comment.author) if comment.author else "[deleted]",
                        "timestamp": datetime.fromtimestamp(
                            comment.created_utc
                        ).isoformat(),
                        "score": comment.score,
                        "parent_text": post.title,
                        "type": "comment",
                    })
                    comment_count += 1

            print(f"   [OK] Scraped {post_count} posts from r/{sub_name}")

        except Exception as e:
            print(f"   [ERROR] Error scraping r/{sub_name}: {e}")
            continue

    if not all_data:
        print("\n[!]  No data scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print(f"\n? Total scraped: {len(df)} items")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SCRAPED_DATA_DIR / f"reddit_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"? Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = scrape_reddit(sort="hot")
    if len(df) > 0:
        print(f"\nSample data:")
        print(df[["text", "source", "score"]].head(10).to_string())
