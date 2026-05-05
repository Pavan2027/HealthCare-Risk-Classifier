"""
Configuration for web scrapers.
Defines target subreddits, YouTube search queries, and health keywords.
"""

# --- Reddit Configuration ----------------------------------------------------
HEALTH_SUBREDDITS = [
    "health",
    "medicine",
    "COVID19",
    "Coronavirus",
    "nutrition",
    "AlternativeHealth",
    "NaturalRemedies",
    "antivax",
    "DebunkThis",
    "IsItBullshit",
    "AskDocs",
    "Supplements",
]

REDDIT_SORT_OPTIONS = ["hot", "new", "top"]
REDDIT_TIME_FILTER = "week"  # For top posts
REDDIT_MAX_POSTS_PER_SUB = 50
REDDIT_MAX_COMMENTS_PER_POST = 10

# --- YouTube Configuration ---------------------------------------------------
YOUTUBE_SEARCH_QUERIES = [
    "COVID vaccine truth",
    "natural cancer cure",
    "alternative medicine works",
    "ivermectin COVID treatment",
    "vaccine side effects danger",
    "essential oils health benefits",
    "detox cleanse health",
    "fluoride water danger",
    "5G health effects",
    "homeopathy evidence",
    "health misinformation debunked",
    "vaccine safety studies",
]

YOUTUBE_MAX_RESULTS_PER_QUERY = 10
YOUTUBE_MAX_COMMENTS_PER_VIDEO = 20

# --- Health Keywords (for filtering) -----------------------------------------
HEALTH_KEYWORDS = [
    "covid", "vaccine", "cure", "treatment", "cancer", "disease",
    "medicine", "drug", "therapy", "symptom", "diagnosis", "virus",
    "infection", "immune", "health", "medical", "doctor", "hospital",
    "pharmaceutical", "clinical", "trial", "study", "research",
    "FDA", "WHO", "CDC", "pandemic", "epidemic", "outbreak",
    "ivermectin", "hydroxychloroquine", "antibiotics", "supplement",
    "vitamin", "mineral", "herbal", "homeopathy", "acupuncture",
    "detox", "cleanse", "organic", "natural remedy", "essential oil",
    "side effect", "adverse", "mortality", "mRNA", "booster",
    "mask", "social distancing", "quarantine", "lockdown",
]

# --- Output Configuration ----------------------------------------------------
SCRAPER_OUTPUT_COLUMNS = [
    "text",
    "source",
    "platform",
    "url",
    "author",
    "timestamp",
    "score",  # upvotes/likes
    "parent_text",  # for comments: the post/video title
]
