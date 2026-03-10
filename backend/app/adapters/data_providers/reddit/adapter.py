"""Reddit data provider adapter for market sentiment.

Monitors financial subreddits (wallstreetbets, investing, stocks, etc.)
for sentiment analysis and idea generation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from app.adapters.data_providers.base import BaseDataProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.reddit")

TARGET_SUBREDDITS = [
    "wallstreetbets", "investing", "stocks", "options",
    "SecurityAnalysis", "ValueInvesting", "algotrading",
    "thetagang", "dividends", "ETFs",
]


class RedditAdapter(BaseDataProvider):
    """Reddit adapter for financial sentiment data."""

    def __init__(self):
        super().__init__(name="reddit", enabled=settings.reddit_enabled)
        self._reddit = None

    async def connect(self) -> bool:
        """Initialize Reddit API connection via PRAW."""
        if not settings.reddit_client_id or not settings.reddit_client_secret:
            logger.warning("Reddit API credentials not configured")
            return False
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )
            # Test connection
            sub = self._reddit.subreddit("wallstreetbets")
            _ = sub.display_name
            logger.info("Reddit connection verified")
            return True
        except Exception as e:
            logger.error("Reddit connection failed", error=str(e))
            return False

    async def fetch_daily_bars(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Reddit doesn't provide price bars."""
        return pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Reddit doesn't provide fundamentals."""
        return pd.DataFrame()

    async def fetch_sentiment(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 100,
        time_filter: str = "day",
    ) -> pd.DataFrame:
        """Fetch recent posts from financial subreddits for sentiment analysis."""
        if not self._reddit:
            await self.connect()
        if not self._reddit:
            return pd.DataFrame()

        target_subs = subreddits or TARGET_SUBREDDITS
        all_posts = []

        for sub_name in target_subs:
            try:
                subreddit = self._reddit.subreddit(sub_name)
                for post in subreddit.hot(limit=limit):
                    all_posts.append({
                        "subreddit": sub_name,
                        "title": post.title,
                        "selftext": post.selftext[:2000] if post.selftext else "",
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        "url": post.url,
                        "author": str(post.author) if post.author else None,
                        "flair": post.link_flair_text,
                        "source": "reddit",
                        "received_at": datetime.now(timezone.utc),
                    })

                logger.info("Fetched Reddit posts", subreddit=sub_name, count=min(limit, 100))

            except Exception as e:
                logger.error("Reddit fetch error", subreddit=sub_name, error=str(e))

        return pd.DataFrame(all_posts) if all_posts else pd.DataFrame()

    async def health_check(self) -> Dict[str, Any]:
        """Check Reddit API connectivity."""
        try:
            if not self._reddit:
                await self.connect()
            if self._reddit:
                sub = self._reddit.subreddit("wallstreetbets")
                _ = sub.display_name
                return {"provider": self.name, "status": "healthy",
                        "last_check": datetime.now(timezone.utc).isoformat()}
            return {"provider": self.name, "status": "not_configured"}
        except Exception as e:
            return {"provider": self.name, "status": "unhealthy", "error": str(e)}
