# fetch_bmce.py
# =====================================================
# BMCE Capital Bourse Data Fetcher (Robust Version)
# - Manual run only (no scheduler)
# - Async fetching with caching
# - Resilient error handling and logging
# =====================================================

import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional, Any, List
from urllib.parse import unquote

import pandas as pd
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from loguru import logger

# =====================================================
# Configuration
# =====================================================
class Config:
    BMCE_BASE = "https://www.bmcecapitalbourse.com/bkbbourse"
    SERIES_URL = f"{BMCE_BASE}/api/series/"
    LISTS_URL = f"{BMCE_BASE}/lists/"
    USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )

    OUTPUT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/stocks")
    CACHE_DIR = OUTPUT_DIR / ".cache"
    MASI_PATH = OUTPUT_DIR / "masi_series.csv"
    STOCKS_PATH = OUTPUT_DIR / "stocks_series.csv"
    LISTING_PATH = OUTPUT_DIR / "listing_directory.csv"

    MASI_LISTING_ID = "1356351,102,608"
    CACHE_TTL_SECONDS = 23 * 3600
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3


# Logging setup
logger.add("fetcher.log", rotation="5 MB", level="INFO")


# =====================================================
# Helpers
# =====================================================
def parse_series_json(data: dict, listing_id_hint: str) -> Optional[pd.DataFrame]:
    """Parse series JSON into DataFrame."""
    try:
        lid = data.get("listingId") or listing_id_hint
        name = data.get("name", "")
        prices = data.get("prices", [])

        if not prices:
            return None

        rows = [
            [
                pd.to_datetime(p.get("d"), unit="ms", utc=True)
                .tz_convert("Africa/Casablanca")
                .tz_localize(None),
                float(p["o"]),
                float(p["h"]),
                float(p["l"]),
                float(p["c"]),
                lid,
                name,
            ]
            for p in prices
            if all(k in p for k in ["d", "o", "h", "l", "c"])
        ]

        return pd.DataFrame(
            rows,
            columns=["Date", "Open", "High", "Low", "Close", "listingId", "name"],
        ).sort_values("Date")

    except Exception as e:
        logger.error(f"Failed to parse series for {listing_id_hint}: {e}")
        return None


async def fetch_url(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Fetch URL with retries, return JSON or text."""
    for attempt in range(1, Config.MAX_RETRIES + 1):
        try:
            async with session.get(
                url, params=params, timeout=Config.REQUEST_TIMEOUT
            ) as response:
                response.raise_for_status()
                if "application/json" in response.headers.get("Content-Type", ""):
                    return await response.json()
                return await response.text()
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed for {url}: {e}")
            await asyncio.sleep(1 * attempt)

    logger.error(f"All retries failed for {url}")
    return None


async def fetch_series_one(
    session: aiohttp.ClientSession,
    listing_id: str,
    semaphore: asyncio.Semaphore,
) -> Optional[pd.DataFrame]:
    """Fetch one stock series (with cache)."""
    raw_lid = unquote(listing_id)
    cache_file = Config.CACHE_DIR / f"series_{raw_lid.replace(',', '_')}.json"

    # Check cache validity
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < Config.CACHE_TTL_SECONDS:
        try:
            async with aiofiles.open(cache_file, "r") as f:
                cached_text = await f.read()
                data = json.loads(cached_text)
                logger.info(f"[CACHE HIT] {raw_lid}")
                return parse_series_json(data, raw_lid)
        except Exception:
            logger.warning(f"Cache read failed for {cache_file}")

    # Fetch fresh data
    async with semaphore:
        logger.info(f"[FETCH] {raw_lid}")
        params = {
            "lid": raw_lid,
            "max": "5000",
            "mode": "snap",
            "period": "1d",
            "vt": "yes",
            "intra": "true",
        }
        data = await fetch_url(session, Config.SERIES_URL, params)
        if data:
            try:
                async with aiofiles.open(cache_file, "w") as f:
                    await f.write(json.dumps(data))
            except Exception as e:
                logger.warning(f"Failed to write cache for {raw_lid}: {e}")
            return parse_series_json(data, raw_lid)

    return None


async def discover_listings(session: aiohttp.ClientSession) -> Dict[str, str]:
    """Scrape available stock listings from BMCE site."""
    logger.info("Discovering listings...")
    content = await fetch_url(session, Config.LISTS_URL)
    if not content:
        return {}

    found: Dict[str, str] = {}
    soup = BeautifulSoup(content, "html.parser")

    for a in soup.find_all("a", href=True):
        if "/bkbbourse/details/" in a["href"]:
            match = re.search(r"/details/([^;#?]+)", a["href"])
            if match:
                found[a.get_text(strip=True)] = unquote(match.group(1))

    logger.success(f"Found {len(found)} listings")
    return found


# =====================================================
# Main runner
# =====================================================
async def main_async():
    """Main async entrypoint."""
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": Config.USER_AGENT}
    connector = aiohttp.TCPConnector(limit_per_host=Config.MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        listings = await discover_listings(session)
        if not listings:
            logger.error("No listings discovered. Exiting.")
            return

        # Save listing directory
        pd.DataFrame([{"name": k, "listingId": v} for k, v in listings.items()]).to_csv(
            Config.LISTING_PATH, index=False
        )

        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)

        tasks: List[asyncio.Task] = [
            fetch_series_one(session, Config.MASI_LISTING_ID, semaphore)
        ]
        tasks.extend(
            fetch_series_one(session, lid, semaphore) for lid in listings.values()
        )

        results = await asyncio.gather(*tasks)

        # Split MASI vs stocks
        masi_df = results[0]
        stock_dfs = [df for df in results[1:] if df is not None]

        if masi_df is not None:
            masi_df.to_csv(Config.MASI_PATH, index=False)
            logger.success(f"Saved MASI -> {Config.MASI_PATH}")

        if stock_dfs:
            big_df = pd.concat(stock_dfs).sort_values(["name", "Date"])
            big_df.to_csv(Config.STOCKS_PATH, index=False)
            logger.success(f"Saved {len(stock_dfs)} stocks -> {Config.STOCKS_PATH}")


if __name__ == "__main__":
    asyncio.run(main_async())
