#!/usr/bin/env python3
"""
Bing Images downloader (Bing only).

Usage:
1) Edit SEARCH_QUERY and OUTPUT_DIR below.
2) Install dependencies:
   pip install playwright requests
   playwright install chromium
3) Run:
   python google_image_downloader.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import requests
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


# ----------------------------
# Editable settings
# ----------------------------
SEARCH_QUERY = "capybara"            # Change this keyword
OUTPUT_DIR = "downloads/capybara"    # Change save directory
MAX_IMAGES = 200                      # Number of images to download
HEADLESS = True                       # Keep True on servers without GUI/X display
SCROLL_PAUSE_SECONDS = 2.0
PAGE_LOAD_WAIT_SECONDS = 2.0
REQUEST_TIMEOUT = 15
# ----------------------------


def normalize_extension(content_type: str, url: str) -> str:
    if content_type:
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
        if "gif" in content_type:
            return ".gif"

    match = re.search(r"\.([a-zA-Z0-9]{2,5})(?:\?|$)", url)
    if match:
        ext = f".{match.group(1).lower()}"
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def collect_bing_image_urls(query: str, max_images: int, headless: bool = True) -> list[str]:
    search_url = f"https://www.bing.com/images/search?q={quote_plus(query)}"
    urls: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(search_url, wait_until="domcontentloaded")
        page.wait_for_timeout(int(PAGE_LOAD_WAIT_SECONDS * 1000))

        stagnation_rounds = 0
        last_count = 0

        while len(urls) < max_images and stagnation_rounds < 8:
            for src in page.eval_on_selector_all(
                "img",
                """
                els => {
                  const out = [];
                  for (const e of els) {
                    const candidates = [
                      e.getAttribute('src'),
                      e.getAttribute('data-src'),
                      e.getAttribute('data-src-hq'),
                      e.currentSrc,
                    ];
                    for (const c of candidates) {
                      if (c && c.startsWith('http') && !c.includes('bing.com/th?id=')) {
                        out.push(c);
                      }
                    }
                  }
                  return out;
                }
                """,
            ):
                urls.add(src)
                if len(urls) >= max_images:
                    break

            if len(urls) == last_count:
                stagnation_rounds += 1
            else:
                stagnation_rounds = 0
                last_count = len(urls)

            page.mouse.wheel(0, 9000)
            page.wait_for_timeout(int(SCROLL_PAUSE_SECONDS * 1000))

        browser.close()

    return list(urls)[:max_images]


def download_images(urls: list[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    saved = 0
    for idx, url in enumerate(urls, start=1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            ext = normalize_extension(content_type, url)
            file_path = output_dir / f"img_{idx:05d}{ext}"

            data = response.content
            if not data or b"<html" in data[:200].lower():
                continue

            with open(file_path, "wb") as f:
                f.write(data)
            saved += 1
            print(f"Saved {saved}: {file_path}")
        except Exception as exc:
            print(f"Skip {idx}: {url} -> {exc}")

    return saved


def main() -> None:
    effective_headless = HEADLESS
    if not effective_headless and not os.environ.get("DISPLAY"):
        print("No DISPLAY detected; switching to headless mode automatically.")
        effective_headless = True

    output_dir = Path(OUTPUT_DIR)

    print(f"Query: {SEARCH_QUERY}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Target count: {MAX_IMAGES}")

    try:
        urls = collect_bing_image_urls(SEARCH_QUERY, MAX_IMAGES, headless=effective_headless)
    except PlaywrightError as exc:
        print("Playwright launch failed. Ensure Chromium is installed: playwright install chromium")
        print(f"Original error: {exc}")
        return

    print(f"Collected {len(urls)} candidate image URLs from Bing")
    if not urls:
        print("No URLs collected. Try a different query or network.")
        return

    print("Downloading images...")
    downloaded_count = download_images(urls, output_dir)
    print(f"Finished. Downloaded {downloaded_count} files.")


if __name__ == "__main__":
    main()
