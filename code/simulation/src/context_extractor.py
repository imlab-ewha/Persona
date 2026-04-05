import os
import re
import asyncio
import json
import urllib.parse
from typing import Dict
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

load_dotenv("/home/imlab/Persona/code/simulation/.env")

BROWSER_CONFIG = BrowserConfig(browser_type="chromium", headless=True, verbose=False)
OUTPUT_DIR = "/home/imlab/Persona/data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_safe_filename(text: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', "_", text)

async def fetch_naver_news_summary(keyword: str, max_articles: int = 5, gpt_api_key: str = None) -> Dict[str, str]:
    if not gpt_api_key:
        raise ValueError("GPT API Key가 필요합니다.")

    search_url = f"https://search.naver.com/search.naver?where=news&query={urllib.parse.quote(keyword)}"

    extraction_instruction = f"""
You are given a Naver news search result page.

Task:
- Find up to {max_articles} news results most relevant to the keyword "{keyword}".
- Use only what is visible on the search result page.
- Do NOT invent article contents that are not shown on the page.
- For each result, write a 1-2 sentence summary based on the title/snippet shown on the search page.
- Remove duplicates or near-duplicates.

Return ONLY a valid JSON array like this:
[
  {{"title": "...", "summary": "..."}},
  {{"title": "..." , "summary": "..."}}
]
"""

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="networkidle",
        delay_before_return_html=5,
        remove_overlay_elements=True,
        simulate_user=True,
        override_navigator=True,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(provider="openai/gpt-4o", api_token=gpt_api_key),
            instruction=extraction_instruction,
            input_format="html",
            extra_args={"temperature": 0}
        )
    )

    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        result = await crawler.arun(url=search_url, config=run_config)

        if not result.success:
            print("[CRAWL ERROR]", result.error_message)
            print("[STATUS CODE]", result.status_code)
            return {}

    extracted_content = (result.extracted_content or "").strip()
    # print("[DEBUG extracted_content]")
    # print(extracted_content[:3000])

    if extracted_content.startswith("```"):
        extracted_content = extracted_content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(extracted_content)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse extracted content for keyword: {keyword}")
        print("[JSON ERROR]", e)
        return {}

    # 단일 dict / {"articles":[...]} / list 전부 허용
    if isinstance(parsed, dict):
        if "articles" in parsed and isinstance(parsed["articles"], list):
            article_list = parsed["articles"]
        elif "title" in parsed and "summary" in parsed:
            article_list = [parsed]
        else:
            article_list = []
    elif isinstance(parsed, list):
        article_list = parsed
    else:
        article_list = []

    articles_json = {}
    for item in article_list:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "").strip()
        summary = item.get("summary", "").strip()
        if title and summary:
            articles_json[title] = summary

    return articles_json

if __name__ == "__main__":
    keyword = input("Enter Keyword: ").strip()
    max_articles = 5
    gpt_api_key = os.getenv("OPENAI_API_KEY")

    output = asyncio.run(fetch_naver_news_summary(keyword, max_articles=max_articles, gpt_api_key=gpt_api_key))

    safe_keyword = make_safe_filename(keyword)
    output_filename = os.path.join(OUTPUT_DIR, f"output_{safe_keyword}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 최종 결과 저장됨: {output_filename}")
    print("[RESULT]")
    print(json.dumps(output, ensure_ascii=False, indent=2))