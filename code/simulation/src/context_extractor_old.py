import os
import re
import asyncio
import json
import urllib.parse
from typing import Dict
from dotenv import load_dotenv


from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# .env 절대경로 로드
load_dotenv("/home/imlab/Persona/code/simulation/.env")
# =========================
# Configuration
# =========================
BROWSER_CONFIG = BrowserConfig(browser_type="chromium", headless=True, verbose=False)
OUTPUT_DIR = "/home/imlab/Persona/data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_safe_filename(text: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', "_", text)


# =========================
# 단일 모듈용 함수
# =========================
async def fetch_naver_news_summary(keyword: str, max_articles: int = 5, gpt_api_key: str = None) -> Dict[str, str]:
    """
    키워드 기준 뉴스 검색 → LLM으로 각 기사 요약 → {제목: 요약} JSON 반환
    """
    if not gpt_api_key:
        raise ValueError("GPT API Key가 필요합니다.")

    search_url = f"https://search.naver.com/search.naver?where=news&query={urllib.parse.quote(keyword)}"

    extraction_instruction = f"""
You are given a Naver news search result page.
Identify the {max_articles} most relevant articles related to the keyword '{keyword}'.
For each article, summarize the main content in 1-2 concise sentences.
Return only valid JSON in the format: {{"title": "article title", "summary": "article content summary"}}
Remove duplicates or near-duplicates. Maximum {max_articles} articles.
"""

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        delay_before_return_html=5,
        remove_overlay_elements=True,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(provider="openai/gpt-4o", api_token=gpt_api_key),
            instruction=extraction_instruction
        )
    )

    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        result = await crawler.arun(url=search_url, config=run_config)

    extracted_content = result.extracted_content or ""
    if extracted_content.startswith("```"):
        extracted_content = extracted_content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(extracted_content)
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse extracted content for keyword: {keyword}")
        return {}

    # JSON을 {제목: 요약} 구조로 변환
    articles_json = {}
    if isinstance(parsed, dict):
        article_list = parsed.get("articles", [])
    elif isinstance(parsed, list):
        article_list = parsed
    else:
        article_list = []

    for item in article_list:
        title = item.get("title", "").strip()
        summary = item.get("summary", "").strip()
        if title and summary:
            articles_json[title] = summary

    return articles_json


# =========================
# main
# =========================
if __name__ == "__main__":
    keyword = input("Enter Keyword: ").strip()
    max_articles = 5
    gpt_api_key = os.getenv("OPENAI_API_KEY")


    output = asyncio.run(fetch_naver_news_summary(keyword, max_articles=max_articles, gpt_api_key=gpt_api_key))

    # JSON 저장
    safe_keyword = make_safe_filename(keyword)
    output_filename = os.path.join(OUTPUT_DIR, f"output_{safe_keyword}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 최종 결과 저장됨: {output_filename}")
    print("[RESULT]")
    print(json.dumps(output, ensure_ascii=False, indent=2))