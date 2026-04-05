"""
simulation.py
-------------
시뮬레이션 및 JSON 기반 집계 저장 모듈.
- external_information 테이블을 참조하여 외부 환경 컨텍스트 생성
- 모든 페르소나의 응답을 {질문: {ID: 답변}} 형태의 JSON으로 통합 저장
- persona_response_history 테이블에 timepoint_id별 단일 행 적재
"""
import ast
import streamlit as st
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import re
import asyncio
import nest_asyncio
nest_asyncio.apply()

from sqlalchemy import create_engine, text
from src.persona import build_external_context_text, PARTIES
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.context_extractor import fetch_naver_news_summary
import threading

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(_BASE, "prompts")

def extract_keywords(client, query, event, model, provider="anthropic"):
    # 🌟 굳이 따로 함수 안 만들고 기존 로더 사용
    prompts = load_prompt("keyword_extract")
    
    # {query}, {event} 치환
    final_user_msg = prompts["user"].replace("{query}", query).replace("{event}", event)

    try:
        if provider == "anthropic":
            resp = client.messages.create(
                model=model, max_tokens=1000, system=prompts["system"],
                messages=[{"role": "user", "content": final_user_msg}], 
                temperature=0.0
            )
            ans = resp.content[0].text.strip()
        else:
            resp = client.chat.completions.create(
                model=model, 
                messages=[
                    {"role": "system", "content": prompts["system"]}, 
                    {"role": "user", "content": final_user_msg}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.0
            )
            ans = resp.choices[0].message.content.strip()
        
        ans_clean = re.sub(r'^```json\s*|\s*```$', '', ans, flags=re.MULTILINE).strip()
        print(ans_clean)
        return json.loads(ans_clean)
        
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
        return {"query_keyword": "정치", "event_keyword": "사회이슈"}
    

def extract_keywords_embedding(client, query, event, model, provider="anthropic"):
    # 🌟 굳이 따로 함수 안 만들고 기존 로더 사용
    prompts = load_prompt("keyword_extract_em")
    
    # {query}, {event} 치환
    final_user_msg = prompts["user"].replace("{query}", query).replace("{event}", event)

    try:
        if provider == "anthropic":
            resp = client.messages.create(
                model=model, max_tokens=1000, system=prompts["system"],
                messages=[{"role": "user", "content": final_user_msg}], 
                temperature=0.0
            )
            ans = resp.content[0].text.strip()
        else:
            resp = client.chat.completions.create(
                model=model, 
                messages=[
                    {"role": "system", "content": prompts["system"]}, 
                    {"role": "user", "content": final_user_msg}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.0
            )
            ans = resp.choices[0].message.content.strip()
        
        ans_clean = re.sub(r'^```json\s*|\s*```$', '', ans, flags=re.MULTILINE).strip()
        print(ans_clean)
        return json.loads(ans_clean)
        
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
        return {"query_keyword": "정치", "event_keyword": "사회이슈"}

def get_aggregated_context(kw_dict: dict) -> dict:
    """
    [개선] fetch_naver_news_summary의 결과를 그대로 context에 담습니다.
    결과 구조: {"keyword": "A", "context": {"실제 뉴스 제목": "뉴스 요약 내용", ...}}
    """
    print("🔍 실시간 뉴스 크롤링 시작...")
    gpt_key = os.getenv("OPENAI_API_KEY")
    final_output = {}

    async def run_crawling():
        tasks = {}
        for key in ["query_keyword", "event_keyword"]:
            kw = kw_dict.get(key)
            if kw:
                # max_articles는 필요에 따라 조절 (여기선 10개)
                tasks[key] = (kw, fetch_naver_news_summary(kw, max_articles=5, gpt_api_key=gpt_key))
        
        for key, (kw, task) in tasks.items():
            try:
                # fetch_naver_news_summary가 이미 {제목: 요약}을 반환함
                news_results = await task
                print(news_results)
                final_output[key] = {
                    "keyword": kw,
                    "context": news_results # { "뉴스 제목": "요약" } 형태 유지
                }
            except Exception as e:
                print(f"❌ '{kw}' 크롤링 중 에러: {e}")
                final_output[key] = {"keyword": kw, "context": {}}

    def thread_wrapper():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_crawling())
        loop.close()

    thread = threading.Thread(target=thread_wrapper)
    thread.start()
    thread.join()

    print("✅ 뉴스 수집 완료")
    print(final_output)
    return final_output


# ── 프롬프트 로딩 ────────────────────────────────────────────
def load_prompt(version: str = "v1") -> dict[str, str]:
    path = os.path.join(PROMPTS_DIR, f"{version}.txt")
    with open(path, encoding="utf-8") as f:
        text_content = f.read()
    if "[SYSTEM]" in text_content and "[USER]" in text_content:
        system_part = text_content.split("[SYSTEM]")[1].split("[USER]")[0].strip()
        user_part = text_content.split("[USER]")[1].strip()
    else:
        system_part = text_content.strip()
        user_part = ""
    return {"system": system_part, "user": user_part}

class _SafeDict(dict):
    def __missing__(self, key): return "{" + key + "}"

def _fill(template: str, **kwargs):
    for k, v in kwargs.items():
        template = template.replace("{" + k + "}", str(v))
    return template

def evaluate_context_importance(client, persona, aggregated_context, model, provider="anthropic"):
    p_id = persona.get("persona_id")
    profile_combined = f"{persona['profile']}"
    v11_templates = load_prompt("v11_read_context")
    
    final_output = {}

    for cat_key, data in aggregated_context.items():
        actual_keyword = data["keyword"]
        original_titles = list(data["context"].keys())
        
        if not original_titles:
            final_output[cat_key] = {"keyword": actual_keyword, "context": {}}
            continue

        title_mapping = {f"item_{i}": title for i, title in enumerate(original_titles)}
        
        context_with_ids = "\n".join([f"{tid}: {title}" for tid, title in title_mapping.items()])

        fmt = {
            "profile": profile_combined,
            "context": context_with_ids
        }
        
        # 프롬프트 지시사항 수정 (시스템 프롬프트가 context_1 형태를 요구하므로 맞춰줌)
        system_msg = v11_templates["system"].replace("{profile}", fmt["profile"]).replace("{context}", fmt["context"])
        # 여기서 SYSTEM 지시사항에 "Use item_0, item_1... as keys"를 강제로 주입합니다.
        system_msg += "\n\nCRITICAL: Use the provided IDs (item_0, item_1, etc.) as the KEYS in your JSON output instead of original titles."
        
        user_msg = v11_templates["user"].replace("{profile}", fmt["profile"]).replace("{context}", fmt["context"])

        try:
            if provider == "anthropic":
                resp = client.messages.create(
                    model=model, max_tokens=2000, system=system_msg,
                    messages=[{"role": "user", "content": user_msg}], 
                    temperature=0.0
                )
                ans = resp.content[0].text.strip()
            else:
                resp = client.chat.completions.create(
                    model=model, 
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    response_format={"type": "json_object"}, 
                    temperature=0.0
                )
                ans = resp.choices[0].message.content.strip()
            
            ans_clean = re.sub(r'^```json\s*|\s*```$', '', ans, flags=re.MULTILINE).strip()
            start, end = ans_clean.find('{'), ans_clean.rfind('}')
            if start == -1 or end == -1: raise ValueError("JSON 구조 없음")
            json_str = ans_clean[start:end+1]
            
            # 줄바꿈 문자로 인한 unterminated string literal 방지
            json_str = json_str.replace('\n', ' ').replace('\r', '')

            try:
                raw_importance = json.loads(json_str)
            except:
                # ast.literal_eval 시도 전 따옴표/불리언 보정
                py_str = json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                raw_importance = ast.literal_eval(py_str)

            final_importance = {}
            for tid, analysis in raw_importance.items():
                if tid in title_mapping:
                    original_title = title_mapping[tid]
                    final_importance[original_title] = analysis
            
            final_output[cat_key] = {
                "keyword": actual_keyword,
                "context": final_importance
            }
            
        except Exception as e:
            print(f"⚠️ [ID: {p_id}] '{cat_key}' 파싱 실패: {e}")
            final_output[cat_key] = {"keyword": actual_keyword, "context": {}}

    print(f"✅ [분석 완료] ID: {p_id}", final_output)
    return final_output

def build_top_context(importance_data):
    """
    [역할] 
    1. evaluate_context_importance의 결과물을 입력받습니다.
    2. query_keyword와 event_keyword 각각에서 score 기준 상위 5개의 'thinking'을 추출합니다.
    3. 요청하신 계층 구조로 결과를 반환합니다.
    """
    final_result = {}

    for cat_key in ["query_keyword", "event_keyword"]:
        data = importance_data.get(cat_key)
        if not data:
            continue

        actual_keyword = data["keyword"]
        context_map = data["context"]  # {뉴스제목: {score, thinking}, ...}

        # 1. score(float) 기준 내림차순 정렬
        # importance_map.items() -> [(제목, {score, thinking}), ...]
        sorted_items = sorted(
            context_map.items(), 
            key=lambda x: float(x[1].get('score', 0)), 
            reverse=True
        )

        # 2. 상위 5개 추출 및 thinking 문구만 리스트로 저장
        top_5_thinkings = [
            item[1].get('thinking', '') 
            for item in sorted_items[:5] 
            if item[1].get('thinking')
        ]

        # 3. 요청하신 구조로 데이터 조립
        final_result[cat_key] = {
            "keyword": actual_keyword,
            "context": top_5_thinkings
        }

    return final_result

##지지가 변경해야하는 부분
def extract_survey_values(persona, kw_dict, k):
    """
    persona: 특정 페르소나 정보 >> persona.py에서 생성한 결과 넘겨주므로 신경 안써도 됨
    kw_dict: query랑 event에 대해서 keyword 뽑은 값인데 밑에 toggle로 자세히 정리함
    k: 상위 k개의 키워드 개수 추출
    1. survey_context의 모든 키(Key) 리스트를 추출
    2. kw_dict에서 query_keyword를 변수로 정의합니다 (향후 임베딩/매칭용 더미 로직)
    3. survey_context 내의 모든 문장(value)을 하나의 리스트로 통합하여 반환
    """
    survey_ctx = persona.get('survey_context', {})
    if not survey_ctx or not isinstance(survey_ctx, dict):
        return []

    try:
        all_keys = list(survey_ctx.keys())
        
        # kw_dict 에서 query_keyword의 value값을 변수로 정의
        target_query_kw = kw_dict.get("query_keyword", "")
        
        # 코드 짜야할 과정
        #1. 임베딩 과정
        #이 과정을 통해 top k의 key를 추출(all_keys에서의 특정 값들)해 top_k_list 생성
        #e.g., top_k_list=["social","society", ...]

        #2. 추출된 key들을 기준으로 value값들이 모인 string 생성
        # 1과정을 통해 만들어진 top_k_list를 통해 string 생성하는 것이므로 이 밑 코드는 그대로 쓰면 됨
        all_combined_sentences = []
        for key in all_keys:
            sentences = survey_ctx.get(key, [])
            if isinstance(sentences, list):
                all_combined_sentences.extend(sentences)
        
        return all_combined_sentences
        
    except Exception as e:
        print(f"⚠️ 문장 추출 중 오류 발생: {e}")
        return []
    
def format_top_context(top_context_data: dict) -> str:
    """build_top_context의 결과를 프롬프트용 문자열로 변환"""
    lines = []
    for cat_key in ["query_keyword", "event_keyword"]:
        data = top_context_data.get(cat_key)
        if data:
            lines.append(f"{data['keyword']}")
            for think in data['context']:
                lines.append(f"- {think}")
    return "\n".join(lines) if lines else "관련된 배경지식 없음"

def ask_persona(client, persona, query, options, event, top_context_data, prompt_templates, model, provider, kw_dict, k):
    """
    [개편된 시뮬레이션 엔진]
    1. Profile: 인구통계 정보 + 첫 번째 설문 카테고리의 1인칭 문장들
    2. Context: 뉴스 제목 기반 상위 5개 Thinking 요약
    3. Event: 현재 발생한 사건
    4. Output: 단일 선택 결과(response)와 이유(Reason)
    """
    p_id = persona.get("persona_id", "Unknown")

    base_profile = persona.get('profile', '')
    first_survey_sentences = extract_survey_values(persona, kw_dict, k) 
    survey_text = "\n".join(first_survey_sentences)
    
    profile_combined = f"{base_profile}\n\n[Persona's Values & Background]\n{survey_text}"

    context_combined = format_top_context(top_context_data)

    fmt = {
        "profile": profile_combined,
        "context": context_combined,
        "event": event,
        "query": query,
        "options": "\n".join([f"- {opt}" for opt in options])
    }
    
    # 템플릿 치환
    sys_msg = _fill(prompt_templates["system"], **fmt)
    usr_msg = _fill(prompt_templates["user"], **fmt)

    try:
        if provider == "anthropic":
            resp = client.messages.create(
                model=model, 
                max_tokens=2000, 
                system=sys_msg, 
                messages=[{"role": "user", "content": usr_msg}], 
                temperature=0.0
            )
            ans = resp.content[0].text.strip()
        else:
            resp = client.chat.completions.create(
                model=model, 
                messages=[
                    {"role": "system", "content": sys_msg}, 
                    {"role": "user", "content": usr_msg}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            ans = resp.choices[0].message.content.strip()
        
        ans_clean = re.sub(r'^```json\s*|\s*```$', '', ans, flags=re.MULTILINE).strip()
        start, end = ans_clean.find('{'), ans_clean.rfind('}')
        
        if start != -1 and end != -1:
            json_str = ans_clean[start:end+1]
            try:
                data = json.loads(json_str)
            except:
                data = ast.literal_eval(json_str)
            
            # 최종 결과 반환 (기존 result_map 대신 단일 response 반환)
            return {
                "response": data.get("response", "미응답"),
                "reason": data.get("reason", data.get("reason", "이유 없음"))
            }, query
            
    except Exception as e:
        return {"response": "Error", "reason": str(e)}, query

def simulate_event(client, personas, query, event, options, model, provider, st_bar=None):
    print("시뮬레이션 실행 버튼 누르기 완료")
    """
    [핵심 파이프라인]
    1. 키워드 추출
    2. 실시간 크롤링 (fetch_naver_news_summary 연동)
    3. 페르소나별 루프 (Thinking 생성 -> Top 5 추출 -> 최종 시뮬레이션)
    """
    # Step 1 & 2
    kw_dict = extract_keywords(client, query, event, model, provider)
    aggregated_context = get_aggregated_context(kw_dict)
    
    prompt_templates = load_prompt("v12")
    results = {}
    total = len(personas)

    K_VALUE = 5

    with ThreadPoolExecutor(max_workers=10) as executor:
        def run_pipeline(p):
            # Step 3: 분석
            imp_data = evaluate_context_importance(client, p, aggregated_context, model, provider)
            # Step 4: 요약
            top_ctx = build_top_context(imp_data)
            # Step 5: 결정
            res, _ = ask_persona(client, p, query, options, event, top_ctx, prompt_templates, model, provider, kw_dict, K_VALUE)
            return p["persona_id"], res

        futures = [executor.submit(run_pipeline, p) for p in personas]
        for i, f in enumerate(as_completed(futures)):
            p_id, res = f.result()
            results[p_id] = {query: res}
            if st_bar: st_bar.progress((i + 1) / total)
            
    return results
