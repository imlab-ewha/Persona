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

from sqlalchemy import create_engine, text
from src.persona import build_external_context_text, PARTIES
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
# DB 설정
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", 4040))
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "pdp")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_NAME = os.getenv("DB_NAME", "persona")

tunnel = SSHTunnelForwarder(
    (SSH_HOST, SSH_PORT),
    ssh_username=SSH_USER,
    ssh_password=SSH_PASSWORD,
    remote_bind_address=(DB_HOST, DB_PORT)
)
tunnel.start()

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@127.0.0.1:{tunnel.local_bind_port}/{DB_NAME}"
)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(_BASE, "prompts")

def make_client(provider="openai"):
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    else:
        from openai import OpenAI
        return OpenAI()

def get_prev_week_external_data(engine, year, month, week):
    """
    [역할] 달력상 직전 주차를 계산하여 external_information 데이터를 조회합니다.
    - 2016년 1월 1주차 -> 2015년 12월 4주차
    - 2015년 11월 4주차 -> 2015년 11월 3주차
    """
    # 1. 직전 주차(Year, Month, Week) 계산
    if week > 1:
        prev_y, prev_m, prev_w = year, month, week - 1
    else:
        if month > 1:
            prev_y, prev_m, prev_w = year, month - 1, 4
        else:
            prev_y, prev_m, prev_w = year - 1, 12, 4

    # 2. DB에서 해당 주차의 external_information 조회
    query = text("""
        SELECT * FROM public.external_information 
        WHERE year = :y AND month = :m AND week = :w 
        LIMIT 1
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"y": prev_y, "m": prev_m, "w": prev_w})
        
        if df.empty:
            return None, (prev_y, prev_m, prev_w) # 데이터 없음
        return df.iloc[0], (prev_y, prev_m, prev_w) # 데이터 로우 반환

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
    
    
# ── DB 데이터 조회 (이전 주차 컨텍스트) ──────────────────────────
def get_prev_context_from_db(target_year, target_month, target_week, poll_org_name):
    """
    이전 주차의 외부 지표와 특정 기관의 지지율을 가져옵니다.
    정당명을 하드코딩하지 않고 PARTIES 리스트를 순회하며 자동으로 텍스트를 생성합니다.
    """
    if target_week > 1:
        py, pm, pw = target_year, target_month, target_week - 1
    else:
        pm = target_month - 1 if target_month > 1 else 12
        py = target_year if target_month > 1 else target_year - 1
        pw = 4 

    prev_external_str = "참조할 전주 외부 정보 없음"
    prev_support_str = "데이터 없음"

    with engine.connect() as conn:
        # 1. 통합 외부 정보 조회
        query_ext = text("SELECT * FROM public.external_information WHERE year=:y AND month=:m AND week=:w LIMIT 1")
        ext_df = pd.read_sql(query_ext, conn, params={"y": py, "m": pm, "w": pw})
        if not ext_df.empty:
            prev_external_str = build_external_context_text(ext_df.iloc[0])

        # 2. 통합 party_support 테이블에서 특정 pollster 데이터 조회
        # 💡 SELECT * 로 가져오거나 필요한 컬럼을 동적으로 처리합니다.
        query_supp = text("""
            SELECT * FROM public.party_support 
            WHERE year=:y AND month=:m AND week=:w AND pollster=:p 
            LIMIT 1
        """)
        supp_df = pd.read_sql(query_supp, conn, params={"y": py, "m": pm, "w": pw, "p": poll_org_name})
        
        if not supp_df.empty:
            s = supp_df.iloc[0]
            # 💡 [핵심] PARTIES 리스트를 순회하며 "정당명: 지지율" 형태로 자동 조립
            # s.get(p)를 써서 해당 컬럼이 DB에 있을 때만 포함시킵니다.
            support_parts = []
            for p in PARTIES:
                val = s.get(p)
                if pd.notna(val):
                    support_parts.append(f"{p}: {val}%")
            
            prev_support_str = ", ".join(support_parts)

    return prev_external_str, prev_support_str

import re

def get_news_window_5weeks(engine, year, month, week):
    """DB에서 5주치 뉴스를 가져와 YYYY-MM-DD 형식으로 클렌징하여 반환"""
    query = text("""
        SELECT year, month, week, news_text 
        FROM public.external_information
        WHERE (year < :y) 
           OR (year = :y AND month < :m)
           OR (year = :y AND month = :m AND week <= :w)
        ORDER BY year DESC, month DESC, week DESC
        LIMIT 5;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"y": year, "m": month, "w": week})
    
    news_window = {}
    for _, row in df.iterrows():
        cur_y = row['year']
        key = f"{cur_y}_{row['month']}_{row['week']}"
        lines = row['news_text'].split('\n')
        cleaned_items = []
        last_date = None
        
        for line in lines:
            item = line.strip().strip("- ")
            if not item: continue
            date_match = re.match(r"^(\d{1,2})/(\d{1,2})", item)
            if date_match:
                m, d = int(date_match.group(1)), int(date_match.group(2))
                last_date = f"{cur_y}-{m:02d}-{d:02d}"
                content = item[len(date_match.group(0)):].strip()
            else:
                content = item
                if not last_date: last_date = f"{cur_y}-{row['month']:02d}-??"
            cleaned_items.append(f"{last_date} {content}")
        news_window[key] = cleaned_items
    # print("news: ", news_window)
    return news_window

def evaluate_and_store_importance(client, persona, news_window, year, month, week, model, engine, provider="openai"):
    """
    [역할] 페르소나가 뉴스를 읽고 본인의 프로필에 비추어 중요도(score)와 생각(thinking)을 생성합니다.
    """
    p_id = persona["persona_id"]
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT important_score FROM public.persona_profile_test WHERE persona_id = :pid"),
            {"pid": p_id}
        ).fetchone()
    
    existing_scores = res[0] if res and res[0] else {}
    if isinstance(existing_scores, str):
        existing_scores = json.loads(existing_scores)

    # DB에 없는 주차만 API 호출하여 기억 생성
    missing_news_bundle = {w: items for w, items in news_window.items() if w not in existing_scores}
    if not missing_news_bundle:
        return existing_scores

    v10_templates = load_prompt("v10_readNews")
    fmt = {
        "persona_profile": persona.get('profile', ''),
        "reference_week": f"{year}_{month}_{week}",
        "external_information": json.dumps(missing_news_bundle, ensure_ascii=False, indent=2)
    }
    
    system_msg = _fill(v10_templates["system"], **fmt)
    user_msg = _fill(v10_templates["user"], **fmt)

    try:
        if provider == "anthropic":
            resp = client.messages.create(
                model=model, max_tokens=4000, system=system_msg,
                messages=[{"role": "user", "content": user_msg}], temperature=0.0
            )
            ans = resp.content[0].text.strip()
        else:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"}, temperature=0.0
            )
            ans = resp.choices[0].message.content.strip()
        
        ans_clean = re.sub(r'^```json\s*|\s*```$', '', ans, flags=re.MULTILINE).strip()
        new_data = json.loads(ans_clean)
        
        existing_scores.update(new_data)
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE public.persona_profile_test SET important_score = :js WHERE persona_id = :pid"),
                {"js": json.dumps(existing_scores, ensure_ascii=False), "pid": p_id}
            )
        return existing_scores
    except Exception as e:
        return existing_scores

def build_decayed_news_context(important_score, news_window_keys, target_year, target_month, target_week):
    """
    [역할] 5주치 뉴스 중 가중치(시간 감쇠)가 가장 높은 5개의 'thinking'을 합쳐 반환합니다.
    """
    decay_factor, scored_items = 0.8, []
    # 주차 간 거리 계산 함수
    def get_dist(y, m, w): return (target_year - y) * 48 + (target_month - m) * 4 + (target_week - w)

    for wk in news_window_keys:
        if wk not in important_score: continue
        try:
            y, m, w = map(int, wk.split('_'))
            dist = get_dist(y, m, w)
            for news_text, data in important_score[wk].items():
                raw_score = float(data.get('score', 0))
                # 시간 감쇠 적용: 먼 과거일수록 점수가 낮아짐
                decayed_score = raw_score * (decay_factor ** max(0, dist))
                scored_items.append({
                    "thinking": data.get('thinking', ''), 
                    "final_score": decayed_score
                })
        except: continue
        
    # 최종 점수 기준 내림차순 정렬 후 상위 5개 추출
    top_5 = sorted(scored_items, key=lambda x: x['final_score'], reverse=True)[:5]
    # print('top 5:', top_5)
    return "\n".join([f"- {i['thinking']}" for i in top_5])

def ask_persona(client, persona, query, options, external_context, news_thinking, prompt_templates, model, provider="anthropic"):
    p_id = persona.get("persona_id", "Unknown")
    fmt = {
        "profile": persona.get('profile', ''),
        "news_thinking": news_thinking if news_thinking else "No specific news memories.",
        "context": external_context, 
        "query": query,
        "options": "\n".join([f"- {opt}" for opt in options])
    }
    sys, usr = _fill(prompt_templates["system"], **fmt), _fill(prompt_templates["user"], **fmt)

    try:
        if provider == "anthropic":
            ans = client.messages.create(model=model, max_tokens=2000, system=sys, messages=[{"role": "user", "content": usr}], temperature=0.0).content[0].text
        else:
            ans = client.chat.completions.create(model=model, max_tokens=2000, messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature=0.0).choices[0].message.content
        
        # 🛡️ [강력한 파싱] unmatched ')' 에러 방지용 슬라이싱
        ans_clean = re.sub(r'```json|```', '', ans).strip()
        start, end = ans_clean.find('{'), ans_clean.rfind('}')
        result_map, reason_val = {opt: "0%" for opt in options}, "Parsing failed."

        if start != -1 and end != -1:
            json_str = ans_clean[start:end+1]
            try:
                data = json.loads(json_str)
            except:
                try: data = ast.literal_eval(json_str)
                except Exception as e:
                    print(f"   ⚠️ [ID: {p_id}] 파싱 실패 원문: {json_str}")
                    data = {}
            
            if data:
                # print(data)
                reason_val = data.get("Reason", data.get("reason", "No reason."))
                for opt in options:
                    pure = re.sub(r'^[0-9.\s]+', '', opt).strip()
                    val = data.get(pure) or data.get(opt)
                    if val is not None: result_map[opt] = f"{val}%" if isinstance(val, (int, float)) else str(val)
        
        return {"Result": result_map, "Reason": reason_val}, query
    except Exception as e:
        return {"Result": {"Error": "100%"}, "Reason": str(e)}, query

def process_persona_simulation(client, persona, year, month, week, news_window, 
                               prompt_templates, model, query, options, provider, 
                               external_context, engine, use_news_thinking=True): # 👈 명칭 변경
    """
    [데이터 흐름]
    - use_news_thinking이 True일 때만 DB에서 과거 뉴스 기억(Thinking)을 추출합니다.
    - False일 경우 과거 기억 없이 '현재 사건(context)'만 보고 판단합니다.
    """
    
    if use_news_thinking:
        # 1. 5주치 뉴스에 대한 페르소나의 기억(thinking) 및 점수(score) 로드/생성
        mems = evaluate_and_store_importance(client, persona, news_window, year, month, week, model, engine, provider)
        
        # 2. Score + 시간 감쇠를 통해 가장 지배적인 생각 Top 5 추출 -> {news_thinking}
        news_thinking = build_decayed_news_context(mems, list(news_window.keys()), year, month, week)
    else:
        # 과거 기억을 사용하지 않을 경우
        news_thinking = "No specific thoughts remembered from past news."

    # 3. 최종 시뮬레이션: {news_thinking} 상태에서 새로운 {context}를 만남
    res_obj, _ = ask_persona(
        client=client, persona=persona, query=query, options=options,
        external_context=external_context, # 프롬프트의 {context}로 매핑
        news_thinking=news_thinking,       # 프롬프트의 {news_thinking}으로 매핑
        prompt_templates=prompt_templates, model=model, provider=provider
    )

    return persona["persona_id"], res_obj

def importance_worker(client, persona, news_window, year, month, week, model, engine, provider="openai"):
    """
    기사별 importance score 생성만 수행
    반환:
        persona_id, importance_scores
    """
    p_id = persona["persona_id"]
    try:
        mems = evaluate_and_store_importance(
            client=client,
            persona=persona,
            news_window=news_window,
            year=year,
            month=month,
            week=week,
            model=model,
            engine=engine,
            provider=provider
        )
        return p_id, mems
    except Exception as e:
        return p_id, {"__error__": str(e)}


def simulate_week(client, personas, target_info, external_context, prompt_templates, 
                  model, query, options, engine, provider="openai", st_bar=None, 
                  use_news_thinking=True): # 👈 명칭 변경
    y, m, w = map(int, target_info.split('-'))
    
    # news_window 초기화 (기본값 빈 딕셔너리)
    news_window = {}
    
    # 💡 [핵심] 뉴스 반영 모드이고 엔진이 존재할 때만 DB에서 뉴스 윈도우를 가져옴
    if use_news_thinking and engine is not None:
        news_window = get_news_window_5weeks(engine, y, m, w)
    
    aggregated_responses = {}
    total = len(personas)
    MAX_WORKERS = 10

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_persona_simulation, 
                client, persona, y, m, w, news_window, prompt_templates, 
                model, query, options, provider, 
                external_context, engine, use_news_thinking
            ): persona for persona in personas
        }
        
        for i, f in enumerate(as_completed(futures)):
            try:
                p_id, res = f.result()
                if p_id not in aggregated_responses: aggregated_responses[p_id] = {}
                aggregated_responses[p_id][query] = res
                if st_bar: st_bar.progress((i + 1) / total)
            except Exception as e:
                st.error(f"시뮬레이션 중 오류 발생: {e}")
                
    return aggregated_responses