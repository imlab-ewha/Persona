"""
simulation.py
-------------
시뮬레이션 및 JSON 기반 집계 저장 모듈.
- external_information 테이블을 참조하여 외부 환경 컨텍스트 생성
- 모든 페르소나의 응답을 {질문: {ID: 답변}} 형태의 JSON으로 통합 저장
- persona_response_history 테이블에 timepoint_id별 단일 행 적재
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import re

from sqlalchemy import create_engine, text
from src.persona import build_external_context_text, PARTIES

# # ── DB 설정 및 엔진 생성 ─────────────────────────────────────────
# DB_USER = os.getenv("DB_USER", "pdp")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = os.getenv("DB_PORT", "5432")
# DB_NAME = os.getenv("DB_NAME", "persona")

# engine = create_engine(
#     f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# )

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(_BASE, "prompts")

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
    return template.format_map(_SafeDict(**kwargs))

# ── 개별 페르소나 LLM 호출 ────────────────────────────────────
# def ask_persona(client, persona, week_info, external_context, prev_support, prompt_templates, model, provider="openai"):
#     fmt = dict(
#         profile=persona.get("profile", ""),
#         region=persona.get("region", ""),
#         week_info=week_info,
#         issues=external_context,
#         prev_support=prev_support
#     )

#     system_msg = _fill(prompt_templates["system"], **fmt)

#     # 💡 [프롬프트 스위치 로직]
#     if persona.get("party_leaning"):
#         user_msg = _fill(prompt_templates["user"], **fmt)
#         try:
#             query_text = user_msg.split("Question:")[1].split("Options:")[0].strip()
#         except:
#             query_text = "어느 정당을 지지하십니까?"
#     else:
#         # 값이 없는 경우 (혹은 필터링에서 제외된 경우 기본 질문)
#         user_msg = (
#             f"조사 시기: {week_info}\n"
#             f"전주 주요 정치 이슈: {external_context}\n\n"
#             f"전주 지지율: {prev_support}\n"
#             f"위 유권자가 현재 지지하는 정당은 무엇입니까?\n"
#             f"반드시 다음 중 하나만 답하세요: 더불어민주당 / 국민의힘 / 무당층 / 기타정당"
#         )
#         query_text = "어느 정당을 지지하십니까?"
#     messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

#     try:
#         if provider == "anthropic":
#             response = client.messages.create(
#                 model=model, max_tokens=100, system=system_msg,
#                 messages=[{"role": "user", "content": user_msg}]
#             )
#             answer = response.content[0].text.strip()
#         else:
#             response = client.chat.completions.create(
#                 model=model, max_tokens=100, messages=messages
#             )
#             answer = response.choices[0].message.content.strip()
        
#         # 정당 추출 (집계용)
#         final_party = "무당층"
#         for party in PARTIES:
#             if party in answer:
#                 final_party = party
#                 break
        
#         return final_party, query_text, answer # (집계용정당, 질문키, 원문답변)

#     except Exception as e:
#         print(f"Error calling LLM: {e}")
#         return "무당층", query_text, "응답 실패"
# ── 개별 페르소나 LLM 호출 ────────────────────────────────────
def ask_persona(client, persona, week_info, external_context, prev_support, prompt_templates, model, query, provider="openai"):
    fmt = dict(
        profile=persona.get("profile", ""),
        region=persona.get("region", ""),
        week_info=week_info,
        issues=external_context,
        prev_support=prev_support,
        query=query,
        party_leaning=persona.get("party_leaning", "")
    )

    system_msg = _fill(prompt_templates["system"], **fmt)
    user_msg = _fill(prompt_templates["user"], **fmt)

    # 기록용 질문 텍스트는 입력받은 query 그대로 사용
    query_text = query

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=400,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}]
            )
            answer = response.content[0].text.strip()
        else:
            response = client.chat.completions.create(
                model=model,
                max_tokens=400,
                messages=messages
            )
            full_answer = response.choices[0].message.content.strip()

        # 💡 [핵심] 정규표현식으로 Result와 Reason 섹션 추출
        # Result: 이후 문구와 Reason: 이후 문구를 각각 캡처합니다.
        result_match = re.search(r"Result:\s*(.*)", answer, re.IGNORECASE)
        reason_match = re.search(r"Reason:\s*(.*)", answer, re.IGNORECASE)

        # 결과가 없으면 원문 그대로, 있으면 텍스트만 추출
        res_val = result_match.group(1).strip() if result_match else "N/A"
        rea_val = reason_match.group(1).strip() if reason_match else answer

        # 💡 사용자가 원하는 딕셔너리 형태로 구성
        structured_response = {
            "Result": res_val.replace("[", "").replace("]", ""), # [진보] -> 진보
            "Reason": rea_val
        }

        return structured_response, query
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {"Result": "Error", "Reason": str(e)}, query

# ── DB 데이터 조회 (이전 주차 컨텍스트) ──────────────────────────
def get_prev_context_from_db(engine, target_year, target_month, target_week, poll_org_name):
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

# ── 주차 시뮬레이션 및 JSON 저장 ──────────────────────────────────
def simulate_week(engine, client, personas, target_row, prev_external_row, prompt_templates, model, provider="openai", query=""):
    """
    [역할] 전달받은 직전 주차 데이터를 사용하여 시뮬레이션을 수행합니다.
    """
    t_year, t_month, t_week = int(target_row['year']), int(target_row['month']), int(target_row['week'])
    target_info = f"{t_year}년 {t_month}월 {t_week}주차"
    
    # 1. 이미 main에서 검증된 직전 주차 데이터를 텍스트로 빌드
    external_context = build_external_context_text(prev_external_row)
    persona_pbar = tqdm(
        personas, 
        desc=f"    ㄴ {target_info} 페르소나 응답 수집", 
        leave=False, # 주차가 끝나면 바를 숨김
        position=1   # 바깥쪽 바(0) 아래에 위치하도록 설정
    )
    # 2. 응답 수집
    aggregated_responses = {}
    for persona in persona_pbar:
        res_obj, query_key = ask_persona(
            client, persona, 
            week_info=target_info, 
            external_context=external_context, 
            prev_support="", # 지지율 대조는 분석 단계에서 처리
            prompt_templates=prompt_templates, 
            model=model, provider=provider,
            query=query
        )

        p_id = persona["persona_id"]
        if p_id not in aggregated_responses:
            aggregated_responses[p_id] = {}
            
        # 💡 질문 키 아래에 {"Result": "...", "Reason": "..."} 객체가 저장됨
        aggregated_responses[p_id][query_key] = res_obj

    # DB 저장
    result_df = pd.DataFrame([{
        "timepoint_id": int(target_row['timepoint_id']),
        "response": json.dumps(aggregated_responses, ensure_ascii=False),
        "timestamp": pd.Timestamp.now()
    }])
    result_df.to_sql("persona_response_history", engine, if_exists='append', index=False, schema='public')

    return personas