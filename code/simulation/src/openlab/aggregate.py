# import os
# import json
# import pandas as pd
# from sqlalchemy import create_engine, text
# from src.persona import PARTIES

# # DB 설정
# DB_USER = os.getenv("DB_USER", "pdp")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = os.getenv("DB_PORT", "5432")
# DB_NAME = os.getenv("DB_NAME", "persona")

# engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# # 💡 숫자를 정당 이름으로 변환하는 매핑 테이블
# OPTION_MAP = {
#     "1": "더불어민주당",
#     "2": "국민의힘",
#     "3": "무당층",
#     "4": "기타정당"
# }

# def calculate_final_rates(timepoint_id, year, month, week, poll_org):
#     tid_val = str(timepoint_id) 
    
#     actual_rates = {}
#     with engine.connect() as conn:
#         query_actual = text("""
#             SELECT * FROM public.party_support 
#             WHERE year = :y AND month = :m AND week = :w AND pollster = :p 
#             LIMIT 1
#         """)
#         df_actual = pd.read_sql(query_actual, conn, params={"y": year, "m": month, "w": week, "p": poll_org})
        
#         if df_actual.empty:
#             return {}, {}

#         row = df_actual.iloc[0]
#         actual_rates = {p: float(row[p]) / 100 for p in PARTIES if p in row}

#     # 2. AI 응답 JSON 가져오기
#     ai_counts = {p: 0 for p in PARTIES}
#     total_responses = 0

#     with engine.connect() as conn:
#         query_ai = text("""
#             SELECT response FROM public.persona_response_history 
#             WHERE timepoint_id = :tid 
#             ORDER BY timestamp DESC 
#             LIMIT 1
#         """)
#         result = conn.execute(query_ai, {"tid": tid_val}).fetchone()
        
#         if result and result[0]:
#             responses = result[0] if isinstance(result[0], dict) else json.loads(result[0])
            
#             for p_id, q_data in responses.items():
#                 total_responses += 1
#                 # q_data는 {"질문": "2"} 형태
#                 for ans_text in q_data.values():
#                     # 💡 숫자 응답 처리 로직
#                     clean_ans = str(ans_text).strip()
#                     # 첫 번째 글자가 숫자인지 확인 (혹시 "2." 처럼 올 경우 대비)
#                     first_char = clean_ans[0] if clean_ans else ""
                    
#                     if first_char in OPTION_MAP:
#                         # 숫자를 정당 이름으로 변환
#                         party_name = OPTION_MAP[first_char]
#                         if party_name in ai_counts:
#                             ai_counts[party_name] += 1
#                     else:
#                         # 혹시 숫자가 아니라 텍스트로 답했을 경우를 대비한 기존 로직 유지
#                         for party in PARTIES:
#                             if party in clean_ans:
#                                 ai_counts[party] += 1
#                                 break
#                     break # 한 페르소나당 질문 하나만 처리

#     # 3. 퍼센트 계산
#     ai_rates = {k: v / total_responses for k, v in ai_counts.items()} if total_responses > 0 else {}

#     return ai_rates, actual_rates

import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
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

# 3. SSH 터널 생성 및 시작
tunnel = SSHTunnelForwarder(
    (SSH_HOST, SSH_PORT),
    ssh_username=SSH_USER,
    ssh_password=SSH_PASSWORD,
    remote_bind_address=(DB_HOST, DB_PORT)
)
tunnel.start()

# 4. 터널링된 포트를 사용하여 SQLAlchemy 엔진 생성
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@127.0.0.1:{tunnel.local_bind_port}/{DB_NAME}"
)
def get_dashboard_data(timepoint_id):
    """
    [역할] timepoint_id를 기반으로 원본 프로필과 AI 응답을 병합하여
    대시보드에 필요한 '비율 통계(stats)'와 '상세 내역(df_merged)'을 반환합니다.
    """
    tid_val = str(timepoint_id)

    with engine.connect() as conn:
        # 1. AI 응답 JSON 가져오기
        query_ai = text("""
            SELECT response FROM public.persona_response_history 
            WHERE timepoint_id = :tid 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        result = conn.execute(query_ai, {"tid": tid_val}).fetchone()
        
        if not result or not result[0]:
            return None, None
            
        responses = result[0] if isinstance(result[0], dict) else json.loads(result[0])
        if not responses:
            return None, None
            
        pids = tuple(responses.keys())

        # 2. 원본 페르소나 데이터 가져오기 (하드코딩 제거)
        query_profile = text("""
            SELECT persona_id, gender, birth_year, residence_region, party_leaning 
            FROM public.persona_profile_test 
            WHERE persona_id IN :pids
        """)
        df_profile = pd.read_sql(query_profile, conn, params={"pids": pids})

    # 3. 데이터 병합 및 집계 (동적 카운팅)
    merged_data = []
    orig_counts = {}
    ai_counts = {}
    total_count = 0

    for _, row in df_profile.iterrows():
        p_id = row["persona_id"]
        # 기존 성향
        orig_val = str(row["party_leaning"]).strip() if pd.notna(row["party_leaning"]) else "기타/중도"
        
        q_data = responses.get(p_id, {})
        if not q_data:
            continue
            
        # 첫 번째 질문의 응답 객체 추출
        first_q_key = list(q_data.keys())[0]
        ans_obj = q_data[first_q_key]
        
        ai_val = "미분류"
        reason = ""
        if isinstance(ans_obj, dict):
            ai_val = str(ans_obj.get("Result", "미분류")).strip()
            reason = ans_obj.get("Reason", "")

        # 💡 [핵심] 진보, 보수 등 값이 무엇이든 동적으로 +1
        orig_counts[orig_val] = orig_counts.get(orig_val, 0) + 1
        ai_counts[ai_val] = ai_counts.get(ai_val, 0) + 1
        total_count += 1

        # DataFrame용 Row 추가
        merged_data.append({
            "persona_id": p_id,
            "gender": row["gender"],
            "birth_year": row["birth_year"],
            "region": row["residence_region"],
            "기존 성향": orig_val,
            "AI 변화 성향": ai_val,
            "Reason": reason
        })

    df_merged = pd.DataFrame(merged_data)

    # 4. 통계 산출 (n% -> m% 계산)
    # 기존 데이터와 AI 데이터에 등장한 모든 이념값 추출
    all_ideologies = set(orig_counts.keys()).union(set(ai_counts.keys()))
    stats = {}
    
    for ideology in all_ideologies:
        if ideology == "미분류": continue
        
        orig_rate = (orig_counts.get(ideology, 0) / total_count) * 100 if total_count > 0 else 0
        ai_rate = (ai_counts.get(ideology, 0) / total_count) * 100 if total_count > 0 else 0
        
        stats[ideology] = {
            "orig": orig_rate,
            "ai": ai_rate,
            "diff": ai_rate - orig_rate
        }

    return stats, df_merged