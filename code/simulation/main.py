"""
main.py
-------
페르소나 기반 정당 지지율 시뮬레이션 실행 진입점
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from sqlalchemy import create_engine, text
import pandas as pd

# src 모듈에서 필요한 함수 로드
from src.persona import load_demographic_data, build_personas
from src.simulation import load_prompt, simulate_week, get_prev_week_external_data
from src.validation import calculate_final_rates

# ============================================================
# ⚙️ CONFIG (연구 조건 설정)
# ============================================================
MODEL         = "claude-sonnet-4-6"
PROMPT_VER    = "v7"           

# 1. 페르소나 필터 조건 (SQL WHERE 절 형식)
FILTER_CONDITION = "party_leaning IS NOT NULL"

# 2. 샘플링 설정
SAMPLING      = "full"         # "random" / "full"
N_PERSONAS    = 10             
RANDOM_SEED   = 42             

# 3. 주차 범위 설정
START_WEEK    = 0              
N_WEEKS       = 1              

# 4. 분석/비교 대상 기관
TARGET_POLLSTER = "gallup"

#시뮬레이션 몇 명으로 돌릴지
NUM           = 3
# ============================================================

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_USER = os.getenv("DB_USER", "pdp")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "persona")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

query = "종합소득세 상승"

def get_simulation_timeline(pollster: str) -> pd.DataFrame:
    """비교 대상 기관의 조사 일정을 기준으로 타임라인 확보"""
    query = text("SELECT DISTINCT timepoint_id, year, month, week FROM public.party_support WHERE pollster = :p ORDER BY timepoint_id;")
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"p": pollster})

def main():
    start_time = datetime.now()
    
    # ── [Step 0] 클라이언트 및 프롬프트 준비 ──────────────────
    if MODEL.startswith("claude"):
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        provider = "anthropic"
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        provider = "openai"

    prompt_templates = load_prompt(PROMPT_VER)

    # ── [Step 1] 페르소나 데이터 로딩 ──────────────────────────
    print(f"\n[Step 1] 페르소나 로딩 (Filter: {FILTER_CONDITION})")
    demo_df = load_demographic_data(n_sample=(N_PERSONAS if SAMPLING == "random" else None), 
                                    random_seed=RANDOM_SEED, 
                                    filter_condition=FILTER_CONDITION, limit=NUM)
    
    if demo_df.empty: return
    personas = build_personas(demo_df)

    # ── [Step 2] 타임라인 설정 ────────────────────────────────
    timeline = get_simulation_timeline(TARGET_POLLSTER)
    start_idx = START_WEEK
    end_idx = start_idx + N_WEEKS if N_WEEKS else len(timeline)
    sim_weeks = timeline.iloc[start_idx:end_idx].reset_index(drop=True)

    # ── [Step 3] 주차별 순회 실행 ─────────────────────────────
    print(f"\n[Step 2] 시뮬레이션 및 분석 시작")
    
    for _, target_row in sim_weeks.iterrows():
        curr_y, curr_m, curr_w = int(target_row['year']), int(target_row['month']), int(target_row['week'])
        curr_info = f"{curr_y}년 {curr_m}월 {curr_w}주차"
        
        prev_data_row, prev_info = get_prev_week_external_data(curr_y, curr_m, curr_w)
        
        if prev_data_row is None:
            py, pm, pw = prev_info
            print(f"  ⚠️ [Skip] {curr_info}: "
                  f"직전 주차({py}년 {pm}월 {pw}주차) 외부 데이터가 없어 건너뜁니다.")
            continue

        # 1. AI 시뮬레이션 실행 (기관 정보 배제, 순수 외부 정보 기반)
        personas = simulate_week(
            client=client, 
            personas=personas, 
            target_row=target_row, 
            prev_external_row=prev_data_row,
            prompt_templates=prompt_templates, 
            model=MODEL, 
            provider=provider,
            query = query
        )

        current_tid = target_row['timepoint_id'] # 예시: target_row에 이미 ID가 있는 경우
        # 2. 결과 집계 및 실제 데이터 대조 (여기서 기관 정보 투입)
        # year, month, week 정보를 넘겨서 실제 지표를 가져옵니다.
        # ai_rates, actual_rates = calculate_final_rates(
        #     timepoint_id=current_tid, 
        #     year=curr_y, 
        #     month=curr_m, 
        #     week=curr_w, 
        #     poll_org=TARGET_POLLSTER
        # )
        # if actual_rates:
        #     print(f"\n📊 [{curr_info}] 결과 대조 (기준: {TARGET_POLLSTER})")
            
        #     comparison = []
        #     for party in ai_rates.keys():
        #         ai_val = ai_rates[party]
        #         act_val = actual_rates.get(party, 0)
        #         comparison.append({
        #             "정당": party,
        #             "AI 예측": f"{ai_val:.1%}",
        #             "실제 지표": f"{act_val:.1%}",
        #             "오차": f"{ai_val - act_val:+.1%}"
        #         })
            
        #     # DataFrame을 사용하여 표 형태로 출력
        #     print(pd.DataFrame(comparison).to_string(index=False))
        #     print("-" * 60)

        
    # ── 완료 ──────────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    print(f"\n[완료] 총 소요시간: {elapsed.total_seconds()//60:.0f}분 {elapsed.total_seconds()%60:.0f}초")

if __name__ == "__main__":
    main()