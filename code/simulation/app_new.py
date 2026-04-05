import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import re
import ast

import nest_asyncio
nest_asyncio.apply()

# src 모듈 임포트
from src.persona import load_demographic_data, build_personas
# simulate_week 대신 simulate_event를 가져옴
from src.simulation2 import load_prompt, simulate_event

from openai import OpenAI
from anthropic import Anthropic
from sshtunnel import SSHTunnelForwarder

# ============================================================
# ⚙️ 1. 설정 및 세션 상태 초기화
# ============================================================
MODEL = "claude-sonnet-4-6" # 사용자가 지정한 모델명
PROMPT_VER = "v12" # simulate_event 내부에서 사용하는 프롬프트 버전

SURVEY_QUESTIONS = {
    "정당 지지 확률 추정": {
        "query": "현재 어느 정당을 가장 지지하십니까?",
        "options": ["더불어민주당", "국민의힘", "기타정당", "무당층"],
        "theme": {
            "더불어민주당": "#66bef8dd", "국민의힘": "#f98a7deb", 
            "무당층": "#95a5a6", "기타정당": "#dbd6ff"
        }
    }
}

if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False
if 'sim_result' not in st.session_state:
    st.session_state['sim_result'] = None

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ============================================================
# 🛠️ 2. DB 및 LLM 커넥션 설정
# ============================================================
SSH_HOST, SSH_PORT = os.getenv("SSH_HOST"), int(os.getenv("SSH_PORT", 4040))
SSH_USER, SSH_PASSWORD = os.getenv("SSH_USER"), os.getenv("SSH_PASSWORD")
DB_HOST, DB_PORT = os.getenv("DB_HOST", "127.0.0.1"), int(os.getenv("DB_PORT", 5432))
DB_USER, DB_PASSWORD, DB_NAME = os.getenv("DB_USER", "pdp"), os.getenv("DB_PASSWORD", "1234"), os.getenv("DB_NAME", "persona")

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

if "claude" in MODEL.lower():
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    provider = "anthropic"
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    provider = "openai"

# ============================================================
# 3. UI 렌더링 (입력부)
# ============================================================
st.set_page_config(page_title="실시간 여론 시뮬레이터", page_icon="📊", layout="wide")
st.title("페르소나 기반 실시간 여론 시뮬레이션")

lock_ui = st.session_state['is_running']

with st.sidebar:
    st.subheader("시뮬레이션 설정")
    selected_q_title = st.selectbox("분석할 설문 문항 선택", options=list(SURVEY_QUESTIONS.keys()), disabled=lock_ui)
    q_meta = SURVEY_QUESTIONS[selected_q_title]
    
    target_region = st.selectbox("지역", options=["서울특별시", "부산광역시"], disabled=lock_ui)
    num_personas = st.number_input("시뮬레이션 인원", min_value=1, max_value=100, value=10, disabled=lock_ui)
    context_input = st.text_area("사건 입력", placeholder="예: 정부의 민생 대책 발표", disabled=lock_ui)
    
    run_btn = st.button("시뮬레이션 실행", use_container_width=True, disabled=lock_ui)

# ============================================================
# 4. 시뮬레이션 실행 엔진
# ============================================================

if run_btn:
    if not context_input:
        st.warning("사건을 입력해주세요.")
        st.stop()
    st.session_state['is_running'] = True
    st.session_state['sim_result'] = None
    st.rerun()

if st.session_state['is_running'] and st.session_state['sim_result'] is None:
    try:
        current_tid = int(datetime.now().timestamp())
        
        # A. 페르소나 로드
        region_kw = "서울" if target_region == "서울특별시" else "부산"
        dynamic_filter = f"residence_region LIKE '%{region_kw}%'"
        demo_df = load_demographic_data(filter_condition=dynamic_filter, limit=num_personas)
        personas = build_personas(demo_df, q_meta["query"])
        
        # B. 🌟 변경된 시뮬레이션 함수 호출 (simulate_week -> simulate_event)
        progress_bar = st.progress(0)
        aggregated_responses = simulate_event(
            client=client,
            personas=personas,
            query=q_meta["query"],
            event=context_input,
            options=q_meta["options"],
            model=MODEL,
            provider=provider,
            st_bar=progress_bar
        )

        # C. 🌟 결과 데이터 가공 (확률 루프 제거 -> 단일 응답 추출)
        analysis_data = []
        option_counts = {opt: 0 for opt in q_meta["options"]}
        option_counts["미응답"] = 0

        for p_id, content in aggregated_responses.items():
            # content: {query: {"response": "...", "reason": "..."}}
            res_obj = content.get(q_meta["query"], {})
            winner = res_obj.get("response", "미응답")
            reason = res_obj.get("reason", "이유 없음")
            
            if winner in option_counts:
                option_counts[winner] += 1
            else:
                option_counts["미응답"] += 1
                
            p_info = demo_df[demo_df['persona_id'] == p_id].iloc[0]
            analysis_data.append({
                "id": p_id, "gender": p_info['gender'], "age": p_info['birth_year'],
                "orig": p_info.get('party_leaning', '미정'), "choice": winner,
                "reason": reason
            })

        # D. 결과 세션 저장
        st.session_state['sim_result'] = {
            "analysis": analysis_data, "counts": option_counts,
            "title": selected_q_title, "options": q_meta["options"], "theme": q_meta["theme"]
        }
        
        # E. DB 저장
        result_df = pd.DataFrame([{
            "timepoint_id": current_tid, 
            "response": json.dumps(aggregated_responses, ensure_ascii=False), 
            "timestamp": pd.Timestamp.now()
        }])
        result_df.to_sql("persona_response_history", engine, if_exists='append', index=False, schema='public')

        st.session_state['is_running'] = False
        st.rerun()

    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.session_state['is_running'] = False
        st.rerun()

# ============================================================
# 5. 결과 시각화
# ============================================================
if st.session_state['sim_result'] is not None:
    res = st.session_state['sim_result']
    st.divider()
    st.markdown(f"### 시뮬레이션 결과: {res['title']}")

    # KPI 메트릭
    cols = st.columns(len(res['options']))
    total_analysis = len(res['analysis'])
    for idx, opt in enumerate(res['options']):
        count = res['counts'].get(opt, 0)
        percent = (count / total_analysis) * 100 if total_analysis > 0 else 0
        color = res['theme'].get(opt, "#2c3e50")
        with cols[idx]:
            st.markdown(f"""
                <div style="border-top: 4px solid {color}; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                    <div style="font-size: 14px; font-weight: 600; color: #4a5568;">{opt}</div>
                    <div style="font-size: 24px; font-weight: 800; color: {color};">{percent:.1f}%</div>
                    <div style="font-size: 12px; color: #7f8c8d;">({count}명)</div>
                </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 페르소나별 상세 리포트")
    
    with st.container(height=600):
        for row in res['analysis']:
            c_orig = res['theme'].get(row['orig'], "#7f8c8d")
            c_choice = res['theme'].get(row['choice'], "#7f8c8d")
            
            st.markdown(f"""
                <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 15px; margin-bottom: 12px; background: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 14px;">
                            <b>{row['id']}</b> ({row['gender']}, {row['age']}년생) | 기존 성향: <b style="color:{c_orig}">{row['orig']}</b>
                        </span>
                        <span style="background:{c_choice}; color:white; padding:2px 10px; border-radius:15px; font-size:12px; font-weight:700;">{row['choice']}</span>
                    </div>
                    <div style="font-size: 14px; color: #2d3748; margin-top: 10px; padding-left: 10px; border-left: 3px solid #edf2f7;">
                        "{row['reason']}"
                    </div>
                </div>
            """, unsafe_allow_html=True)