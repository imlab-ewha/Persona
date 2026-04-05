import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import plotly.express as px

# 제공된 src 모듈 임포트
from src.openlab.persona import load_demographic_data, build_personas, build_external_context_text
from src.openlab.simulation import get_prev_week_external_data, simulate_week, load_prompt
from src.openlab.aggregate import get_dashboard_data

from openai import OpenAI
from anthropic import Anthropic
from sshtunnel import SSHTunnelForwarder

# ============================================================
# ⚙️ 1. 초기 설정 및 세션 상태 초기화
# ============================================================
MODEL = "claude-sonnet-4-6"
PROMPT_VER = "openlab"
FILTER_CONDITION = "party_leaning IS NOT NULL AND birth_year <= 2007"

if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False
if 'sim_result' not in st.session_state:
    st.session_state['sim_result'] = None

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# DB 및 커넥션 설정 (기존 유지)
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

if MODEL.startswith("claude"):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    provider = "anthropic"
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    provider = "openai"

prompt_templates = load_prompt(PROMPT_VER)

# ============================================================
# 2. UI 구성 (사이드바 입력부)
# ============================================================

st.set_page_config(page_title="여론 변화 시뮬레이터", page_icon="📊", layout="wide")
st.title("페르소나 기반 여론 변화 시뮬레이션")
st.markdown("특정 사건이나 정책에 따른 여론 변화를 시뮬레이션합니다.")

# 실행 중일 때 UI 잠금
lock_ui = st.session_state['is_running']
st.markdown("""
    <style>
    button[kind="primary"] {
        background-color: #63B3ED !important;
        color: #1A365D !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #3182CE !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("시뮬레이션 설정")
    target_region = st.selectbox("지역", options=["서울특별시", "부산광역시"], disabled=lock_ui)
    num_personas = st.number_input("시뮬레이션 할 페르소나 인원", min_value=1, max_value=30, value=25, disabled=lock_ui)
    custom_query = st.text_area("사건 입력", placeholder="발생 가능한 사건 및 전달할 메시지를 입력하세요.", disabled=lock_ui)
    run_btn = st.button("시뮬레이션 실행", use_container_width=True, disabled=lock_ui, type='primary')

# ============================================================
# 3. 시뮬레이션 실행 엔진
# ============================================================
if run_btn:
    if not custom_query:
        st.warning("사건을 입력해주세요.")
        st.stop()
    st.session_state['is_running'] = True
    st.session_state['sim_result'] = None
    st.rerun()

if st.session_state['is_running'] and st.session_state['sim_result'] is None:
    status_container = st.empty()
    progress_container = st.empty()

    try:
        # --- [준비 단계] ---
        # 현재 시점 설정 (예: 2026년 3월 4주차 시뮬레이션 가정)
        target_year, target_month, target_week = 2026, 3, 4
        current_tid = int(datetime.now().timestamp())
        
        # LLM 클라이언트 설정
        if MODEL.startswith("claude"):
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            provider = "anthropic"
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            provider = "openai"
        
        prompt_templates = load_prompt(PROMPT_VER)

        # 1. 전주 외부 데이터 조회 (simulation.py의 함수 활용)
        prev_external_row, _ = get_prev_week_external_data(engine, target_year, target_month, target_week)
        
        # 2. 페르소나 데이터 로딩 및 빌드
        region_kw = "서울" if target_region == "서울특별시" else "부산"
        dynamic_filter = f"party_leaning IS NOT NULL AND residence_region LIKE '%%{region_kw}%%'"
        demo_df = load_demographic_data(filter_condition=dynamic_filter, limit=num_personas)
        personas = build_personas(demo_df, custom_query)

        # --- [실행 단계: 병렬 처리 호출] ---
        status_container.info("시뮬레이션 진행 중 입니다.")
        progress_bar = progress_container.progress(0)

        # simulate_week에 넘겨줄 target_row 더미 생성
        target_row_dummy = {
            'year': target_year, 'month': target_month, 'week': target_week, 
            'timepoint_id': current_tid
        }

        # 🌟 핵심: simulation.py에 정의한 병렬 함수를 직접 호출!
        # 이 함수 내부에서 DB 저장까지 한 번에 수행됩니다.
        aggregated_responses = simulate_week(
            client=client,
            personas=personas,
            target_row=target_row_dummy,
            prompt_templates=prompt_templates,
            model=MODEL,
            provider=provider,
            query=custom_query,
            st_bar=progress_bar
        )

        # --- [결과 정리 단계] ---
        # 대시보드 표시를 위해 데이터 재조회
        stats, df_merged = get_dashboard_data(current_tid)
        st.session_state['sim_result'] = {
            "stats": stats, "df": df_merged, "query": custom_query, 
            "region": target_region, "count": num_personas
        }
        status_container.empty()
        progress_container.empty()

        st.session_state['is_running'] = False
        st.rerun()

    except Exception as e:
        st.error(f"시뮬레이션 중 오류 발생: {e}")
        st.session_state['is_running'] = False
        st.stop()

# ============================================================
# 4. 결과 대조 및 대시보드 시각화
# ============================================================
if st.session_state['sim_result'] is not None:
    res = st.session_state['sim_result']
    stats = res['stats']
    df_merged = res['df']

    st.markdown("#### 시뮬레이션 정보 요약")
    info_col1, info_col2 = st.columns(2)
    info_col1.info(f"**지역:** {res['region']}\n\n**명수:** {res['count']}명")
    info_col2.success(f"**질의:** {res['query']}")
    st.divider()

    st.markdown("#### 정치적 지지율 변동 예측")
    
    if stats and df_merged is not None:
        # 지표 카드 (기존 디자인 유지)
        cols = st.columns(len(stats))
        theme_colors = {
            "진보": {"text": "#2e89c6dd", "bg": "#eff6fb91", "border": "#3182ce"},
            "보수": {"text": "#eb5b5beb", "bg": "#fdefefa0", "border": "#e53e3e"},
            "중도": {"text": "#626262", "bg": "#f7f7f7", "border": "#4a5568"},
            "기타/중도": {"text": "#34495e", "bg": "#f8f9fa", "border": "#4a5568"}
        }

        for idx, (ideology, stat) in enumerate(stats.items()):
            col = cols[idx]
            theme = theme_colors.get(ideology, {"text": "#2c3e50", "bg": "#ffffff"})
            diff_color = "#f16464" if stat['diff'] < 0 else "#4b7ed0" if stat['diff'] > 0 else "#7f8c8d"
            diff_sign = "+" if stat['diff'] > 0 else ""
            # 카드 배경색을 아주 연하게 처리하거나 흰색으로 유지하고, 왼쪽 선으로만 색상을 강조합니다.
            html = f"""
            <div style="
                background-color: {theme['bg']}; 
                border-left: 5px solid {theme['border']}; 
                border-radius: 8px; 
                padding: 15px 20px; 
                margin-bottom: 10px;
            ">
                <div style="color: {theme['text']}; font-weight: 700; font-size: 16px; text-transform: uppercase; letter-spacing: 0.5px;">
                    {ideology} 진영
                </div>
                <div style="display: flex; align-items: baseline; gap: 8px; margin: 10px 0;">
                    <span style="font-size: 20px; color: #a0aec0; font-weight: 500;">{stat['orig']:.1f}%</span>
                    <span style="font-size: 14px; color: #cbd5e0;">&rarr;</span>
                    <span style="font-size: 28px; font-weight: 800; color: #1a202c; line-height: 1;">{stat['ai']:.1f}%</span>
                </div>
                <div style="
                    display: inline-block;
                    color: {diff_color}; 
                    font-weight: 700; 
                    font-size: 15px;
                ">
                    {diff_sign}{stat['diff']:.1f}%p
                </div>
            </div>
            """
            col.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 개별 페르소나 응답 상세")

        with st.container(height=600):
            for _, row in df_merged.iterrows():
                def get_color(ideology):
                    if ideology == "진보": return "#73b9e8"
                    if ideology == "보수": return "#e86557"
                    return "#7f8c8d"

                color_orig = get_color(row['기존 성향'])
                color_new = get_color(row['AI 변화 성향'])

                card_html = f"""
                    <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #edf2f7; padding-bottom: 10px;">
                            <div style="font-size: 14px; color: #4a5568;">
                                <span style="font-weight: 700; color: #2d3748;">{row['persona_id']}</span> &nbsp;|&nbsp; {row['gender']} &nbsp;|&nbsp; {row['birth_year']}년생 &nbsp;|&nbsp; {row['region']}
                            </div>
                            <div style="font-size: 14px; font-weight: 700; background-color: #f8fafc; padding: 5px 12px; border-radius: 20px; border: 1px solid #e2e8f0;">
                                <span style="color: {color_orig}">{row['기존 성향']}</span><span style="color: #a0aec0; margin: 0 6px;">&rarr;</span><span style="color: {color_new}">{row['AI 변화 성향']}</span>
                            </div>
                        </div>
                        <div style="font-size: 15px; color: #2d3748; line-height: 1.6;">
                            {row['Reason']}
                        </div>
                    </div>
                    """
                st.markdown(card_html, unsafe_allow_html=True)