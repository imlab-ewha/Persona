import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import plotly.express as px

# 제공된 src 모듈 임포트
from src.persona import load_demographic_data, build_personas, build_external_context_text
from src.simulation import get_prev_week_external_data, ask_persona, load_prompt
from src.aggregate import get_dashboard_data

from openai import OpenAI
from anthropic import Anthropic

# ============================================================
# ⚙️ CONFIG (초기 설정)
# ============================================================
MODEL = "claude-sonnet-4-6" # 사용자가 지정한 모델명
PROMPT_VER = "v5"
FILTER_CONDITION = "party_leaning IS NOT NULL"

# 환경변수 로드 및 DB 엔진 설정
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
DB_USER = os.getenv("DB_USER", "pdp")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "persona")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# LLM 클라이언트 준비
if MODEL.startswith("claude"):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    provider = "anthropic"
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    provider = "openai"

prompt_templates = load_prompt(PROMPT_VER)

# ============================================================
# Streamlit UI 구성
# ============================================================
st.set_page_config(page_title="정당 지지율 시뮬레이터", page_icon="📊", layout="wide")

st.title("페르소나 기반 정당 지지율 시뮬레이션")
st.markdown("특정 기간의 여론 변화를 시뮬레이션합니다.")

# --- 1. 입력부 (Sidebar) ---
with st.sidebar:
    st.header("시뮬레이션 설정")
    
    # 1) 지역 (서울특별시 / 부산광역시)
    target_region = st.selectbox("지역", options=["서울특별시", "부산광역시"])
    
    # 2) 페르소나 명수
    num_personas = st.number_input("시뮬레이션 할 페르소나 명수", min_value=1, max_value=1000, value=10)
    
    # 3) 질의 (Query)
    custom_query = st.text_area("사건 입력", placeholder="발생 가능한 사건 및 전달할 메시지를 입력하세요.")
    
    run_btn = st.button("시뮬레이션 실행", use_container_width=True)

# --- 2. 실행부 ---
if run_btn:
    st.divider()
    
    # 내부적으로 사용할 시점 정보 (프롬프트에 주차 정보가 필요할 수 있으므로 텍스트만 유지)
    curr_info = "2026년 3월 4주차"
    current_tid = int(datetime.now().timestamp())
    
    # 💡 외부 데이터 조회 로직 완전히 삭제
    external_context = "" 
    
    # UI에 설정 정보 요약 표시 (외부 데이터 UI 삭제됨)
    st.subheader("시뮬레이션 정보 요약")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.info(f"**지역:** {target_region}\n\n**명수:** {num_personas}명")
    with info_col2:
        st.success(f"**질의:** {custom_query}")

    # [Step 3] 페르소나 로딩
    st.write("---")
    status_text = st.empty()
    status_text.write(f"{target_region} 페르소나 데이터를 불러오는 중 입니다...")
    
    region_map = {
        "서울특별시": "서울",
        "부산광역시": "부산"
    }

    # 지역 필터링 적용 (서울특별시 -> 서울, 부산광역시 -> 부산)
    region_keyword = region_map.get(target_region, "서울")
    
    # 매핑된 키워드로 DB 필터링 (예: residence_region LIKE '%%부산%%')
    dynamic_filter = f"party_leaning IS NOT NULL AND residence_region LIKE '%%{region_keyword}%%'"
    
    demo_df = load_demographic_data(n_sample=None, random_seed=42, filter_condition=dynamic_filter, limit=num_personas)
    
    if demo_df.empty:
        st.error(f"조건에 맞는 {target_region} 페르소나가 없습니다. DB를 확인해주세요.")
        st.stop()
        
    personas = build_personas(demo_df)
    
    # [Step 4] 시뮬레이션 실행 (UI 프로그레스 바 연동)
    status_text.write("AI 시뮬레이션을 진행하고 있습니다.")
    progress_bar = st.progress(0)
    
    aggregated_responses = {}

    for i, persona in enumerate(personas):
        res_obj, original_query_key = ask_persona(
            client=client, 
            persona=persona, 
            week_info=curr_info, 
            external_context=external_context,
            prev_support="", 
            prompt_templates=prompt_templates, 
            model=MODEL, 
            provider=provider,
            query=custom_query
        )
        
        p_id = persona["persona_id"]
        if p_id not in aggregated_responses:
            aggregated_responses[p_id] = {}
            
        # DB에 저장할 때는 UI에서 입력받은 custom_query를 키(Key)로 덮어씌움
        aggregated_responses[p_id][custom_query] = res_obj
        
        # 프로그레스 바 업데이트
        progress_bar.progress((i + 1) / len(personas))

    # [Step 5] DB에 시뮬레이션 결과 저장
    result_df = pd.DataFrame([{
        "timepoint_id": current_tid,
        "response": json.dumps(aggregated_responses, ensure_ascii=False),
        "timestamp": pd.Timestamp.now()
    }])
    result_df.to_sql("persona_response_history", engine, if_exists='append', index=False, schema='public')

    status_text.success("시뮬레이션 완료!")
    status_text.empty() # 상태 텍스트 지우기

    # ============================================================
    # 📊 [Step 6] 결과 대조 및 대시보드 시각화 (새로 추가된 로직)
    # ============================================================
    st.divider()
    st.markdown("#### 정치적 지지율 변동 예측")
    
    # src/aggregate.py의 함수를 호출하여 통계(stats)와 상세데이터(df_merged)를 바로 가져옴
    stats, df_merged = get_dashboard_data(current_tid)

    if stats and df_merged is not None:
        # 1. 지표 카드 (동적 생성)
        cols = st.columns(len(stats))
        
        # 이념별 색상 테마 (없으면 기본값 적용)
        theme_colors = {
            "진보": {"text": "#66bef8dd", "bg": "#f0f8ff"}, # 파란색 톤
            "보수": {"text": "#f98a7deb", "bg": "#fff0f0"}, # 붉은색 톤
            "중도": {"text": "#323232", "bg": "#d8d6d6"}, # 회색 톤
            "기타/중도": {"text": "#34495e", "bg": "#f8f9fa"}
        }
        
        for idx, (ideology, stat) in enumerate(stats.items()):
            col = cols[idx]
            theme = theme_colors.get(ideology, {"text": "#2c3e50", "bg": "#ffffff"})
            
            diff_color = "#ff9286" if stat['diff'] < 0 else "#aff7ee" if stat['diff'] > 0 else "#7f8c8d"
            diff_sign = "+" if stat['diff'] > 0 else ""
            
            # HTML/CSS를 활용한 KPI 카드 디자인 (보현님이 올려주신 이미지 스타일)
            html = f"""
            <div style="background-color: {theme['bg']}; border: 1px solid {theme['text']}40; border-radius: 10px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="color: {theme['text']}; font-weight: 700; font-size: 16px; margin-bottom: 12px;">{ideology} 진영</div>
                <div style="font-size: 32px; font-weight: 800; color: #2c3e50; margin-bottom: 8px;">
                    <span style="font-size: 18px; color: #95a5a6; font-weight: 500;">{stat['orig']:.1f}% &rarr; </span>
                    {stat['ai']:.1f}%
                </div>
                <div style="color: {diff_color}; font-weight: 700; font-size: 16px;">
                    {diff_sign}{stat['diff']:.1f}%p
                </div>
            </div>
            """
            col.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. 깔끔한 개별 페르소나 응답 테이블
        st.markdown("#### 개별 페르소나 응답 상세")
        
        # Streamlit column_config를 활용해 Reason 컬럼이 짤리지 않게 넓게 표시
        with st.container(height=600):
            for _, row in df_merged.iterrows():
                
                # 성향별 색상 매핑
                def get_color(ideology):
                    if ideology == "진보": return "#73b9e8"
                    if ideology == "보수": return "#e86557"
                    return "#7f8c8d" # 기타/중도
                
                color_orig = get_color(row['기존 성향'])
                color_new = get_color(row['AI 변화 성향'])
                
                # 💡 HTML/CSS로 세련된 카드 레이아웃 구성
                card_html = f"""
                    <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #edf2f7; padding-bottom: 10px;">
                            <div style="font-size: 14px; color: #4a5568;">
                                <span style="font-weight: 700; color: #2d3748;">👤 {row['persona_id']}</span> &nbsp;|&nbsp; {row['gender']} &nbsp;|&nbsp; {row['birth_year']}년생 &nbsp;|&nbsp; {row['region']}
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
    else:
        st.warning("분석할 시뮬레이션 데이터를 찾지 못했습니다.")