import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import ast
import re
import plotly.graph_objects as go
from datetime import datetime
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# ── [1] 경로 및 환경 변수 설정 ──
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))

project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 사용자 정의 모듈 로드
from src.anes_utils import preprocess_anes_2024
from src.persona import build_personas
from src.simulation2 import simulate_week, load_prompt
from src.validation import calculate_validation_metrics

# ── [2] 전역 설정 및 설문 데이터 딕셔너리 ──
CSV_PATH = "/home/imlab/Persona/data/validation/ANES_TimeSeries/anes_timeseries_cdf_csv_20260205.csv"
MODEL = "claude-sonnet-4-6" 
PROMPT_VER = "v10_simulation"

st.set_page_config(page_title="ANES Validation Dashboard", layout="wide")

# [핵심] 딕셔너리 키를 짧게 변경하여 Selectbox에서 ... 으로 잘리는 현상 방지
SURVEY_QUESTIONS = {
    "Federal Spending (Poor)": {
        "target": "VCF0886",
        "query": "aid to the poor",
        "options": ["1. Increased", "2. Same", "3. Decreased", "8. DK"]
    },
    "Role of Government": {
        "target": "VCF9131",
        "query": "Next, I am going to ask you to choose which of two statements I read comes closer to your own opinion. You might agree to some extent with both, but we want to know which one is closer to your views: ONE, the less government the better; or TWO, there are more things that government should be doing",
        "options": ["1. Less government the better", "2. More things government should be doing", "8. DK"]
    }
}

# ── [3] 동적 데이터 로드 및 필터링 ──
@st.cache_data
def get_processed_data(target_col, opt_tuple):
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
        return pd.DataFrame()
    df = preprocess_anes_2024(CSV_PATH)
    if len(df) == 0: return pd.DataFrame()
    
    valid_values = []
    for opt in opt_tuple:
        try:
            valid_values.append(float(opt.split('.')[0]))
        except: pass
        
    df[target_col] = pd.to_numeric(df.get(target_col, np.nan), errors='coerce')
    valid_df = df[df[target_col].isin(valid_values)].copy()
    return valid_df

# ── [4] 사이드바 설정 ──
with st.sidebar:
    st.header("시뮬레이션 설정")
    
    selected_topic = st.selectbox("분석할 설문 주제 선택", list(SURVEY_QUESTIONS.keys()))
    TARGET_VCF = SURVEY_QUESTIONS[selected_topic]["target"]
    query = SURVEY_QUESTIONS[selected_topic]["query"]
    options = SURVEY_QUESTIONS[selected_topic]["options"]
    
    # 튜플로 변환하여 캐시 해시 충돌 방지
    df_anes_filtered = get_processed_data(TARGET_VCF, tuple(options))
    
    st.divider()
    
    if len(df_anes_filtered) > 0:
        max_v = len(df_anes_filtered)
        sample_n = st.number_input("추출 페르소나 인원", min_value=1, max_value=max_v, value=min(10, max_v))
    else:
        st.error(f"조건({TARGET_VCF} in {options})에 맞는 데이터가 없습니다.")
        st.stop()
    
    final_api_key = os.getenv("ANTHROPIC_API_KEY") if "claude" in MODEL else os.getenv("OPENAI_API_KEY")
    
    st.markdown("""
        <style>
        button[kind="primary"] {
            background-color: #63B3ED !important;
            color: white !important;
            border: none !important;
        }
        button[kind="primary"]:hover {
            background-color: #3182CE !important;
        }
        </style>
    """, unsafe_allow_html=True)
        
    run_simulation = st.button("시뮬레이션 시작", use_container_width=True, type="primary")

# ── [5] 메인 화면 레이아웃 ──
st.title("페르소나 검증 성능 리포트")

# 선택된 문항을 깔끔하게 화면에 노출
st.markdown("#### 분석 문항")
options_html = "".join([
    f'<span style="background-color: #F7FAFC; color: #4A5568; padding: 5px 12px; border-radius: 15px; font-size: 13px; font-weight: 500; border: 1px solid #E2E8F0;">{opt}</span>' 
    for opt in options
])

# 질문과 선택지를 담은 깔끔한 카드 UI
question_card_html = f"""
<div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; background-color: #FFFFFF; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
    <div style="font-size: 12px; font-weight: 700; color: #718096; margin-bottom: 8px; letter-spacing: 0.5px;">
        TARGET QUESTION
    </div>
    <div style="font-size: 16px; color: #2D3748; line-height: 1.6; font-weight: 500; margin-bottom: 16px;">
        {query}
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
        <span style="font-size: 12px; font-weight: 700; color: #A0AEC0; margin-right: 4px;">OPTIONS :</span>
        {options_html}
    </div>
</div>
"""
st.markdown(question_card_html, unsafe_allow_html=True)
st.divider()

if run_simulation:
    if not final_api_key:
        st.error("서버에 API Key(.env)가 설정되어 있지 않습니다.")
        st.stop()

    today_str = datetime.now().strftime("%Y-%m-%d")
    sampled_df = df_anes_filtered.sample(n=sample_n)
    
    client = Anthropic(api_key=final_api_key) if "claude" in MODEL else OpenAI(api_key=final_api_key)
    prompt_templates = load_prompt(PROMPT_VER)

    with st.status(f"시뮬레이션 진행 중... ({today_str})", expanded=True) as status:
        personas = build_personas(sampled_df, query=query)
        progress_bar = st.progress(0)
        
        results = simulate_week(
            client=client, personas=personas, target_info=today_str, 
            external_context=None, prompt_templates=prompt_templates,
            model=MODEL, query=query, options=options, engine=None,
            provider="anthropic" if "claude" in MODEL else "openai",
            st_bar=progress_bar, use_news_thinking=False
        )

        status.update(label="시뮬레이션 및 데이터 집계 완료!", state="complete")

    # ── [6] 전체 요약 지표 (6개 지표 100% 노출) ──
    all_act = {opt: 0 for opt in options}
    all_pre = {opt: 0 for opt in options}
    
    for _, row in sampled_df.iterrows():
        # 1. 실제 데이터(Actual) 집계
        val = row[TARGET_VCF]
        for opt in options:
            if opt.startswith(str(int(val)) + "."):
                all_act[opt] += 1
                break
                
        # 2. 시뮬레이션 데이터(Predicted) 집계
        p_id = row['persona_id']
        p_res = results.get(p_id, {}).get(query, {}).get("Result", {})
        
        # 💡 [핵심 수정] 100%를 찾는 대신, 가장 높은 확률을 가진 옵션을 승자로 선택
        try:
            # % 기호가 있으면 제거하고 숫자로 변환하여 최대값 키 추출
            winner = max(p_res, key=lambda k: float(str(p_res[k]).replace('%','').strip() or 0))
            # 모든 값이 0이면 미응답 처리
            if float(str(p_res[winner]).replace('%','').strip() or 0) == 0:
                winner = "미응답"
        except:
            winner = "미응답"

        if winner in all_pre:
            all_pre[winner] += 1
    
    # 지표 계산 실행
    global_m = calculate_validation_metrics(all_act, all_pre)
    
    # 💡 이제 데이터가 채워졌으므로 아래 블록이 정상 작동합니다.
    if global_m:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("JS Distance", global_m["JS_Distance"])
        m_col2.metric("KL Divergence", global_m["KL_Divergence"])
        m_col3.metric("Cramer's V", global_m["Cramer_V"])
        m_col4.metric("Cronbach's Alpha", global_m["Cronbach_Alpha"])
        
        m_col5, m_col6, m_col7, m_col8 = st.columns(4)
        m_col5.metric("Entropy (Actual)", global_m["Entropy_Actual"])
        m_col6.metric("Entropy (Predicted)", global_m["Entropy_Predicted"])
        m_col7.metric("총 샘플 수", f"{sample_n}명")

    # ── [7] 5개 집단별 시각화 ──
    st.divider()
    st.subheader("정당 지지 집단별 세부 비교")
    vcf0302_groups = ["Republican", "Independent", "No preference", "Other", "Democrat"]
    
    x_labels = [opt.split('.')[1].strip()[:5] if '.' in opt else opt[:5] for opt in options]
    
    g_cols = st.columns(5)
    for idx, group_name in enumerate(vcf0302_groups):
        with g_cols[idx]:
            st.markdown(f"**{group_name}**")
            sub_df = sampled_df[sampled_df['Initial Party Preference'] == group_name]
            
            if len(sub_df) > 0:
                act_d = {opt: 0 for opt in options}
                pred_d = {opt: 0 for opt in options}
                
                for _, row in sub_df.iterrows():
                    val = row[TARGET_VCF]
                    for opt in options:
                        if opt.startswith(str(int(val)) + "."): act_d[opt] += 1
                        
                    p_id = row['persona_id']
                    p_res = results.get(p_id, {}).get(query, {}).get("Result", {})
                    
                    # 💡 [수정] 100% 대신 가장 높은 확률을 가진 옵션을 승자로 선택
                    try:
                        winner = max(p_res, key=lambda k: float(str(p_res[k]).replace('%','').strip() or 0))
                        if float(str(p_res[winner]).replace('%','').strip() or 0) == 0: winner = "미응답"
                    except: winner = "미응답"
                    
                    if winner in pred_d: pred_d[winner] += 1

                g_m = calculate_validation_metrics(act_d, pred_d)

                fig = go.Figure(data=[
                    go.Bar(name='Actual', x=x_labels, y=list(act_d.values()), marker_color="#D4D4D4"),
                    go.Bar(name='Predicted', x=x_labels, y=list(pred_d.values()), marker_color='#90CDF4')
                ])
                fig.update_layout(barmode='group', height=180, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig, width="stretch", key=f"chart_{group_name}_{idx}")

                if g_m:
                    metrics_html = f"""
                    <div style="font-size: 11px; color: #4A5568; line-height: 1.4; background-color: #F7FAFC; padding: 8px; border-radius: 5px;">
                        <b>JSD:</b> {g_m['JS_Distance']} | <b>KLD:</b> {g_m['KL_Divergence']}<br>
                        <b>Cramer:</b> {g_m['Cramer_V']} | <b>Alpha:</b> {g_m['Cronbach_Alpha']}<br>
                        <b>Ent(A):</b> {g_m['Entropy_Actual']} | <b>Ent(P):</b> {g_m['Entropy_Predicted']}
                    </div>
                    """
                    st.markdown(metrics_html, unsafe_allow_html=True)
            else:
                st.info("데이터 없음")

    # ── [8] 개별 페르소나 응답 상세 (수정본) ──
    st.divider()
    st.markdown("#### 개별 페르소나 응답 상세")
    
    party_theme = {
        "Republican": {"color": "#4A5568"},
        "Democrat": {"color": "#3182CE"},
        "Independent": {"color": "#718096"},
        "No preference": {"color": "#A0AEC0"},
        "Other": {"color": "#4A5568"}
    }
    dynamic_colors = ["#2B6CB0", "#3182CE", "#4299E1", "#63B3ED", "#90CDF4"]

    with st.container(height=600):
        for p in personas:
            p_id = p.get("persona_id", "Unknown")
            res_content = results.get(p_id, {}).get(query, {})
            p_res = res_content.get("Result", {})
            
            # 💡 [수정] 가장 높은 확률 옵션 추출
            try:
                winner = max(p_res, key=lambda k: float(str(p_res[k]).replace('%','').strip() or 0))
                if float(str(p_res[winner]).replace('%','').strip() or 0) == 0: winner = "미응답"
            except: winner = "미응답"
            
            matching_rows = sampled_df[sampled_df['persona_id'] == p_id]
            if matching_rows.empty: continue
            
            p_info = matching_rows.iloc[0]
            gender_val = str(p_info.get('gender', 'Unknown'))
            party_val = str(p_info.get('Initial Party Preference', 'Other'))
            b_year_val = f"{int(p_info.get('birth_year', 0))}년생" if pd.notna(p_info.get('birth_year')) else "미상"
            
            color_orig = party_theme.get(party_val, {"color": "#718096"})['color']
            
            # 💡 인덱스 에러 방지
            try: opt_idx = options.index(winner)
            except: opt_idx = 0
            color_new = dynamic_colors[opt_idx % len(dynamic_colors)]
            
            # 💡 [추가] 확률 분포를 보여주는 칩(Chips) 생성
            prob_chips = "".join([
                f"<span style='background:#f1f5f9; color:#475569; padding:2px 6px; border-radius:4px; font-weight:700; margin-right:8px; font-size:11px;'>"
                f"{k[:8]} {v}</span>" 
                for k, v in p_res.items() if str(v) != "0%" and str(v) != "0"
            ])
            
            safe_reason = str(res_content.get('Reason', '설명 없음')).replace('{', '&#123;').replace('}', '&#125;')
            
            card_html = f"""
                <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #edf2f7; padding-bottom: 8px;">
                        <div style="font-size: 14px; color: #4a5568;">
                            <span style="font-weight: 700; color: #2d3748;">ID: {p_id}</span> &nbsp;|&nbsp; {gender_val} &nbsp;|&nbsp; {b_year_val} &nbsp;|&nbsp; <span style="color: {color_orig}; font-weight: 700;">{party_val}</span>
                        </div>
                        <div style="font-size: 13px; font-weight: 700;">
                            <span style="color: {color_new}">{winner}</span>
                        </div>
                    </div>
                    <div style="margin-bottom: 12px;">{prob_chips}</div>
                    <div style="font-size: 14px; color: #2d3748; line-height: 1.5; border-left: 3px solid #edf2f7; padding-left: 15px;">{safe_reason}</div>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)