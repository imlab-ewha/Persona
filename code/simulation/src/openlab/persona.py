"""
persona.py
----------
페르소나 및 외부 환경 정보 통합 로딩 모듈.
- PostgreSQL DB(persona_profile, external_information)를 단일 소스로 사용합니다.
- 모든 속성은 NULL/NaN/빈 문자열일 경우 자동으로 프로필/컨텍스트 생성에서 제외됩니다.
- survey(JSON) 데이터와 사회경제 지표를 자연어 형태로 변환하여 LLM에 제공합니다.
"""

import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv
from src.openlab.conditioning import get_relevant_attributes

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

PARTIES = ['더불어민주당', '국민의힘', '무당층', '기타정당']

# ── [매핑 설정] 페르소나 프로필 레이블 ──────────────────────────────
DEMO_LABELS = {
    "birth_year": "출생년도",
    "gender": "성별",
    "marital_status": "혼인 상태",
    "household_income": "가구 총 소득",
    "education_level": "교육 수준",
    "occupation_type": "직업 유형",
    "household_member": "가족수",
    "has_health_insurance": "건강보험 유무",
    "region": "거주 지역",
    "residence_district": "거주 구",
    "political_ideology": "정치적 이데올로기",
    "party_affiliation": "지지 정당",
    "party_leaning": "정당 지지 성향",
    "political_interest": "정치적 관심도",
    "media_trust": "미디어 신뢰도",
    "issue_involvement": "이슈 관여도",
    "political_discussion_frequency": "정치 토론 빈도",
    "religion": "종교",
    "gender_discrimination_perception": "여성 차별 인식",
    "sexual_minority_discrimination_perception": "동성애자 차별 인식",
    "network_size": "네트워크 사이즈",
    "happiness_index": "행복지수",
    "life_satisfaction": "삶의 만족도",
    "media_usage": "미디어 이용"
}

# ── [매핑 설정] 외부 환경 지표 레이블 ──────────────────────────────
EXTERNAL_LABELS = {
    "economic_growth_rate": "경제성장률",
    "gini_coefficient": "지니계수",
    "unemployment_rate": "실업률",
    "crime_rate": "범죄율",
    "news_text": "주요 뉴스 이슈"
}

# ── 1. 페르소나 데이터 로딩 (동적 필터링) ──────────────────────────
def load_demographic_data(n_sample=None, random_seed=42, filter_condition=None, limit=None) -> pd.DataFrame:
    """
    persona_profile 테이블에서 filter_condition에 맞는 데이터를 로드합니다.
    예: filter_condition="party_leaning IS NOT NULL"
    """
    query = "SELECT * FROM public.persona_profile_test"

    if filter_condition and filter_condition.strip():
        query += f" WHERE {filter_condition}"
    # query += " ORDER BY persona_id ASC"
    query += " ORDER BY RANDOM()"

    if limit:
        query += f" LIMIT {limit};"
    else:
        query +=";"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if n_sample and n_sample < len(df):
        df = df.sample(n=n_sample, random_state=random_seed).reset_index(drop=True)

    print(f"  [데이터 로딩] {len(df)}명의 페르소나를 확보했습니다.")
    return df

# ── 2. 외부 환경 정보 로딩 ──────────────────────────────────────────
def load_external_data(engine) -> pd.DataFrame:
    """
    external_information 테이블의 모든 주차 데이터를 가져옵니다.
    """
    query = "SELECT * FROM public.external_information ORDER BY timepoint_id;"
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

# ── 3. 텍스트 변환 로직 (Null-Skip 적용) ────────────────────────────
def _is_valid(val):
    """값이 유효한지(Null/NaN/Empty 아님) 확인"""
    return pd.notna(val) and str(val).strip() != "" and str(val).lower() != "nan"

def build_combined_profile_text(row: pd.Series, relevant_attrs: list = None) -> str:
    """개별 페르소나의 자연어 프로필 생성"""
    for col, label in DEMO_LABELS.items():
        # 💡 relevant_attrs가 존재할 경우, 해당 리스트에 포함된 컬럼만 텍스트로 만듦
        if relevant_attrs and col not in relevant_attrs:
            continue

    parts = []
    # 기본 컬럼 처리
    for col, label in DEMO_LABELS.items():
        val = row.get(col)
        if _is_valid(val):
            if col == "has_health_insurance":
                val_str = "있음" if val is True or str(val).lower() == 'true' else "없음"
                parts.append(f"{label}: {val_str}")
            else:
                parts.append(f"{label}: {val}")

    # JSON Survey 데이터 처리
    survey = row.get("survey")
    if pd.notna(survey):
        if isinstance(survey, str):
            try: survey = json.loads(survey)
            except: survey = {}
        if isinstance(survey, dict):
            for q, a in survey.items():
                if _is_valid(a):
                    parts.append(f"{q}: {a}")

    return ", ".join(parts)

def build_external_context_text(row: pd.Series) -> str:
    """주차별 외부 환경 컨텍스트 생성"""
    parts = []
    for col, label in EXTERNAL_LABELS.items():
        val = row.get(col)
        if _is_valid(val):
            if col == "news_text":
                parts.append(f"[{label}]\n{val}")
            else:
                parts.append(f"{label}: {val}")

    return "\n\n".join(parts) if parts else "참조할 특이 외부 정보 없음"

# ── 4. 시뮬레이션용 데이터 구조체 변환 ───────────────────────────────
def build_personas(df: pd.DataFrame, query) -> list[dict]:
    """DB DataFrame을 시뮬레이션용 딕셔너리 리스트로 변환"""

    relevant_attrs = None
    if query:
        relevant_attrs = get_relevant_attributes(query)
        # 현재 condition.py는 빈 리스트를 반환하므로, 리스트가 비어있으면 전체 사용으로 간주
        if not relevant_attrs:
            relevant_attrs = None

    personas = []
    for _, row in df.iterrows():
        raw_party = row.get("party_leaning")
        valid_party = raw_party if _is_valid(raw_party) else None
        personas.append({
            "persona_id": str(row["persona_id"]),
            "profile": build_combined_profile_text(row, relevant_attrs=relevant_attrs),
            "region": f"{row.get('residence_region', '')} {row.get('residence_district', '')}".strip(),
            "party_leaning": valid_party,
            "source": row.get("source", "original"),
            "last_support": None
        })
    return personas