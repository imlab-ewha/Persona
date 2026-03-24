"""
generate_synthetic.py
---------------------
- 통계청 인구 분포(IPF 개념)와 베이지안 네트워크(BN)를 사용하여 Synthetic 페르소나 생성.
- 원본(서울) 데이터와 생성된 Synthetic 데이터를 합쳐 `persona_profile_test` 테이블에 적재.
"""

import os
import uuid
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator, ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork

# ── DB 설정 및 엔진 생성 ─────────────────────────────────────────
DB_USER = os.getenv("DB_USER", "pdp")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "persona")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def load_region_original_data(region_kw: str) -> pd.DataFrame:
    """특정 지역(서울 또는 부산) 원본 데이터를 로드합니다."""
    query = f"SELECT * FROM public.persona_profile WHERE residence_region LIKE '%%{region_kw}%%';"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    print(f"✅ [{region_kw} 데이터 로드] {len(df)}명 확보")
    return df

def get_target_population_distribution(csv_path: str, region_name: str, current_year: int = 2026) -> pd.DataFrame:
    """CSV를 읽어 특정 지역의 성별 및 '출생년도'별 타겟 인구 비율을 추출합니다."""
    df_pop = pd.read_csv(csv_path, encoding='cp949')
    # 💡 행정구역 명칭(서울특별시, 부산광역시 등)으로 필터링
    df_region = df_pop[df_pop['행정구역'].str.contains(region_name)].iloc[0]
    
    pop_data = []
    for gender_str, gender_val in [('남', '남성'), ('여', '여성')]:
        for age in range(101):
            col_suffix = f"{age}세" if age < 100 else "100세 이상"
            col_name = f"2026년02월_{gender_str}_{col_suffix}"
            
            if col_name in df_region:
                count_str = str(df_region[col_name]).replace(',', '')
                count = int(count_str) if count_str.isdigit() else 0
                birth_year = current_year - age
                pop_data.append({
                    'gender': gender_val,
                    'birth_year': birth_year,
                    'target_count': count
                })
                
    target_df = pd.DataFrame(pop_data)
    target_df['target_ratio'] = target_df['target_count'] / target_df['target_count'].sum()
    print(f"✅ [인구 통계] {region_name} 타겟 분포 계산 완료")
    return target_df

def balance_data_by_marginals(original_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """원본 데이터의 성별/출생년도를 타겟 분포에 맞춰 가중치 리샘플링을 진행합니다."""
    original_df['gender'] = original_df['gender'].astype(str).str.strip()
    original_df['birth_year'] = pd.to_numeric(original_df['birth_year'], errors='coerce')
    
    original_stats = original_df.groupby(['gender', 'birth_year']).size().reset_index(name='orig_count')
    original_stats['orig_ratio'] = original_stats['orig_count'] / original_stats['orig_count'].sum()
    
    merged_stats = pd.merge(original_stats, target_df, on=['gender', 'birth_year'], how='left')
    merged_stats['weight'] = np.where(
        merged_stats['orig_ratio'] > 0, 
        merged_stats['target_ratio'] / merged_stats['orig_ratio'], 
        0
    )
    merged_stats['weight'] = merged_stats['weight'].fillna(0)
    
    df_weighted = pd.merge(original_df, merged_stats[['gender', 'birth_year', 'weight']], on=['gender', 'birth_year'], how='left')
    df_weighted['weight'] = df_weighted['weight'].fillna(0)
    df_weighted['weight'] = np.where(df_weighted['weight'] <= 0, 0.0001, df_weighted['weight'])
    
    sample_size = len(original_df)
    balanced_df = df_weighted.sample(n=sample_size, replace=True, weights='weight', random_state=42).reset_index(drop=True)
    
    print(f"✅ [분포 보정] IPF 가중치 기반 {len(balanced_df)}건 리샘플링 완료")
    return balanced_df

def generate_synthetic_personas(train_df: pd.DataFrame, original_columns: list, num_samples: int, region_code: str) -> pd.DataFrame:
    """
    [핵심] IPF로 보정된 train_df(CSV 비율 + DB 속성)를 사용하여 
    Raw 데이터 기반의 BN을 학습합니다. (논문 기반 Whitelist / Blacklist 적용)
    """
    # 1. 학습에 사용할 순수 원본 컬럼만 추출
    exclude_cols = ['persona_id', 'source']
    train_cols = [c for c in original_columns if c in train_df.columns and c not in exclude_cols]
    
    # 2. 데이터 준비 (Raw 값을 문자열로 인식시켜 범주형 확률 계산)
    bn_train_data = train_df[train_cols].copy()
    for col in bn_train_data.columns:
        bn_train_data[col] = bn_train_data[col].astype(str).replace(['nan', 'None', '<NA>'], 'NaN')
    
    # ── [논문 반영] Whitelist (fixed_edges) 및 Blacklist 설정 ──
    # pgmpy의 HillClimbSearch에서는 fixed_edges가 Whitelist, black_list가 Blacklist 역할을 합니다.
    
    whitelist_edges = []
    blacklist_edges = []
    
    # 논문 명시 관계 1: 나이는 결혼 상태, 교육 수준, 직업에 영향을 미침
    if 'birth_year' in train_cols:
        if 'marital_status' in train_cols:
            whitelist_edges.append(('birth_year', 'marital_status'))
            blacklist_edges.append(('marital_status', 'birth_year')) # 역방향 차단
        if 'education' in train_cols:
            whitelist_edges.append(('birth_year', 'education'))
            blacklist_edges.append(('education', 'birth_year')) # 역방향 차단
        if 'occupation' in train_cols:
            whitelist_edges.append(('birth_year', 'occupation'))
            blacklist_edges.append(('occupation', 'birth_year')) # 역방향 차단
            
        # 논문 명시 관계 2: 나이와 성별은 서로 영향을 주지 않음
        if 'gender' in train_cols:
            blacklist_edges.append(('birth_year', 'gender'))
            blacklist_edges.append(('gender', 'birth_year'))

    # 논문 명시 관계 3: 가구원 수(가족수)는 주거 형태/크기에 영향을 미침
    if 'household_size' in train_cols and 'housing_size' in train_cols:
        whitelist_edges.append(('household_size', 'housing_size'))
        blacklist_edges.append(('housing_size', 'household_size')) # 역방향 차단

    # (선택) 추가적인 상식적 Blacklist (결과물인 '현재 태도/인식'이 '타고난 인구통계'를 바꾸지 못하게 함)
    # 예: 정당 지지 성향이 출생년도나 성별을 결정할 수 없음
    attitude_cols = ['political_interest', 'party_leaning', 'happiness_index', 'life_satisfaction']
    demographic_cols = ['birth_year', 'gender']
    for ac in attitude_cols:
        for dc in demographic_cols:
            if ac in train_cols and dc in train_cols:
                blacklist_edges.append((ac, dc))

    expert_knowledge = ExpertKnowledge(
        required_edges=whitelist_edges,
        forbidden_edges=blacklist_edges
    )

    # 3. BN 구조 학습 (Raw 값 사이의 인과관계 파악)
    print(f"⏳ [BN 학습] {len(train_cols)}개 Raw 컬럼 기반으로 시뮬레이션 모델 구축 중...")
    hc = HillClimbSearch(bn_train_data)
    
    # 모델 추정 시 fixed_edges와 black_list 파라미터 전달
    best_model_structure = hc.estimate(
        scoring_method=BIC(bn_train_data), # pgmpy 최신 버전에서는 BicScore 사용
        expert_knowledge=expert_knowledge
    )
    
    # 4. 파라미터 학습 및 샘플링
    model = DiscreteBayesianNetwork(best_model_structure.edges())
    model.add_nodes_from(bn_train_data.columns)
    model.fit(bn_train_data, estimator=MaximumLikelihoodEstimator)
    
    # CSV 인구 비율이 녹아든 모델로부터 새로운 페르소나 추출
    print(f"⏳ [데이터 생성] {num_samples}건의 가상 페르소나 샘플링 중...")
    synthetic_df = model.simulate(n_samples=num_samples, seed=42)
    synthetic_df = synthetic_df.replace('NaN', np.nan)
    
    # 식별자 및 출처 추가
    synthetic_df['persona_id'] = [f"{region_code}_{uuid.uuid4().hex[:6]}" for _ in range(num_samples)]
    synthetic_df['source'] = 'synthetic'

    target_cols = original_columns.copy()
    if 'source' not in target_cols:
        target_cols.append('source')

    # 학습에서 제외된 컬럼 빈 값 처리
    for col in exclude_cols:
        if col not in synthetic_df.columns and col in target_cols:
            synthetic_df[col] = np.nan
            
    # 정해진 컬럼 순서대로 정렬 (source 포함)
    synthetic_df = synthetic_df[target_cols]
    print(f"✅ [데이터 생성] Synthetic 페르소나 {len(synthetic_df)}건 생성 완료")
    
    return synthetic_df

def load_to_test_db(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, mode='append'):
    """원본과 합성 데이터를 병합하여 DB에 적재합니다. (mode: 'replace' 또는 'append')"""
    original_df['source'] = 'original'
    if 'weight' in original_df.columns:
        original_df = original_df.drop(columns=['weight'])
        
    final_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    if 'birth_year' in final_df.columns:
        # 💡 [수정] col -> 'birth_year'로 직접 지정해 줍니다.
        final_df['birth_year'] = pd.to_numeric(final_df['birth_year'], errors='coerce').astype('Int64')
            
    # 2. 실수형 (여기는 for col in float_cols 루프가 있어서 괜찮습니다)
    float_cols = ['has_health_insurance', 'political_ideology', 'party_affiliation', 'political_interest', 'media_trust', 'issue_involvement', 'political_discussion_frequency', 'religion', 'gender_discrimination_perception', 'sexual_minority_discrimination_perception', 'happiness_index', 'life_satisfaction', 'survey']
    for col in float_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype(float)
            
    # 3. 텍스트형 (여기도 루프가 있어서 괜찮습니다)
    text_cols = ['persona_id', 'gender', 'marital_status', 'education', 'occupation', 'residence_region', 'household_size', 'household_members', 'household_income', 'housing_type', 'housing_ownership', 'housing_size', 'residence_district', 'party_leaning', 'network_size', 'media_usage', 'source']
    for col in text_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)
            
    print(f"⏳ [DB 적재] {len(final_df)}건 데이터를 persona_profile_test에 {mode} 중...")
    try:
        # 💡 [중요] if_exists=mode 를 통해 replace/append 결정
        final_df.to_sql('persona_profile_test', engine, if_exists=mode, index=False, schema='public')
        print(f"✅ [DB 적재] {mode} 완료!")
    except Exception as e:
        print(f"❌ [DB 적재 실패] 에러 발생: {e}")

    
if __name__ == "__main__":
    csv_path = '/home/imlab/Persona/data/etc/202602_202602_연령별인구현황_월간.csv'
    
    # 💡 [수정] 각 튜플에 'SE'와 'BS'를 추가하여 3개로 만듭니다.
    regions = [
        ('서울', '서울특별시', 'SE'), # kw, full_name, code
        ('부산', '부산광역시', 'BS')  # kw, full_name, code
    ]
    
    for i, (kw, full_name, code) in enumerate(regions):
        print(f"\n🚀 [{full_name}] 프로세스 시작")
        
        # 1. 데이터 로드
        df_original = load_region_original_data(kw)
        if df_original.empty:
            print(f"⚠️ {full_name} 데이터가 없어 건너뜁니다.")
            continue
            
        original_columns = df_original.columns.tolist()
        
        # 2. 분포 계산 및 리샘플링
        df_target_pop = get_target_population_distribution(csv_path, full_name)
        df_balanced = balance_data_by_marginals(df_original, df_target_pop)
        
        # 3. Synthetic 생성
        df_synthetic = generate_synthetic_personas(df_balanced, original_columns, num_samples=len(df_original), region_code=code)
        
        # 4. DB 적재 (첫 번째 지역이면 replace, 그 다음부터는 append)
        db_mode = 'replace' if i == 0 else 'append'
        load_to_test_db(df_original, df_synthetic, mode=db_mode)

    print("\n✨ 모든 지역(서울, 부산) 프로세스가 완료되었습니다.")