import pandas as pd
import numpy as np

# 1. ANES 코드 -> 자연어 값 상세 매핑
ANES_VALUE_MAP = {
    "VCF0104": {1: "Male", 2: "Female", 3: "Other"},
    "VCF0105": {1: "White non-Hispanic", 2: "Black non-Hispanic", 3: "Asian or Pacific Islander", 4: "American Indian or Alaska Native", 5: "Hispanic"},
    "VCF0110": {1: "Grade school or less", 2: "High school", 3: "Some college", 4: "College or advanced degree"},
    "VCF0114": {1: "0 to 16 percentile", 2: "17 to 33 percentile", 3: "34 to 67 percentile", 4: "68 to 95 percentile", 5: "96 to 100 percentile"},
    "VCF0112": {1: "Northeast", 2: "North Central", 3: "South", 4: "West"},
    "VCF0116": {1: "Working now", 2: "Temporarily laid off", 4: "Unemployed", 5: "Retired", 6: "Permanently disabled", 7: "Homemaker", 8: "Student"},
    "VCF0128": {1: "Protestant", 2: "Catholic", 3: "Jewish", 4: "Other and none", 0: "None"},
    "VCF0147": {1: "Married", 2: "Never married", 3: "Divorced", 4: "Separated", 5: "Widowed", 7: "Partners; not married"},
    "VCF0148": {0: "Lower class", 1: "Average working", 2: "Working-NA", 3: "Upper working", 4: "Average middle", 5: "Middle class-NA", 6: "Upper middle", 7: "Upper class"},
    "VCF0146": {1: "Yes, own", 2: "No, not owned"},
    "VCF0301": {1: "Strong Democrat", 2: "Weak Democrat", 3: "Independent-Democrat", 4: "Independent", 5: "Independent-Republican", 6: "Weak Republican", 7: "Strong Republican"},
    "VCF0302": {1: "Republican", 2: "Independent", 3: "No preference", 4: "Other", 5: "Democrat"},
    "VCF0803": {1: "Extremely liberal", 2: "Liberal", 3: "Slightly liberal", 4: "Moderate", 5: "Slightly conservative", 6: "Conservative", 7: "Extremely conservative"},
    "VCF0310": {1: "Not much interested", 2: "Somewhat interested", 3: "Very much interested"},
    "VCF0605": {1: "Few big interests", 2: "Benefit of all"},
    "VCF0700": {1: "Democratic candidate", 2: "Republican candidate", 7: "Other candidate"},
    "VCF0374": {1: "Yes", 5: "No"},
    "VCF0380": {1: "Yes", 5: "No"},
    "VCF0386": {1: "Yes", 5: "No"},
    "VCF0392": {1: "Yes", 5: "No"}  
}

COLUMN_MAPPING = {
    "VCF0101": "birth_year",
    "VCF0104": "gender",
    "VCF0105": "race",
    "VCF0147": "marital_status",
    "VCF0114": "household_income",
    "VCF0110": "education_level",
    "VCF0116": "occupation_type",
    "VCF0112": "region",
    "VCF0803": "political_ideology",
    "VCF0301": "party_leaning",
    "VCF0310": "political_interest",
    "VCF0128": "religion",
    "VCF0148": "social_class",
    "VCF0146": "home_ownership",
    "VCF0302": "Initial Party Preference"
}

def preprocess_anes_2024(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df_2024 = df[df['VCF0004'] == 2024].copy()
    
    # 1단계: 수치형 변환 및 시스템 결측치(0, 8, 9, 99) null 처리
    vcf_cols = list(ANES_VALUE_MAP.keys()) + ["VCF0201", "VCF0212", "VCF0218", "VCF0224", "VCF0101"]
    for col in vcf_cols:
        if col in df_2024.columns:
            df_2024[col] = pd.to_numeric(df_2024[col], errors='coerce')
            df_2024[col] = df_2024[col].replace([0, 8, 9, 99], np.nan)

    # 2단계: 텍스트 매핑 (매핑되지 않는 값은 자동으로 NaN 유지)
    for col, mapping in ANES_VALUE_MAP.items():
        if col in df_2024.columns:
            df_2024[col] = df_2024[col].map(mapping)

    # 3단계: Survey JSON 생성 (NaN을 제외하고 딕셔너리 빌드)
    def create_survey_json(row):
        survey_dict = {}
        # 온도계 수치 처리 (NaN이면 int 변환을 건너뜀)
        thermo_map = {"VCF0201": "Liberals", "VCF0212": "Conservatives", "VCF0218": "Democrats", "VCF0224": "Republicans"}
        for col, label in thermo_map.items():
            val = row.get(col)
            if pd.notna(val):
                # float 형태일 수 있으므로 안전하게 int로 변환
                survey_dict[f"Feeling Thermometer ({label})"] = f"{int(float(val))} degrees"
        
        eval_map = {
            "VCF0700": "Presidential Election Forecast",
            "VCF0374": "Likes Democratic Party", "VCF0380": "Dislikes Democratic Party",
            "VCF0386": "Likes Republican Party", "VCF0392": "Dislikes Republican Party",
            "VCF0605": "Gov View", "VCF0302": "Initial Party Preference"
        }
        for col, label in eval_map.items():
            val = row.get(col)
            if pd.notna(val):
                survey_dict[label] = val
            
        return survey_dict

    df_2024['survey'] = df_2024.apply(create_survey_json, axis=1)
    
    # 4단계: 컬럼명 변경 및 ID 생성
    df_2024 = df_2024.rename(columns=COLUMN_MAPPING)
    if 'birth_year' in df_2024.columns:
        # Age가 NaN이면 birth_year도 NaN이 됨 (행은 유지)
        df_2024['birth_year'] = 2026 - df_2024['birth_year']
        
    df_2024['persona_id'] = [str(i) for i in range(len(df_2024))]
    
    return df_2024