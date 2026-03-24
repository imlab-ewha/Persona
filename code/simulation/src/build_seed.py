# -*- coding: utf-8 -*-
"""
fill_party_leaning_ipf.py

목적:
- persona_profile.csv의 party_leaning 컬럼을
  외부 이념성향 분포표(성별/연령별/가구소득별/교육정도별)를 반영하여 채운다.
- 5단계 이념 분포를 3단계(보수/중도/진보)로 합쳐서 사용한다.
- 개인별 확률을 초기화한 뒤, IPF(Iterative Proportional Fitting) 스타일로 반복 보정한다.

출력:
- party_leaning이 채워진 CSV
- 각 persona별 보수/중도/진보 확률 CSV
- 검증용 요약표 CSV
"""

import os
import re
import math
import numpy as np
import pandas as pd


# =========================================================
# 1. 경로 설정
# =========================================================
PERSONA_PATH = "/home/imlab/Persona/data/processed2/persona_profile.csv"
IDEOLOGY_DIST_PATH = "/home/imlab/Persona/data/etc/이념적_성향_20260325023335.csv"

OUTPUT_DIR = "/home/imlab/Persona/data/processed2"
OUTPUT_PERSONA_PATH = os.path.join(OUTPUT_DIR, "persona_profile_filled.csv")
OUTPUT_PROB_PATH = os.path.join(OUTPUT_DIR, "persona_party_leaning_probabilities.csv")
OUTPUT_CHECK_PATH = os.path.join(OUTPUT_DIR, "party_leaning_ipf_validation.xlsx")

REFERENCE_YEAR = 2026
RANDOM_SEED = 42

# 반복 설정
MAX_ITER = 200
TOL = 1e-8
EPS = 1e-12

# 최종 할당 방식: "sample" 또는 "argmax"
ASSIGNMENT_MODE = "sample"


# =========================================================
# 2. 기본 유틸
# =========================================================
def normalize_prob_dict(d):
    total = sum(max(v, 0.0) for v in d.values())
    if total <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: max(v, 0.0) / total for k, v in d.items()}


def row_normalize(arr):
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return arr / row_sum


def safe_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in {"-", "", "nan", "None"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# =========================================================
# 3. 외부 분포표 로딩 및 정리
# =========================================================
def load_ideology_distribution(path):
    """
    원본 CSV 구조 예시:
    1행: "특성별(1)", "특성별(2)", 2024, ...
    2행: "특성별(1)", "특성별(2)", 매우 보수적, 다소 보수적, ...
    이후 데이터행

    반환:
    {
        "전체": {"전체": {"보수": ..., "중도": ..., "진보": ...}},
        "성별": {"남자": {...}, "여자": {...}},
        "연령별": {"19~29세": {...}, ...},
        "가구소득별": {"100만원 미만": {...}, ...},
        "교육정도별": {"고졸": {...}, ...}
    }
    """
    # 인코딩 이슈 대비
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    df = None
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, header=None)
            break
        except Exception as e:
            last_error = e

    if df is None:
        raise RuntimeError(f"이념 분포 CSV를 읽지 못했습니다: {last_error}")

    if df.shape[0] < 3 or df.shape[1] < 7:
        raise ValueError("이념 분포 CSV 형식이 예상과 다릅니다.")

    col_names = [
        "특성별(1)",
        "특성별(2)",
        "매우 보수적",
        "다소 보수적",
        "중도적",
        "다소 진보적",
        "매우 진보적",
    ]

    data = df.iloc[2:, :7].copy()
    data.columns = col_names

    for c in ["매우 보수적", "다소 보수적", "중도적", "다소 진보적", "매우 진보적"]:
        data[c] = data[c].apply(safe_float)

    data["보수"] = data["매우 보수적"].fillna(0) + data["다소 보수적"].fillna(0)
    data["중도"] = data["중도적"].fillna(0)
    data["진보"] = data["다소 진보적"].fillna(0) + data["매우 진보적"].fillna(0)

    dist = {}

    for _, row in data.iterrows():
        dim = str(row["특성별(1)"]).strip()
        grp = str(row["특성별(2)"]).strip()

        if dim not in dist:
            dist[dim] = {}

        p = normalize_prob_dict({
            "보수": float(row["보수"]),
            "중도": float(row["중도"]),
            "진보": float(row["진보"]),
        })
        dist[dim][grp] = p

    return dist


# =========================================================
# 4. persona 변수 매핑 함수
# =========================================================
def map_gender(x):
    s = str(x).strip()
    if s == "남성":
        return "남자"
    if s == "여성":
        return "여자"
    return np.nan


def birth_year_to_age_group(birth_year, reference_year=2024):
    if pd.isna(birth_year):
        return np.nan
    try:
        birth_year = int(float(birth_year))
    except Exception:
        return np.nan

    age = reference_year - birth_year
    if 19 <= age <= 29:
        return "19~29세"
    elif 30 <= age <= 39:
        return "30~39세"
    elif 40 <= age <= 49:
        return "40~49세"
    elif 50 <= age <= 59:
        return "50~59세"
    elif age >= 60:
        return "60세 이상"
    else:
        return np.nan


def parse_income_to_monthly_manwon(income_str):
    """
    persona_profile의 household_income 예:
    - "3천~5천"
    - "5천~7천"
    - "1천 미만" 등일 가능성
    가정:
    - 연 가구소득(만원 단위) 범주라고 보고, 월 소득(만원)으로 환산
    """
    if pd.isna(income_str):
        return np.nan

    s = str(income_str).strip().replace(" ", "")
    if s == "":
        return np.nan

    # 예: "3천~5천"
    m = re.match(r"(\d+)천~(\d+)천", s)
    if m:
        low = int(m.group(1)) * 1000
        high = int(m.group(2)) * 1000
        annual_mid_manwon = (low + high) / 2.0
        monthly_mid_manwon = annual_mid_manwon / 12.0
        return monthly_mid_manwon

    # 예: "7천이상", "7천 이상"
    m = re.match(r"(\d+)천이상", s)
    if m:
        low = int(m.group(1)) * 1000
        annual_mid_manwon = low + 1000  # 상단 열린 구간 임시 midpoint
        monthly_mid_manwon = annual_mid_manwon / 12.0
        return monthly_mid_manwon

    # 예: "1천미만"
    m = re.match(r"(\d+)천미만", s)
    if m:
        high = int(m.group(1)) * 1000
        annual_mid_manwon = high / 2.0
        monthly_mid_manwon = annual_mid_manwon / 12.0
        return monthly_mid_manwon

    # 숫자만 있는 경우
    nums = re.findall(r"\d+", s)
    if nums:
        vals = [int(v) for v in nums]
        annual_mid_manwon = sum(vals) / len(vals) * 1000
        monthly_mid_manwon = annual_mid_manwon / 12.0
        return monthly_mid_manwon

    return np.nan


def monthly_income_to_group(monthly_manwon):
    if pd.isna(monthly_manwon):
        return np.nan

    x = float(monthly_manwon)
    if x < 100:
        return "100만원 미만"
    elif x < 200:
        return "100~200만원 미만"
    elif x < 300:
        return "200~300만원 미만"
    elif x < 400:
        return "300~400만원 미만"
    elif x < 500:
        return "400~500만원 미만"
    elif x < 600:
        return "500~600만원 미만"
    else:
        return "600만원 이상"


def map_household_income_group(x):
    monthly = parse_income_to_monthly_manwon(x)
    return monthly_income_to_group(monthly)


def map_education(x):
    s = str(x).strip()
    if s in {"초졸이하", "초졸 이하"}:
        return "초졸 이하"
    elif s in {"고졸미만", "중졸"}:
        return "중졸"
    elif s in {"고졸", "대학재학"}:
        return "고졸"
    elif s in {"대학졸업", "대학원재학", "대학원졸업"}:
        return "대졸 이상"
    return np.nan


# =========================================================
# 5. 초기 확률 생성
# =========================================================
def get_overall_distribution(dist):
    """
    우선순위:
    - "전체" / "소계"
    - "전체" / "전체"
    """
    if "전체" in dist:
        if "소계" in dist["전체"]:
            return dist["전체"]["소계"]
        if "전체" in dist["전체"]:
            return dist["전체"]["전체"]

    # fallback
    return {"보수": 1/3, "중도": 1/3, "진보": 1/3}


def combine_distributions(prob_list, mode="log_average"):
    """
    prob_list: [{"보수":..., "중도":..., "진보":...}, ...]
    mode:
    - "average": 단순 평균
    - "log_average": 로그평균 후 normalize (권장)
    """
    labels = ["보수", "중도", "진보"]

    if len(prob_list) == 0:
        return {"보수": 1/3, "중도": 1/3, "진보": 1/3}

    if mode == "average":
        out = {lab: 0.0 for lab in labels}
        for p in prob_list:
            for lab in labels:
                out[lab] += p.get(lab, 0.0)
        out = {lab: out[lab] / len(prob_list) for lab in labels}
        return normalize_prob_dict(out)

    elif mode == "log_average":
        scores = {lab: 0.0 for lab in labels}
        for p in prob_list:
            for lab in labels:
                scores[lab] += math.log(max(p.get(lab, 0.0), EPS))
        scores = {lab: scores[lab] / len(prob_list) for lab in labels}

        max_score = max(scores.values())
        exps = {lab: math.exp(scores[lab] - max_score) for lab in labels}
        return normalize_prob_dict(exps)

    else:
        raise ValueError(f"지원하지 않는 결합 모드: {mode}")


def initialize_person_probs(df, dist):
    overall = get_overall_distribution(dist)

    probs = []
    for _, row in df.iterrows():
        plist = []

        g = row["gender_group"]
        a = row["age_group"]
        i = row["income_group"]
        e = row["education_group"]

        if pd.notna(g) and "성별" in dist and g in dist["성별"]:
            plist.append(dist["성별"][g])
        if pd.notna(a) and "연령별" in dist and a in dist["연령별"]:
            plist.append(dist["연령별"][a])
        if pd.notna(i) and "가구소득별" in dist and i in dist["가구소득별"]:
            plist.append(dist["가구소득별"][i])
        if pd.notna(e) and "교육정도별" in dist and e in dist["교육정도별"]:
            plist.append(dist["교육정도별"][e])

        if len(plist) == 0:
            p = overall
        else:
            # overall도 같이 넣어 극단치 완화
            p = combine_distributions([overall] + plist, mode="log_average")

        probs.append([p["보수"], p["중도"], p["진보"]])

    arr = np.array(probs, dtype=float)
    arr = row_normalize(arr)
    return arr


# =========================================================
# 6. IPF 스타일 보정
# =========================================================
def build_constraints(dist):
    """
    사용할 제약조건만 추출
    """
    constraints = []

    mapping = [
        ("gender_group", "성별"),
        ("age_group", "연령별"),
        ("income_group", "가구소득별"),
        ("education_group", "교육정도별"),
    ]

    for col, dim in mapping:
        if dim not in dist:
            continue
        for grp, target_prob in dist[dim].items():
            if grp in {"소계", "전체"}:
                continue
            constraints.append((col, grp, target_prob))

    return constraints


def ipf_adjust(df, init_probs, dist, max_iter=200, tol=1e-8, verbose=True):
    """
    개인별 3-class 확률을 가지고,
    각 subgroup의 class composition이 외부 target distribution과 맞도록 반복 보정.
    """
    probs = init_probs.copy()
    labels = ["보수", "중도", "진보"]
    constraints = build_constraints(dist)

    n = len(df)

    for it in range(max_iter):
        prev = probs.copy()

        for col, grp, target in constraints:
            mask = (df[col] == grp).values
            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            subgroup = probs[idx, :]
            current_sum = subgroup.sum(axis=0)  # 현재 subgroup 내 각 class 예상합
            subgroup_n = len(idx)

            target_sum = np.array([
                target["보수"] * subgroup_n,
                target["중도"] * subgroup_n,
                target["진보"] * subgroup_n,
            ], dtype=float)

            scale = target_sum / np.maximum(current_sum, EPS)

            probs[idx, :] = probs[idx, :] * scale.reshape(1, -1)
            probs[idx, :] = row_normalize(probs[idx, :])

        # 수렴 체크
        delta = np.abs(probs - prev).max()
        if verbose and ((it + 1) % 10 == 0 or it == 0):
            print(f"[IPF] iter={it+1:03d}, max_delta={delta:.12f}")

        if delta < tol:
            if verbose:
                print(f"[IPF] 수렴 완료: iter={it+1}, max_delta={delta:.12f}")
            break

    return probs


# =========================================================
# 7. 최종 라벨 할당
# =========================================================
def assign_labels(prob_array, mode="sample", random_seed=42):
    rng = np.random.default_rng(random_seed)
    labels = np.array(["보수", "중도", "진보"])

    if mode == "argmax":
        idx = np.argmax(prob_array, axis=1)
        return labels[idx]

    elif mode == "sample":
        out = []
        for p in prob_array:
            p = p / p.sum()
            out.append(rng.choice(labels, p=p))
        return np.array(out)

    else:
        raise ValueError(f"지원하지 않는 할당 모드: {mode}")


# =========================================================
# 8. 검증용 테이블 생성
# =========================================================
def summarize_distribution(df, col, label_col="party_leaning"):
    tab = pd.crosstab(df[col], df[label_col], normalize="index") * 100
    for lab in ["보수", "중도", "진보"]:
        if lab not in tab.columns:
            tab[lab] = 0.0
    tab = tab[["보수", "중도", "진보"]]
    return tab


def make_target_df(dist, dim_name):
    rows = []
    if dim_name not in dist:
        return pd.DataFrame(columns=["group", "보수", "중도", "진보"])

    for grp, p in dist[dim_name].items():
        if grp in {"소계", "전체"}:
            continue
        rows.append({
            "group": grp,
            "보수": p["보수"] * 100,
            "중도": p["중도"] * 100,
            "진보": p["진보"] * 100,
        })
    return pd.DataFrame(rows)


# =========================================================
# 9. 메인 실행
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1] 데이터 로딩")
    persona_df = pd.read_csv(PERSONA_PATH)
    dist = load_ideology_distribution(IDEOLOGY_DIST_PATH)

    print("[2] 변수 매핑")
    persona_df["gender_group"] = persona_df["gender"].apply(map_gender)
    persona_df["age_group"] = persona_df["birth_year"].apply(
        lambda x: birth_year_to_age_group(x, REFERENCE_YEAR)
    )
    persona_df["income_group"] = persona_df["household_income"].apply(map_household_income_group)
    persona_df["education_group"] = persona_df["education"].apply(map_education)

    print("[3] 초기 확률 생성")
    init_probs = initialize_person_probs(persona_df, dist)

    print("[4] IPF 보정")
    final_probs = ipf_adjust(
        df=persona_df,
        init_probs=init_probs,
        dist=dist,
        max_iter=MAX_ITER,
        tol=TOL,
        verbose=True,
    )

    print("[5] 최종 party_leaning 할당")
    persona_df["party_leaning"] = assign_labels(
        final_probs, mode=ASSIGNMENT_MODE, random_seed=RANDOM_SEED
    )

    prob_df = pd.DataFrame(final_probs, columns=["party_leaning_prob_conservative",
                                                 "party_leaning_prob_moderate",
                                                 "party_leaning_prob_progressive"])
    prob_df.insert(0, "persona_id", persona_df["persona_id"].values)

    print("[6] 결과 저장")
    # persona_df.to_csv(OUTPUT_PERSONA_PATH, index=False, encoding="utf-8-sig")
    save_df = persona_df.drop(
        columns=["gender_group", "age_group", "income_group", "education_group"],
        errors="ignore"
    )
    save_df.to_csv(OUTPUT_PERSONA_PATH, index=False, encoding="utf-8-sig")
    
    prob_df.to_csv(OUTPUT_PROB_PATH, index=False, encoding="utf-8-sig")

    print("[7] 검증표 생성")
    with pd.ExcelWriter(OUTPUT_CHECK_PATH, engine="openpyxl") as writer:
        overall_assigned = (
            persona_df["party_leaning"]
            .value_counts(normalize=True)
            .reindex(["보수", "중도", "진보"], fill_value=0) * 100
        ).to_frame("assigned_pct")
        overall_assigned.to_excel(writer, sheet_name="overall_assigned")

        target_overall = pd.DataFrame([{
            "보수": get_overall_distribution(dist)["보수"] * 100,
            "중도": get_overall_distribution(dist)["중도"] * 100,
            "진보": get_overall_distribution(dist)["진보"] * 100,
        }])
        target_overall.to_excel(writer, sheet_name="overall_target", index=False)

        checks = [
            ("gender_group", "성별", "check_gender"),
            ("age_group", "연령별", "check_age"),
            ("income_group", "가구소득별", "check_income"),
            ("education_group", "교육정도별", "check_education"),
        ]

        for col, dim_name, sheet_name in checks:
            assigned = summarize_distribution(persona_df, col).reset_index()
            assigned.columns = ["group", "assigned_보수", "assigned_중도", "assigned_진보"]

            target = make_target_df(dist, dim_name)
            target.columns = ["group", "target_보수", "target_중도", "target_진보"]

            merged = pd.merge(target, assigned, on="group", how="outer")
            merged.to_excel(writer, sheet_name=sheet_name, index=False)

    print("완료")
    print(f"- persona 결과: {OUTPUT_PERSONA_PATH}")
    print(f"- 확률 결과: {OUTPUT_PROB_PATH}")
    print(f"- 검증 엑셀: {OUTPUT_CHECK_PATH}")


if __name__ == "__main__":
    main()