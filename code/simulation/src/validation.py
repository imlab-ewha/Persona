import numpy as np
import pandas as pd
from scipy.stats import entropy, chi2_contingency

def calculate_validation_metrics(actual_dict, predicted_dict):
    """
    Actual(실제)과 Predicted(예측) 분포를 비교하여 6가지 주요 지표를 산출합니다.
    """
    # 데이터가 없거나 모든 값이 0인 경우 처리
    if not actual_dict or sum(actual_dict.values()) == 0 or sum(predicted_dict.values()) == 0:
        return None

    # 1. 카테고리 동기화 (1.0과 7.0 고정)
    all_categories = sorted(list(set(actual_dict.keys()) | set(predicted_dict.keys())))
    
    epsilon = 1e-10
    act_vec = np.array([actual_dict.get(cat, 0) for cat in all_categories])
    pre_vec = np.array([predicted_dict.get(cat, 0) for cat in all_categories])
    
    # 확률 밀도로 변환 (Sum to 1)
    p = (act_vec + epsilon) / (np.sum(act_vec) + epsilon * len(all_categories))
    q = (pre_vec + epsilon) / (np.sum(pre_vec) + epsilon * len(all_categories))

    # [Metric 1] KL Divergence
    kl_div = entropy(p, q)

    # [Metric 2] JS Distance (대칭적 거리)
    m = 0.5 * (p + q)
    js_dist = np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))

    # [Metric 3] Entropy (불확실성)
    entropy_act = entropy(p)
    entropy_pre = entropy(q)

    # [Metric 4] Cramer's V (연관도)
    try:
        contingency_table = np.array([act_vec, pre_vec])
        r, k = contingency_table.shape
        
        # 유효한 보기가 2개 이상 남아있을 때만 연관성 계산
        if k > 1:
            chi2 = chi2_contingency(contingency_table)[0]
            n = np.sum(contingency_table)
            if n > 0:
                phi2 = chi2 / n
                # 분모가 0이 되는 것을 방지
                denom = min(k - 1, r - 1)
                if denom > 0:
                    cramer_v = np.sqrt(phi2 / denom)
                    cramer_v = round(cramer_v, 4)
                else:
                    cramer_v = "-"
            else:
                cramer_v = "-"
        else:
            cramer_v = "-"
            
    except Exception:
        cramer_v = "-"

    # [Metric 5] Cronbach's Alpha (일관성)
    df_alpha = pd.DataFrame({'actual': act_vec, 'pred': pre_vec})
    def cronbach_alpha(df):
        try:
            item_vars = df.var(axis=0, ddof=1).sum()
            total_var = df.sum(axis=1).var(ddof=1)
            k = df.shape[1]
            if total_var == 0: return 0.0
            return (k / (k - 1)) * (1 - (item_vars / total_var))
        except: return 0.0
    
    c_alpha = cronbach_alpha(df_alpha)

    return {
        "KL_Divergence": round(kl_div, 4),
        "JS_Distance": round(js_dist, 4),
        "Cramer_V": cramer_v,
        "Cronbach_Alpha": round(c_alpha, 4),
        "Entropy_Actual": round(entropy_act, 4),
        "Entropy_Predicted": round(entropy_pre, 4)
    }