"""
소아청소년과 혼잡도 예측
사용법:
    from train.predict import predict
    result = predict("역삼동", 1, "월요일")
"""
import json
import numpy as np
import joblib

_artifact = None

DAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

def _load():
    global _artifact
    if _artifact is not None:
        return
    with open("model/model_registry.json") as f:
        registry = json.load(f)
    latest_file = registry["versions"][registry["latest"]]["filename"]
    _artifact = joblib.load(f"model/{latest_file}")


def predict(dong: str, month: int, day_of_week: str) -> dict:
    """
    Args:
        dong        : 동 이름  (예: "역삼동")
        month       : 월       (1 ~ 12)
        day_of_week : 요일     (예: "월요일")
    Returns:
        {
          "동", "자치구", "월", "요일",
          "예측_혼잡도",          # 여유 / 보통 / 혼잡 / 매우혼잡
          "확률",                 # 각 클래스 확률 (%)
          "의료기관수",
          "영업클리닉수",
        }
    """
    _load()

    model      = _artifact["model"]
    le_dong    = _artifact["le_dong"]
    le_label   = _artifact["le_label"]
    lookup     = _artifact["lookup"]
    feat_cols  = _artifact["feature_cols"]

    # ── 입력 검증 ───────────────────────────────────────────────────────────────
    if day_of_week not in DAYS:
        return {"error": f"요일은 {DAYS} 중 하나여야 합니다."}
    if not (1 <= month <= 12):
        return {"error": "월은 1~12 사이여야 합니다."}
    if dong not in le_dong.classes_:
        similar = [d for d in le_dong.classes_ if dong.replace("동", "") in d]
        return {"error": f"'{dong}'을(를) 찾을 수 없습니다.", "유사동": similar[:5]}

    # ── 룩업 ────────────────────────────────────────────────────────────────────
    row = lookup[
        (lookup["동"] == dong) &
        (lookup["월"] == month) &
        (lookup["요일"] == day_of_week)
    ]
    if row.empty:
        return {"error": "해당 조건의 데이터를 찾을 수 없습니다."}

    row = row.iloc[0]
    X = row[feat_cols].values.reshape(1, -1)

    # ── 예측 ────────────────────────────────────────────────────────────────────
    pred_idx  = model.predict(X)[0]
    pred_prob = model.predict_proba(X)[0]
    label     = le_label.inverse_transform([pred_idx])[0]

    order = ["여유", "보통", "혼잡", "매우혼잡"]
    proba = {cls: round(float(p) * 100, 1) for cls, p in zip(le_label.classes_, pred_prob)}
    proba = {k: proba[k] for k in order if k in proba}

    confidence = round(float(pred_prob.max()) * 100, 1)

    # 엔트로피 기반 불확실도 (0=완전 확신, 100=완전 불확실)
    p = pred_prob + 1e-10
    max_entropy = np.log(len(p))
    entropy = round(float(-np.sum(p * np.log(p)) / max_entropy) * 100, 1)

    # 1·2위 확률 차이 (클수록 명확한 예측)
    sorted_p = np.sort(pred_prob)[::-1]
    margin = round(float(sorted_p[0] - sorted_p[1]) * 100, 1)

    return {
        "동":           dong,
        "자치구":       row["자치구"],
        "월":           month,
        "요일":         day_of_week,
        "예측_혼잡도":  label,
        "신뢰도(%)":    confidence,
        "불확실도(%)":  entropy,
        "1·2위차(%)":   margin,
        "확률(%)":      proba,
        "의료기관수":   int(row["의료기관수"]),
        "영업클리닉수": round(float(row["영업클리닉수"]), 1),
    }


if __name__ == "__main__":
    import json
    tests = [
        ("역삼동", 1, "월요일"),
        ("목동",   7, "토요일"),
        ("상계동", 12, "일요일"),
        ("여의도동", 3, "수요일"),
    ]
    for dong, month, day in tests:
        result = predict(dong, month, day)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()
