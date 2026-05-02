"""
소아청소년과 혼잡도 예측 모델 학습
입력: 동, 월, 요일
출력: 여유 / 보통 / 혼잡 / 매우혼잡
"""
import os, re, json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
df_inst    = pd.read_csv("data/result/서울_소아청소년과_자치구동별_기관수.csv",       encoding="utf-8-sig")
df_visits  = pd.read_csv("data/result/서울_자치구별_소아진료건수_월평균.csv",         encoding="utf-8-sig")
df_hours   = pd.read_csv("data/result/서울_소아청소년과_기관정보_진료시간.csv",       encoding="utf-8-sig")
df_weather = pd.read_csv("data/result/월별_평균기상.csv",                          encoding="utf-8-sig")

DAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

# ── 2. 진료시간 파일에서 자치구·동 추출 ────────────────────────────────────────
def _extract_gu(addr):
    m = re.search(r"서울특별시\s+([가-힣]+구)", str(addr))
    return m.group(1) if m else None

def _extract_dong(addr):
    groups = re.findall(r"\(([^()]+)\)", str(addr))
    if not groups:
        return None
    raw = groups[-1].split(",")[0].strip()
    m = re.match(r"^([가-힣]+(?:동\d*가?|\d*가))", raw)
    return m.group(1) if m else None

def _hhmm_to_h(val):
    """HHMM 정수 → 시간 (float). 예: 1730 → 17.5"""
    if pd.isna(val) or val <= 0:
        return None
    v = int(val)
    return (v // 100) + (v % 100) / 60

df_hours["자치구"] = df_hours["소재지"].apply(_extract_gu)
df_hours["동"]    = df_hours["소재지"].apply(_extract_dong)

# ── 3. (자치구, 동, 요일) 별 영업클리닉수·평균진료시간 집계 ────────────────────
op_records = []
for _, row in df_hours.iterrows():
    gu   = row["자치구"]
    dong = row["동"]
    if not gu or not dong:
        continue
    for day in DAYS:
        c_h = _hhmm_to_h(row.get(f"진료시간({day})C"))
        s_h = _hhmm_to_h(row.get(f"진료시간({day})S"))
        is_open  = bool(c_h and s_h and c_h > s_h)
        op_hours = (c_h - s_h) if is_open else 0.0
        op_records.append({"자치구": gu, "동": dong, "요일": day,
                           "is_open": int(is_open), "op_hours": op_hours})

df_op = pd.DataFrame(op_records)
df_op_agg = (
    df_op.groupby(["자치구", "동", "요일"])
    .agg(
        영업클리닉수=("is_open",   "sum"),
        평균진료시간=("op_hours", lambda x: x[x > 0].mean() if (x > 0).any() else 0.0),
    )
    .reset_index()
)

# ── 4. 동 단위 데이터 없을 때 폴백용 구 평균 영업률 ────────────────────────────
_inst_for_rate = df_inst[(df_inst["동"] != "알수 없음") & (df_inst["자치구"] != "알수 없음")]
df_op_agg_r = df_op_agg.merge(_inst_for_rate[["자치구", "동", "의료기관수"]],
                               on=["자치구", "동"], how="left")
df_op_agg_r["영업률"] = df_op_agg_r["영업클리닉수"] / df_op_agg_r["의료기관수"].clip(lower=1)

gu_day_fallback = (
    df_op_agg_r.groupby(["자치구", "요일"])
    .agg(평균영업률=("영업률", "mean"), 평균진료시간_fb=("평균진료시간", "mean"))
    .reset_index()
)

# ── 5. 피처 행렬 생성 ──────────────────────────────────────────────────────────
df_valid = _inst_for_rate.copy()
gu_total = df_valid.groupby("자치구")["의료기관수"].sum().rename("구_총기관수")

df_visits["합산"] = (
    df_visits["소아청소년과_평균진료건수"].fillna(0)
    + df_visits["소아치과_평균진료건수"].fillna(0)
)

rows = []
for _, ir in df_valid.iterrows():
    dong    = ir["동"]
    gu      = ir["자치구"]
    n_inst  = ir["의료기관수"]
    gu_n    = gu_total.get(gu, n_inst)
    ratio   = n_inst / gu_n  # 이 동의 기관 비중

    for month in range(1, 13):
        vrow = df_visits[(df_visits["진료월"] == month) & (df_visits["시군구"] == gu)]
        if vrow.empty:
            continue
        gu_visits    = vrow.iloc[0]["합산"]
        daily_visits = gu_visits * ratio / 30  # 이 동의 추정 일 평균 진료건수

        wx = df_weather[df_weather["월"] == month].iloc[0]

        for day_idx, day in enumerate(DAYS):
            orow = df_op_agg[(df_op_agg["자치구"] == gu) & (df_op_agg["동"] == dong)
                             & (df_op_agg["요일"] == day)]
            if not orow.empty:
                open_n = orow.iloc[0]["영업클리닉수"]
                avg_h  = orow.iloc[0]["평균진료시간"]
            else:
                fb = gu_day_fallback[(gu_day_fallback["자치구"] == gu)
                                     & (gu_day_fallback["요일"] == day)]
                rate   = fb.iloc[0]["평균영업률"]   if not fb.empty else 0.5
                avg_h  = fb.iloc[0]["평균진료시간_fb"] if not fb.empty else 8.0
                open_n = n_inst * rate

            open_n   = max(open_n, 0.1)
            open_rate = min(open_n / n_inst, 1.0)

            # 클리닉당 일 부하 (기본 타깃 재료)
            load = daily_visits / open_n

            # 기상 가중치: 기온 극단(±15°C 기준) + 강수량
            temp_w = 1 + 0.2 * abs(wx["평균기온"] - 15) / 30
            rain_w = 1 + 0.1 * min(wx["평균강수량"] / 300, 1)
            adj_load = load * temp_w * rain_w

            rows.append({
                "동":           dong,
                "자치구":       gu,
                "월":           month,
                "요일":         day,
                "요일_idx":     day_idx,
                "의료기관수":   n_inst,
                "영업클리닉수": open_n,
                "영업률":       open_rate,
                "평균진료시간": avg_h,
                "구_월평균진료건수":   gu_visits,
                "동_일평균진료건수":   daily_visits,
                "평균기온":     wx["평균기온"],
                "평균일교차":   wx["평균일교차"],
                "평균습도":     wx["평균습도"],
                "평균강수량":   wx["평균강수량"],
                "평균풍속":     wx["평균풍속"],
                "_adj_load":    adj_load,
            })

df_train = pd.DataFrame(rows)
print(f"학습 샘플 수: {len(df_train):,}")

# ── 6. 타깃 변수 생성 (4분위 기반 혼잡도) ─────────────────────────────────────
q25, q50, q75 = df_train["_adj_load"].quantile([0.25, 0.5, 0.75])

def _label(v):
    if v <= q25: return "여유"
    if v <= q50: return "보통"
    if v <= q75: return "혼잡"
    return "매우혼잡"

df_train["혼잡도"] = df_train["_adj_load"].apply(_label)
print("\n혼잡도 분포:")
print(df_train["혼잡도"].value_counts())

# ── 7. 인코딩 ──────────────────────────────────────────────────────────────────
le_dong  = LabelEncoder().fit(df_train["동"])
le_gu    = LabelEncoder().fit(df_train["자치구"])
le_label = LabelEncoder().fit(df_train["혼잡도"])

df_train["동_enc"]     = le_dong.transform(df_train["동"])
df_train["자치구_enc"] = le_gu.transform(df_train["자치구"])
df_train["label"]      = le_label.transform(df_train["혼잡도"])

# 주기 인코딩
df_train["월_sin"] = np.sin(2 * np.pi * df_train["월"]      / 12)
df_train["월_cos"] = np.cos(2 * np.pi * df_train["월"]      / 12)
df_train["요일_sin"] = np.sin(2 * np.pi * df_train["요일_idx"] / 7)
df_train["요일_cos"] = np.cos(2 * np.pi * df_train["요일_idx"] / 7)

FEATURE_COLS = [
    "의료기관수", "영업클리닉수", "영업률", "평균진료시간",
    "구_월평균진료건수", "동_일평균진료건수",
    "평균기온", "평균일교차", "평균습도", "평균강수량", "평균풍속",
    "월_sin", "월_cos", "요일_sin", "요일_cos",
    "자치구_enc", "동_enc",
]

X = df_train[FEATURE_COLS].values
y = df_train["label"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 8. 모델 학습 ───────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)

y_pred = model.predict(X_te)
print(f"\n테스트 정확도: {accuracy_score(y_te, y_pred):.4f}")
print(classification_report(y_te, y_pred, target_names=le_label.classes_))

# 피처 중요도
fi = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
print("피처 중요도 (상위 10):")
for name, imp in fi[:10]:
    print(f"  {name:25s}: {imp:.4f}")

# ── 9. 버전 관리 & 저장 ────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

REGISTRY_PATH = "model/model_registry.json"
if os.path.exists(REGISTRY_PATH):
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    next_num = max(int(v.lstrip("v")) for v in registry["versions"]) + 1
else:
    registry = {"latest": None, "versions": {}}
    next_num = 1

version    = f"v{next_num}"
model_path = f"model/dong_congestion_{version}.pkl"

lookup_cols = ["동", "자치구", "월", "요일", "요일_idx"] + FEATURE_COLS
artifact = {
    "version":      version,
    "model":        model,
    "le_dong":      le_dong,
    "le_gu":        le_gu,
    "le_label":     le_label,
    "feature_cols": FEATURE_COLS,
    "lookup":       df_train[lookup_cols].reset_index(drop=True),
}
joblib.dump(artifact, model_path)

test_acc = float(accuracy_score(y_te, y_pred))
registry["latest"] = version
registry["versions"][version] = {
    "filename":       os.path.basename(model_path),
    "trained_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "sklearn_version": sklearn.__version__,
    "n_samples":      len(df_train),
    "test_accuracy":  round(test_acc, 4),
}
with open(REGISTRY_PATH, "w") as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

print(f"\n모델 저장 완료 → {model_path}  [{version}]")
print(f"레지스트리 업데이트 → {REGISTRY_PATH}")
