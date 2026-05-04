import pandas as pd

raw = pd.read_csv("data/monthly_weather.csv", encoding="utf-8-sig", header=None)

# 0행: 컬럼 대분류, 1행: 소분류 → 실제 데이터는 2행부터
df = raw.iloc[2:].copy()
df.columns = [
    "시점", "평균기온", "평균최고기온", "극점최고기온",
    "평균최저기온", "극점최저기온", "강수량",
    "평균습도", "최소습도", "평균운량", "일조시간",
    "평균풍속", "최대풍속", "최대순간풍속"
]

df = df.reset_index(drop=True)

# 연도·월 분리 및 숫자 변환
df["연도"] = df["시점"].str.extract(r"(\d{4})").astype(int)
df["월"] = pd.to_numeric(df["시점"].str.extract(r"\.\s*(\d{1,2})")[0], errors="coerce")
df = df.dropna(subset=["월"])  # 연간 합계 행 등 월 정보 없는 행 제거
df["월"] = df["월"].astype(int)

# 2000년 이후 필터링
df = df[df["연도"] >= 2000].copy()

numeric_cols = ["평균기온", "평균최고기온", "평균최저기온", "강수량", "평균습도", "평균풍속"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# 일교차 = 평균최고기온 - 평균최저기온
df["평균일교차"] = df["평균최고기온"] - df["평균최저기온"]

# 월별 평균 집계
result = (
    df.groupby("월", as_index=False)
    .agg(
        평균기온=("평균기온", "mean"),
        평균일교차=("평균일교차", "mean"),
        평균습도=("평균습도", "mean"),
        평균강수량=("강수량", "mean"),
        평균풍속=("평균풍속", "mean"),
    )
    .round(2)
)

output_path = "data/result/월별_평균기상.csv"
result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"저장 완료: {output_path}")
print(f"사용 연도 범위: {df['연도'].min()} ~ {df['연도'].max()} ({df['연도'].nunique()}개 연도)")
print()
print(result.to_string(index=False))
