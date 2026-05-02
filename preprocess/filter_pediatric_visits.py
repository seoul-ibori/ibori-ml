import pandas as pd

df = pd.read_csv(
    "data/origin_data/국민건강보험공단_시군구별 진료과목별 진료 정보_20231231.csv",
    encoding="cp949"
)

# 서울특별시, 2022~2023년, 소아청소년과·소아치과 필터링
filtered = df[
    (df["시도"] == "서울특별시") &
    (df["진료년도"].isin([2022, 2023])) &
    (df["진료과목명"].isin(["소아청소년과", "소아치과"]))
].copy()

# 지정 칼럼 제거
filtered = filtered.drop(columns=["진료년도", "시도", "진료과목코드", "진료인원(명)"])

# 월·자치구·진료과목별 연도 평균 집계 (2022, 2023 평균)
grouped = (
    filtered
    .groupby(["진료월", "시군구", "진료과목명"], as_index=False)["진료건수(건)"]
    .mean()
    .round(1)
)

# 진료과목명을 칼럼으로 피벗
result = grouped.pivot_table(
    index=["진료월", "시군구"],
    columns="진료과목명",
    values="진료건수(건)"
).reset_index()

result.columns.name = None
result = result.rename(columns={
    "소아청소년과": "소아청소년과_평균진료건수",
    "소아치과": "소아치과_평균진료건수"
})

result = result.sort_values(["진료월", "시군구"]).reset_index(drop=True)

output_path = "data/result/서울_자치구별_소아진료건수_월평균.csv"
result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"저장 완료: {output_path}")
print(f"행 수: {len(result)}")
print(result.head(10).to_string(index=False))
