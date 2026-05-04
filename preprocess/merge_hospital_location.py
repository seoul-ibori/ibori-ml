import pandas as pd
import re

df1 = pd.read_csv("data/result/서울_소아청소년과_기관정보.csv", encoding="utf-8-sig")
df2 = pd.read_csv("data/hospital_location_2.csv", encoding="utf-8-sig")


def normalize(addr):
    if pd.isna(addr):
        return ""
    return re.sub(r"\s+", " ", addr).strip()


df1["_addr"] = df1["소재지"].apply(normalize)
df2["_addr"] = df2["주소"].apply(normalize)

# 같은 주소에 여러 행이 있을 경우 첫 번째만 사용
df2_dedup = df2.drop_duplicates(subset="_addr", keep="first")

join_cols = [
    "진료시간(월요일)C", "진료시간(화요일)C", "진료시간(수요일)C", "진료시간(목요일)C",
    "진료시간(금요일)C", "진료시간(토요일)C", "진료시간(일요일)C", "진료시간(공휴일)C",
    "진료시간(월요일)S", "진료시간(화요일)S", "진료시간(수요일)S", "진료시간(목요일)S",
    "진료시간(금요일)S", "진료시간(토요일)S", "진료시간(일요일)S", "진료시간(공휴일)S",
    "병원경도", "병원위도",
]

merged = df1.merge(
    df2_dedup[["_addr"] + join_cols],
    on="_addr",
    how="left"
)

result = merged[["요양기관명", "소재지", "요양종별"] + join_cols].copy()

output_path = "data/result/서울_소아청소년과_기관정보_진료시간.csv"
result.to_csv(output_path, index=False, encoding="utf-8-sig")

matched = result["병원경도"].notna().sum()
unmatched = result["병원경도"].isna().sum()
print(f"저장 완료: {output_path}")
print(f"전체 행 수: {len(result)}")
print(f"진료시간 매칭 성공: {matched}개")
print(f"진료시간 미매칭 (NaN): {unmatched}개")
print()
print(result.head(5).to_string(index=False))
