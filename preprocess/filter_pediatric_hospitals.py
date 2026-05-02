import pandas as pd

df = pd.read_csv(
    "data/origin_data/기관정보.csv",
    encoding="utf-8-sig"
)

filtered = df[
    (df["소아청소년과 개설"] == "Y") &
    (df["시도명"] == "서울특별시")
].copy()

drop_cols = [
    "기준연도", "시도코드", "시도명", "요양종별코드",
    "내과 개설", "외과 개설", "산부인과 개설", "소아청소년과 개설",
    "정형외과 개설", "신생아중환자실 병상 보유", "중환자실 병상 보유",
    "분만실 병상 보유", "중앙응급의료센터 지정", "권역응급의료센터 지정",
    "지역응급의료센터 지정", "지역응급의료기관 지정", "권역외상센터 지정",
    "권역암센터 지정", "지역암센터 지정", "권역심뇌혈관질환센터 지정",
    "지역심뇌혈관질환센터 지정", "공공의료기관", "지역보건의료기관",
    "권역모자의료센터 지정", "지역모자의료센터 지정",
    "분만취약지지원사업 의료기관 지정", "의료취약지지원사업 의료기관 지정",
    "중증모자의료센터 지정",
]

filtered = filtered.drop(columns=[c for c in drop_cols if c in filtered.columns])

output_path = "data/result/서울_소아청소년과_기관정보.csv"
filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"저장 완료: {output_path}")
print(f"행 수: {len(filtered)}")
print(f"칼럼: {list(filtered.columns)}")
