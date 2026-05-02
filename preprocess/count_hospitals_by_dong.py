import pandas as pd
import re

df = pd.read_csv("data/result/서울_소아청소년과_기관정보.csv", encoding="utf-8-sig")


def extract_gu(addr):
    m = re.search(r'서울특별시\s+([가-힣]+구)', addr)
    return m.group(1) if m else "알수 없음"


def extract_dong(addr):
    # 주소에서 모든 괄호 그룹 찾기 → 마지막 괄호가 동 정보
    groups = re.findall(r'\(([^()]+)\)', addr)
    if not groups:
        return "알수 없음"

    raw = groups[-1].split(',')[0].strip()

    # 동/가로 끝나는 앞부분만 추출 (예: 역삼동650-9 → 역삼동, 을지로6가 → 을지로6가)
    m = re.match(r'^([가-힣]+(?:동\d*가?|\d*가))', raw)
    return m.group(1) if m else "알수 없음"


df["자치구"] = df["소재지"].apply(extract_gu)
df["동"] = df["소재지"].apply(extract_dong)

result = (
    df.groupby(["자치구", "동"], as_index=False)
    .size()
    .rename(columns={"size": "의료기관수"})
    .sort_values(["자치구", "동"])
    .reset_index(drop=True)
)

output_path = "data/result/서울_소아청소년과_자치구동별_기관수.csv"
result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"저장 완료: {output_path}")
print(f"행 수: {len(result)}")
print()
print("알수 없음 현황:")
unknown = result[result["자치구"].eq("알수 없음") | result["동"].eq("알수 없음")]
print(unknown.to_string(index=False))
print()
print("샘플 (상위 20행):")
print(result.head(20).to_string(index=False))
