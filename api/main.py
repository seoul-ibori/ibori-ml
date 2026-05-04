from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from train.predict import predict

app = FastAPI(
    title="소아청소년과 혼잡도 예측 API",
    description="소아청소년과의 혼잡도를 예측합니다.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    dong: str = Field(..., example="역삼동", description="동 이름")
    month: int = Field(..., ge=1, le=12, example=1, description="월 (1~12)")
    day_of_week: str = Field(..., example="월요일", description="요일 (월요일~일요일)")


class PredictResponse(BaseModel):
    dong: str
    gu: str
    month: int
    day_of_week: str
    congestion_level: str = Field(..., description="여유 / 보통 / 혼잡 / 매우혼잡")
    confidence: float = Field(..., description="예측 신뢰도 (%)")
    uncertainty: float = Field(..., description="엔트로피 기반 불확실도 (%)")
    probability: dict = Field(..., description="각 혼잡도 클래스별 확률 (%)")
    clinic_count: int = Field(..., description="동 내 의료기관 수")
    open_clinic_count: float = Field(..., description="해당 요일 영업 클리닉 수")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "소아청소년과 혼잡도 예측 API"}


@app.post("/predict/dong", response_model=PredictResponse, tags=["Prediction"])
def predict_congestion(req: PredictRequest):
    result = predict(req.dong, req.month, req.day_of_week)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result)

    return PredictResponse(
        dong=result["동"],
        gu=result["자치구"],
        month=result["월"],
        day_of_week=result["요일"],
        congestion_level=result["예측_혼잡도"],
        confidence=result["신뢰도(%)"],
        uncertainty=result["불확실도(%)"],
        probability=result["확률(%)"],
        clinic_count=result["의료기관수"],
        open_clinic_count=result["영업클리닉수"],
    )
