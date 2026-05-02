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
    동: str
    자치구: str
    월: int
    요일: str
    예측_혼잡도: str
    신뢰도: float = Field(..., description="예측 신뢰도 (%)")
    불확실도: float = Field(..., description="엔트로피 기반 불확실도 (%)")
    확률: dict = Field(..., description="각 혼잡도 클래스별 확률 (%)")
    의료기관수: int
    영업클리닉수: float


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "소아청소년과 혼잡도 예측 API"}


@app.post("/predict/dong", response_model=PredictResponse, tags=["Prediction"])
def predict_congestion(req: PredictRequest):
    result = predict(req.dong, req.month, req.day_of_week)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result)

    return PredictResponse(
        동=result["동"],
        자치구=result["자치구"],
        월=result["월"],
        요일=result["요일"],
        예측_혼잡도=result["예측_혼잡도"],
        신뢰도=result["신뢰도(%)"],
        불확실도=result["불확실도(%)"],
        확률=result["확률(%)"],
        의료기관수=result["의료기관수"],
        영업클리닉수=result["영업클리닉수"],
    )
