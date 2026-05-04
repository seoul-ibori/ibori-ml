#!/bin/sh
# OCI Object Storage에서 최신 모델 파일 다운로드

REGISTRY="model/model_registry.json"
FILENAME=$(python3 -c "import json; d=json.load(open('$REGISTRY')); print(d['versions'][d['latest']]['filename'])")

ENDPOINT="https://${OCI_NAMESPACE}.compat.objectstorage.${OCI_REGION}.oraclecloud.com"
BUCKET="ibori-ml-models"

echo "모델 다운로드 중: $FILENAME"

AWS_ACCESS_KEY_ID=$OCI_ACCESS_KEY \
AWS_SECRET_ACCESS_KEY=$OCI_SECRET_KEY \
aws s3 cp \
  "s3://${BUCKET}/model/${FILENAME}" \
  "model/${FILENAME}" \
  --endpoint-url "$ENDPOINT" \
  --region "$OCI_REGION"

echo "다운로드 완료: model/$FILENAME"
