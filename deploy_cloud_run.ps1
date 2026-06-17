# Deploy EfficientNet-B3 API to Google Cloud Run
# Project: wezard-similarity-score | Region: us-central1 | Service: vegdetect-api

$ErrorActionPreference = "Stop"

$Project = "wezard-similarity-score"
$Region = "us-central1"
$Bucket = "gs://vegdetect-pos-models/models"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$ModelCandidates = @(
    (Join-Path $RepoRoot "results_new\efficientnet_b3.onnx"),
    (Join-Path $RepoRoot "api_efficientnet_b3\efficientnet_b3.onnx"),
    (Join-Path $RepoRoot "results\efficientnet_b3.onnx")
)

$ModelLocal = $ModelCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$ThresholdLocal = Join-Path $RepoRoot "api_efficientnet_b3\class_thresholds.csv"

if (-not $ModelLocal) {
    Write-Error "No efficientnet_b3.onnx found. Train with train.py first."
}

Write-Host "=== Step 1: gcloud project ===" -ForegroundColor Cyan
gcloud config set project $Project

Write-Host "`n=== Step 2: Upload model + thresholds to GCS ===" -ForegroundColor Cyan
gsutil cp $ModelLocal "$Bucket/efficientnet_b3.onnx"
if (Test-Path $ThresholdLocal) {
    gsutil cp $ThresholdLocal "$Bucket/class_thresholds.csv"
}

Write-Host "`n=== Step 3: Cloud Build + Deploy ===" -ForegroundColor Cyan
Set-Location $RepoRoot
gcloud builds submit --config=api_efficientnet_b3/cloudbuild.yaml --project=$Project .

Write-Host "`n=== Step 4: Service URL ===" -ForegroundColor Cyan
$url = gcloud run services describe vegdetect-api --region=$Region --project=$Project --format="value(status.url)"
Write-Host "API:    $url" -ForegroundColor Green
Write-Host "Swagger: $url/docs" -ForegroundColor Green
Write-Host "UI:      $url/" -ForegroundColor Green
