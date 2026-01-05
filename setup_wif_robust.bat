@echo off
set PROJECT_ID=propane-net-247501
set POOL_NAME=github-pool
set PROVIDER_NAME=github-provider

echo ---------------------------------------------------
echo 1. Creating/Updating OIDC Provider using JSON file...
call gcloud iam workload-identity-pools providers create-oidc %PROVIDER_NAME% ^
    --location="global" ^
    --workload-identity-pool="%POOL_NAME%" ^
    --display-name="GitHub OIDC Provider" ^
    --issuer-uri="https://token.actions.githubusercontent.com" ^
    --attribute-mapping-file="oidc_mapping.json"

if %ERRORLEVEL% NEQ 0 (
    echo Provider might already exist. Attempting update...
    call gcloud iam workload-identity-pools providers update-oidc %PROVIDER_NAME% ^
        --location="global" ^
        --workload-identity-pool="%POOL_NAME%" ^
        --display-name="GitHub OIDC Provider" ^
        --issuer-uri="https://token.actions.githubusercontent.com" ^
        --attribute-mapping-file="oidc_mapping.json"
)

echo.
echo ---------------------------------------------------
echo 2. Ensuring Service Account exists...
call gcloud iam service-accounts create ticketsmith-ci --display-name="TicketSmith CI Account" 2>nul
echo (It is okay if it already existed)

echo.
echo ---------------------------------------------------
echo 3. Getting Provider Name (COPY THIS)...
echo.
call gcloud iam workload-identity-pools providers describe %PROVIDER_NAME% ^
    --location="global" ^
    --workload-identity-pool="%POOL_NAME%" ^
    --format="value(name)"
echo.
echo ---------------------------------------------------
