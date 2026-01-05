@echo off
set PROJECT_ID=propane-net-247501
set POOL_NAME=github-pool
set PROVIDER_NAME=github-provider

echo ---------------------------------------------------
echo 1. Creating OIDC Provider (Windows Escaped)...

:: The caret (^) allows line continuation.
:: We use double quotes around the mapping argument to protect the commas.
call gcloud iam workload-identity-pools providers create-oidc %PROVIDER_NAME% ^
    --location="global" ^
    --workload-identity-pool="%POOL_NAME%" ^
    --display-name="GitHub OIDC Provider" ^
    --issuer-uri="https://token.actions.githubusercontent.com" ^
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"

echo.
echo ---------------------------------------------------
echo 2. Getting Provider Name (COPY THIS)...
call gcloud iam workload-identity-pools providers describe %PROVIDER_NAME% ^
    --location="global" ^
    --workload-identity-pool="%POOL_NAME%" ^
    --format="value(name)"
echo.
echo ---------------------------------------------------
