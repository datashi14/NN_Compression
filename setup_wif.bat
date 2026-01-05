@echo off
set PROJECT_ID=propane-net-247501
set POOL_NAME=github-pool
set PROVIDER_NAME=github-provider

echo Creating OIDC Provider...
call gcloud iam workload-identity-pools providers create-oidc %PROVIDER_NAME% ^
    --location="global" ^
    --workload-identity-pool="%POOL_NAME%" ^
    --display-name="GitHub OIDC Provider" ^
    --issuer-uri="https://token.actions.githubusercontent.com" ^
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"

echo Creating Service Account...
call gcloud iam service-accounts create ticketsmith-ci --display-name="TicketSmith CI Account"

echo Getting Provider Name...
call gcloud iam workload-identity-pools providers describe %PROVIDER_NAME% ^
    --location="global" ^
    --workload-identity-pool="%POOL_NAME%" ^
    --format="value(name)"
