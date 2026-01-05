$ProjectID = "propane-net-247501"
$PoolID = "github-pool"
$ProviderID = "github-provider"
$Location = "global"

# Get Access Token
$Token = gcloud auth print-access-token
$Headers = @{
    "Authorization" = "Bearer $Token"
    "Content-Type"  = "application/json"
}

# Construct API URL
$Url = "https://iam.googleapis.com/v1/projects/$ProjectID/locations/$Location/workloadIdentityPools/$PoolID/providers?workloadIdentityProviderId=$ProviderID"

# Construct Body
$Body = @{
    name             = "projects/$ProjectID/locations/$Location/workloadIdentityPools/$PoolID/providers/$ProviderID"
    displayName      = "GitHub OIDC Provider"
    description      = "Created via Direct API"
    state            = "ACTIVE"
    oidc             = @{
        issuerUri = "https://token.actions.githubusercontent.com"
    }
    attributeMapping = @{
        "google.subject"       = "assertion.sub"
        "attribute.repository" = "assertion.repository"
        "attribute.ref"        = "assertion.ref"
    }
} | ConvertTo-Json -Depth 5

Write-Host "Creating Provider via API..."
try {
    $Response = Invoke-RestMethod -Uri $Url -Method Post -Headers $Headers -Body $Body
    Write-Host "Success! Provider Created:"
    Write-Host $Response.name
}
catch {
    Write-Error "API Call Failed:"
    Write-Host $_.Exception.Response.GetResponseStream()
    $Reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
    $Reader.ReadToEnd()
}
