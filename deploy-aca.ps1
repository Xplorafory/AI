param(
    [string]$RG = "xplora-rg",
    [string]$LOC = "westeurope",
    [string]$ACR = "xploraacr443708",
    [string]$REPO = "xplora-rag-api",
    [string]$TAG = "v1",
    [string]$ENVNAME = "xplora-env",
    [string]$APPNAME = "xplora-rag-app",
    [string]$PGS = "xplorapg-flex",
    [string]$PGDB = "xploraforys",
    [string]$PGADM = "pgadmin"
)
# Generate secure password
$PGPW = [Guid]::NewGuid().ToString()
# Image name
$IMAGE = "{0}.azurecr.io/{1}:{2}" -f $ACR, $REPO, $TAG
$acrTag = ("{0}:{1}" -f $REPO, $TAG)
Write-Host "Building and pushing image to ACR..."
az acr build -r $ACR -t $acrTag -f docker/Dockerfile.api .
if ($LASTEXITCODE -ne 0) {
    throw "ACR build failed."
}
Write-Host "Image built and pushed: $IMAGE"
# Create Postgres flexible server
Write-Host "Creating Postgres Flexible Server..."
az postgres flexible-server create `
    --resource-group $RG `
    --name $PGS `
    --location $LOC `
    --tier Burstable `
    --sku-name B1ms `
    --storage-size 32 `
    --version 16 `
    --admin-user $PGADM `
    --admin-password $PGPW `
    --public-access 0.0.0.0-255.255.255.255
# Get Postgres host
$PGHOST = az postgres flexible-server show -g $RG -n $PGS --query "fullyQualifiedDomainName" -o tsv
$PG_DSN = "postgresql://{0}:{1}@{2}:5432/{3}" -f $PGADM, $PGPW, $PGHOST, $PGDB
Write-Host "Postgres connection string: $PG_DSN"
# Init database extensions
Write-Host "Initializing pgvector extension..."
$createSql = @"
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb_chunks_1536 (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW()
);
"@
try {
    echo $createSql | docker run -i --rm --network host postgres:16 psql $PG_DSN
    Write-Host "pgvector and table ready."
}
catch {
    Write-Warning "Could not run psql in container. Run CREATE EXTENSION manually in Postgres."
}
# Deploy to Azure Container Apps
Write-Host "Deploying Container App..."
az containerapp up `
  --name $APPNAME `
  --resource-group $RG `
  --environment $ENVNAME `
  --image $IMAGE `
  --target-port 8000 `
  --ingress external `
  --env-vars `
    OPENAI_API_KEY=$env:OPENAI_API_KEY `
    PG_DSN=$PG_DSN
Write-Host "Deployment complete!"