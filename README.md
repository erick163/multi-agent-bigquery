# Multi-Agent BigQuery System

A multi-agent system for GCP billing analysis built with Google's Agent Development Kit (ADK).

## Architecture

```
coordinator_agent (SequentialAgent)
├── data_analysis_agent   - Statistical analysis and pattern detection
├── forecasting_agent     - Cost predictions and trend forecasting
└── business_logic_agent  - Business recommendations and regional analysis
```

## Quick Start (Local)

```bash
# Clone and setup
git clone https://github.com/erick163/multi-agent-bigquery.git
cd multi-agent-bigquery
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your GCP project details

# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## GCP Deployment

### 1. Create Service Account

```bash
export PROJECT_ID=your-project-id

# Create service account
gcloud iam service-accounts create bq-agent-sa --display-name="BigQuery Agent"

# Grant permissions
for role in bigquery.dataViewer bigquery.jobUser aiplatform.user; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:bq-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/$role"
done
```

### 2. Create Staging Bucket

```bash
gsutil mb -l us-central1 gs://${PROJECT_ID}-agent-staging
```

### 3. Deploy to Cloud Run

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
EOF

# Build and deploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/bq-agent
gcloud run deploy bq-agent \
  --image=gcr.io/$PROJECT_ID/bq-agent \
  --region=us-central1 \
  --service-account=bq-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_STAGING_BUCKET=${PROJECT_ID}-agent-staging,BQ_BILLING_TABLE=your_billing_table" \
  --no-allow-unauthenticated
```

### 4. Deploy to Compute Engine (Alternative)

```bash
gcloud compute instances create bq-agent-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --service-account=bq-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --scopes=cloud-platform \
  --metadata=startup-script='#!/bin/bash
    apt-get update && apt-get install -y python3-pip git
    git clone https://github.com/erick163/multi-agent-bigquery.git /opt/agent
    cd /opt/agent && pip3 install -r requirements.txt'
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID |
| `GOOGLE_CLOUD_STAGING_BUCKET` | Yes | GCS bucket for Vertex AI |
| `BQ_BILLING_TABLE` | Yes | Billing export table name |
| `GOOGLE_CLOUD_LOCATION` | No | Region (default: `us-central1`) |
| `BQ_DATASET_ID` | No | Dataset (default: `detailed_billing`) |
| `GEMINI_MODEL` | No | Model (default: `gemini-2.5-flash`) |

## Usage

```python
from agents import coordinator_agent

response = coordinator_agent.generate(
    "Analyze billing for customer 'acme' and recommend optimizations"
)
```

## Security

- **Read-only BigQuery**: All writes blocked via `WriteMode.BLOCKED`
- **Minimal IAM**: Service account has only required permissions
- **No secrets in code**: All config via environment variables

## License

MIT
