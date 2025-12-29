# Multi-Agent BigQuery System

A multi-agent system for GCP billing analysis built with Google's Agent Development Kit (ADK). Features specialized agents for data analysis, forecasting, and business recommendations with read-only BigQuery access.

## Architecture

```
coordinator_agent (SequentialAgent)
├── data_analysis_agent   - Statistical analysis and pattern detection
├── forecasting_agent     - Cost predictions and trend forecasting
└── business_logic_agent  - Business recommendations and regional analysis
```

## Quick Start

### 1. Prerequisites

- Python 3.9+
- GCP project with BigQuery billing export enabled
- Service account with required permissions

### 2. Installation

```bash
pip install -r requirements.txt
```

### 3. Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required variables:
- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `GOOGLE_CLOUD_STAGING_BUCKET` - GCS bucket for Vertex AI
- `BQ_BILLING_TABLE` - Your billing export table name

### 4. Authentication

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 5. Usage

```python
from multi_tool_agent_github.agents import coordinator_agent

# Run a complete analysis workflow
response = coordinator_agent.generate(
    "Analyze billing data for customer 'acme' and provide cost optimization recommendations"
)
```

## GCP VM Deployment (Private, Restricted Access)

### Service Account Permissions

Create a service account with minimal permissions:

```bash
# Create service account
gcloud iam service-accounts create bq-agent-sa \
    --display-name="BigQuery Agent Service Account"

# Grant read-only BigQuery access
gcloud projects add-iam-policy-binding YOUR_PROJECT \
    --member="serviceAccount:bq-agent-sa@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding YOUR_PROJECT \
    --member="serviceAccount:bq-agent-sa@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"

# Grant Vertex AI access
gcloud projects add-iam-policy-binding YOUR_PROJECT \
    --member="serviceAccount:bq-agent-sa@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Grant GCS read access for staging bucket
gsutil iam ch serviceAccount:bq-agent-sa@YOUR_PROJECT.iam.gserviceaccount.com:objectViewer \
    gs://YOUR_STAGING_BUCKET
```

### VPC Service Controls

Create a perimeter to protect BigQuery and Vertex AI:

```bash
# Create access policy (if not exists)
gcloud access-context-manager policies create \
    --organization=YOUR_ORG_ID \
    --title="BQ Agent Policy"

# Create service perimeter
gcloud access-context-manager perimeters create bq-agent-perimeter \
    --title="BigQuery Agent Perimeter" \
    --resources="projects/YOUR_PROJECT_NUMBER" \
    --restricted-services="bigquery.googleapis.com,aiplatform.googleapis.com,storage.googleapis.com" \
    --policy=YOUR_POLICY_ID
```

### Private VM Deployment (No SSH)

```bash
# Create VPC with Private Google Access
gcloud compute networks create bq-agent-vpc --subnet-mode=custom

gcloud compute networks subnets create bq-agent-subnet \
    --network=bq-agent-vpc \
    --region=us-central1 \
    --range=10.0.0.0/24 \
    --enable-private-ip-google-access

# Create VM with no external IP
gcloud compute instances create bq-agent-vm \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --network-interface=network=bq-agent-vpc,subnet=bq-agent-subnet,no-address \
    --service-account=bq-agent-sa@YOUR_PROJECT.iam.gserviceaccount.com \
    --scopes=cloud-platform \
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --metadata-from-file=startup-script=startup.sh
```

### Cloud Run (Private) Deployment

For HTTP API access within VPC only:

```bash
# Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT/bq-agent

# Deploy to Cloud Run (internal only)
gcloud run deploy bq-agent \
    --image=gcr.io/YOUR_PROJECT/bq-agent \
    --platform=managed \
    --region=us-central1 \
    --service-account=bq-agent-sa@YOUR_PROJECT.iam.gserviceaccount.com \
    --ingress=internal \
    --no-allow-unauthenticated \
    --vpc-connector=YOUR_VPC_CONNECTOR
```

## Security Features

- **Read-Only Access**: All BigQuery operations use `WriteMode.BLOCKED`
- **No External IP**: VM has no public internet access
- **VPC Service Controls**: Perimeter protection for data access
- **Minimal IAM**: Service account has only required permissions
- **Shielded VM**: Secure boot and integrity monitoring enabled

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | - | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | No | `us-central1` | GCP region |
| `GOOGLE_CLOUD_STAGING_BUCKET` | Yes | - | GCS staging bucket |
| `BQ_DATASET_ID` | No | `detailed_billing` | BigQuery dataset |
| `BQ_BILLING_TABLE` | Yes | - | Billing export table |
| `BQ_PRICING_TABLE` | No | `cloud_pricing_export` | Pricing table |
| `BQ_CUSTOMERS_TABLE` | No | `gcp_customers` | Customers table |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model |

## License

Private repository - internal use only.
