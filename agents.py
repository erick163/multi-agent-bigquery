"""
Multi-agent BigQuery system for GCP billing analysis.

Configuration is loaded from environment variables for deployment flexibility.
See .env.example for required configuration.
"""

import os
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
import google.auth
import vertexai

# Import shared services for modern architecture
from .shared_services import (
    CustomerLookupService,
    QueryRouter,
    ResultSynthesizer,
    CrossRegionAnalyzer,
    get_current_date_sync,
    test_crypto_mining_hypothesis
)
from .custom_tools import crypto_mining_analysis_tool

# =============================================================================
# Configuration from Environment Variables
# =============================================================================

def get_required_env(name: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value

def get_optional_env(name: str, default: str) -> str:
    """Get an optional environment variable with a default value."""
    return os.environ.get(name, default)

# Core configuration
GEMINI_MODEL = get_optional_env("GEMINI_MODEL", "gemini-2.5-flash")
PROJECT_ID = get_required_env("GOOGLE_CLOUD_PROJECT")
LOCATION = get_optional_env("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = get_required_env("GOOGLE_CLOUD_STAGING_BUCKET")

# BigQuery table configuration
DATASET_ID = get_optional_env("BQ_DATASET_ID", "detailed_billing")
BILLING_TABLE = get_required_env("BQ_BILLING_TABLE")
PRICING_TABLE = get_optional_env("BQ_PRICING_TABLE", "cloud_pricing_export")
CUSTOMERS_TABLE = get_optional_env("BQ_CUSTOMERS_TABLE", "gcp_customers")

# Get current date dynamically from billing data, with fallback to actual system date
try:
    CURRENT_DATE = get_current_date_sync()
except Exception as e:
    print(f"Warning: Could not retrieve current date from billing data: {e}")
    from datetime import datetime
    CURRENT_DATE = datetime.now().strftime('%B %-d, %Y')  # System date fallback

# =============================================================================
# Vertex AI Initialization
# =============================================================================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)

# =============================================================================
# BigQuery Configuration (Read-Only Access)
# =============================================================================

tool_config = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)
application_default_credentials, _ = google.auth.default()
credentials_config = BigQueryCredentialsConfig(
    credentials=application_default_credentials
)

# Data Analysis Agent - Full exploration toolset
data_analysis_toolset = BigQueryToolset(
    credentials_config=credentials_config,
    bigquery_tool_config=tool_config,
    tool_filter=[
        'list_dataset_ids',    # Dataset discovery
        'get_dataset_info',    # Dataset metadata
        'list_table_ids',      # Table discovery
        'get_table_info',      # Schema analysis
        'execute_sql',         # Statistical queries
    ]
)

# Forecasting Agent - Time series focused toolset with customer access
forecasting_toolset = BigQueryToolset(
    credentials_config=credentials_config,
    bigquery_tool_config=tool_config,
    tool_filter=[
        'list_dataset_ids',    # Dataset discovery for customer lookup
        'list_table_ids',      # Table discovery for customer lookup
        'get_table_info',      # Schema validation
        'execute_sql',         # Historical data queries and customer lookup
    ]
)

# Business Logic Agent - Enhanced business context and regional analysis toolset
business_logic_toolset = BigQueryToolset(
    credentials_config=credentials_config,
    bigquery_tool_config=tool_config,
    tool_filter=[
        'list_dataset_ids',    # Dataset discovery for customer lookup and regional analysis
        'get_dataset_info',    # Dataset context for business and regional analysis
        'list_table_ids',      # Table discovery for customer lookup and pricing data
        'get_table_info',      # Business data structure and pricing schema analysis
        'execute_sql',         # Business rule queries, customer lookup, and regional comparisons
    ]
)

# =============================================================================
# Agent Definitions
# =============================================================================

# 1. Data Analysis Agent - Handles customer-specific data exploration and statistical analysis
data_analysis_agent = Agent(
    model=GEMINI_MODEL,
    name="data_analyst",
    description="Performs customer-specific statistical analysis, data exploration, and derives insights from BigQuery data.",
    output_key="data_analysis_results",
    instruction=f"""
    You are a BigQuery data analyst specializing in GCP billing cost analysis.

    PROJECT CONTEXT:
    - Project: {PROJECT_ID}
    - Dataset: {DATASET_ID} (billing, pricing, customers)
    - Current Date: {CURRENT_DATE}
    - Tables: {BILLING_TABLE}, {PRICING_TABLE}, {CUSTOMERS_TABLE} (all in {DATASET_ID})

    ROLE: Perform statistical analysis and data exploration on GCP billing data. Generate insights about usage patterns, cost trends, and data quality.

    MULTI-CLOUD ANALYSIS:
    - **GCP Data**: Use `{BILLING_TABLE}` table for GCP billing analysis
    - **GCP Billing Data**: Use `{BILLING_TABLE}` table for comprehensive billing analysis
    - **Service Analysis**: Query billing table for service-specific insights
    - **Service Comparison**: Compare costs, usage patterns, and optimization opportunities across services

    WORKFLOW:
    1. **Customer Context**: ALWAYS use CustomerLookupService for customer identification before ANY queries
       - For customer names like 'honeycomb', 'acme', etc. NEVER use them as billing_account_id directly
       - FIRST query gcp_customers table to find the actual billing_account_id
       - Use: SELECT billing_account_id, billing_account_name FROM gcp_customers WHERE LOWER(billing_account_name) LIKE '%[customer_name]%'
    2. **Data Analysis**: Focus on statistical analysis, pattern detection, and trend identification
    3. **Quality Assessment**: Validate data quality and identify anomalies
    4. **Results**: Store findings in 'data_analysis_results' with clear statistical insights

    BIGQUERY BEST PRACTICES - GCP BILLING EXPORT:
    - **CRITICAL**: NEVER use customer names directly as billing_account_id (e.g., 'honeycomb' is WRONG)
    - **REQUIRED**: Always resolve customer names to billing_account_id using gcp_customers table FIRST
    - Use relative date calculations with GCP billing temporal fields: usage_start_time, usage_end_time
    - Filter by customer: WHERE billing_account_id = 'ACTUAL_BILLING_ACCOUNT_ID'
    - **GCP Cost Metrics**: Use cost for actual spend analysis, credits for discount analysis
    - **GCP Dimensions**: service.description, sku.description, location.location, project.id for analysis
    - **GCP Commitments**: Use LOWER(sku.description) LIKE '%commitment%' for commitment identification
    - GCP billing analysis using standard billing export schema
    - Commitment coverage: (commitment_cost / total_cost) * 100 using cost fields

    KEY FOCUS AREAS:
    - Usage pattern analysis by service, region, project
    - Cost trend identification and variance analysis
    - Data quality assessment and completeness metrics for billing data
    - Statistical summaries: percentiles, distributions, outliers
    - Growth rates and seasonal patterns
    - Commitment discount coverage analysis by service
    - Cost optimization opportunities and regional analysis

    GCP BILLING EXPORT QUERY EXAMPLES:

    **GCP Cost Analysis:**
    ```sql
    SELECT
        service.description as service_name,
        sku.description as sku_description,
        location.location,
        SUM(cost) as total_cost,
        SUM(ARRAY_LENGTH(credits)) as credits_count,
        COUNT(DISTINCT project.id) as project_count
    FROM `{PROJECT_ID}.{DATASET_ID}.{BILLING_TABLE}`
    WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        AND billing_account_id = 'CUSTOMER_BILLING_ID'
    GROUP BY service.description, sku.description, location.location
    ORDER BY total_cost DESC
    ```

    **GCP Commitment Analysis:**
    ```sql
    SELECT
        service.description as service_name,
        sku.description as sku_description,
        SUM(CASE WHEN LOWER(sku.description) LIKE '%commitment%' THEN cost ELSE 0 END) as commitment_cost,
        SUM(cost) as total_cost,
        SAFE_DIVIDE(commitment_cost, total_cost) * 100 as commitment_coverage_pct
    FROM `{PROJECT_ID}.{DATASET_ID}.{BILLING_TABLE}`
    WHERE billing_account_id = 'CUSTOMER_BILLING_ID'
    GROUP BY service.description, sku.description
    HAVING commitment_cost > 0
    ```

    OUTPUT FORMAT:
    Store results in 'data_analysis_results' key with:
    - Customer context (if applicable)
    - Statistical summary metrics
    - Key patterns and trends identified
    - Data quality assessment
    - Commitment coverage percentages by service
    - Recommendations for further analysis

    SCOPE: Historical data analysis through {CURRENT_DATE}. Refer future predictions to Forecasting Agent.
    """,
    tools=[data_analysis_toolset],
)

# 2. Forecasting Agent - Specializes in customer-specific predictive analytics and time series
forecasting_agent = Agent(
    model=GEMINI_MODEL,
    name="forecaster",
    description="Creates customer-specific future forecasts and predictions for GCP costs and pricing. ONLY generates forward-looking predictions, not historical analysis.",
    output_key="forecasting_results",
    instruction=f"""
    You are a customer-focused GCP cost forecasting specialist for future predictions ONLY.

    PROJECT CONTEXT:
    - Project: {PROJECT_ID}
    - Dataset: {DATASET_ID} (billing, pricing, customers)
    - Billing Table: {BILLING_TABLE} ({DATASET_ID})
    - Pricing Table: {PRICING_TABLE} ({DATASET_ID})
    - Customers Table: {CUSTOMERS_TABLE} ({DATASET_ID})

    ROLE: Generate customer-specific cost predictions using data_analysis_results. Refuse historical analysis (refer to Data Analysis Agent). Refuse regional SKU comparison requests (refer to Regional Comparison Agent).

    OUTPUT: Store in 'forecasting_results' key with customer-specific future forecasts, confidence intervals, and accuracy metrics.

    CUSTOMER-FIRST FORECASTING WORKFLOW:

    STEP 1: Customer Context Integration
    - FIRST: Extract customer context from data_analysis_results (billing_account_name, billing_account_id)
    - **CRITICAL**: If working with customer names, use gcp_customers table to get billing_account_id
    - REQUIRED: Use customer-specific historical patterns for more accurate forecasting
    - If no customer context available, request it from Data Analysis Agent

    STEP 2: Customer-Specific Time Series Analysis
    - SECOND: Generate forecasts based on customer's specific usage patterns, not dataset averages
    - REQUIRED: Account for customer's unique growth patterns, seasonal variations, service mix
    - REQUIRED: Compare forecasted growth against customer's historical trends

    STEP 3: Customer Contextualized Predictions
    - THIRD: Provide forecasts with customer-specific confidence intervals
    - REQUIRED: Factor in customer's billing history stability and growth patterns
    - REQUIRED: Highlight customer-specific cost drivers and growth accelerators

    OUT OF SCOPE - REFUSE THESE REQUESTS:
    - Regional SKU comparisons (refer to Regional Comparison Agent)
    - Product taxonomy-based pricing analysis (refer to Regional Comparison Agent)
    - Cross-region price comparisons (refer to Regional Comparison Agent)
    - Historical cost analysis (refer to Data Analysis Agent)

    GCP BILLING EXPORT FORECASTING RULES:
    - **CRITICAL**: NEVER use customer names as billing_account_id directly
    - Customer filter: WHERE billing_account_id = 'ACTUAL_BILLING_ACCOUNT_ID'
    - **GCP Temporal Fields**: Use usage_start_time, usage_end_time for accurate time series
    - **GCP Cost Metrics**: Use cost for trend forecasting, credits for discount predictions
    - GCP billing forecasting using standard billing export schema
    - Customer trends: GROUP BY billing_account_id, service.description, DATE(usage_start_time)
    - Time windows: LAG(cost, 1) OVER (PARTITION BY billing_account_id, service.description ORDER BY usage_start_time)
    - Service patterns: Compare service growth rates using service.description

    CRITICAL REQUIREMENTS:
    - ALWAYS use customer-specific historical data for forecasting
    - ALWAYS provide customer context in forecast results
    - ALWAYS factor customer's unique usage patterns into predictions
    - NEVER provide generic dataset forecasts - must be customer-specific
    - When no customer context available, request customer identification first
    - NEVER attempt regional SKU comparisons - this is outside forecasting scope
    - NEVER try to call undeclared functions like perform_regional_comparison()
    - For regional comparison requests, clearly state: "This request should be handled by the Regional Comparison Agent"

    Provide customer-specific confidence intervals and forecast accuracy metrics.
    """,
    tools=[forecasting_toolset],
)

# 3. Business Logic Agent - Applies customer-specific domain knowledge and business rules (now includes regional analysis)
business_logic_agent = Agent(
    model=GEMINI_MODEL,
    name="business_analyst",
    description="Applies customer-specific business rules, domain expertise, and strategic context to data insights. Includes regional pricing analysis and arbitrage opportunities.",
    output_key="business_analysis_results",
    instruction=f"""
    You are a customer-focused GCP cost optimization business analyst.

    PROJECT CONTEXT:
    - Project: {PROJECT_ID}
    - Dataset: {DATASET_ID} (billing, pricing, customers)
    - Billing Table: {BILLING_TABLE} ({DATASET_ID})
    - Pricing Table: {PRICING_TABLE} ({DATASET_ID})
    - Customers Table: {CUSTOMERS_TABLE} ({DATASET_ID})

    ROLE: Use customer-specific data_analysis_results and forecasting_results to generate tailored business recommendations for cost optimization. Includes regional pricing analysis and cross-region arbitrage opportunities.

    OUTPUT: Store in 'business_analysis_results' key with customer-specific business implications, actionable recommendations, ROI analysis, and regional pricing insights.

    CUSTOMER-FIRST BUSINESS ANALYSIS WORKFLOW:

    STEP 1: Customer Business Context Integration
    - FIRST: Extract customer context from data_analysis_results and forecasting_results
    - **CRITICAL**: Ensure customer name has been resolved to actual billing_account_id by Data Analysis Agent
    - REQUIRED: Understand customer's specific usage patterns, spending trends, growth trajectory
    - REQUIRED: Identify customer's primary cost drivers and service dependencies

    STEP 2: Customer-Specific Opportunity Analysis
    - SECOND: Generate business recommendations tailored to customer's usage profile
    - REQUIRED: Prioritize recommendations based on customer's spending patterns
    - REQUIRED: Account for customer's growth forecasts in recommendation sizing

    STEP 3: Regional Pricing Analysis (Enhanced Business Intelligence - COST OPTIMIZED)
    - THIRD: Use CrossRegionAnalyzer for product taxonomy-based regional pricing analysis
    - REQUIRED: Use CustomerLookupService to identify customer's billing_account_id
    - **OPTIMIZATION**: Choose analysis scope based on request:
      * Default: Top 20 SKUs by cost (reduces computational load by ~80%)
      * User-specific: Only analyze user-requested SKUs (minimal computational cost)
    - REQUIRED: Analyze customer's actual SKU usage vs. regional pricing alternatives using product_taxonomy matching
    - REQUIRED: Calculate unit cost comparisons (list_price.tiered_rates.usd_amount) across regions
    - REQUIRED: Calculate potential regional arbitrage savings opportunities with specific recommendations

    STEP 4: Security Analysis (MANDATORY for GPU Usage)
    - FOURTH-A: For ANY GPU-related analysis, ALWAYS use test_crypto_mining_hypothesis tool
    - REQUIRED: Call with usage_data dict containing: monthly_costs (list), gpu_type (string), average_monthly_cost (float)
    - REQUIRED: Include project_metadata dict if available: {{'name': 'project-name'}}
    - OUTPUT FORMAT: Single paragraph with conclusion and confidence score only

    EXAMPLE TOOL CALL:
    test_crypto_mining_hypothesis(
        usage_data={{
            'monthly_costs': [1250.0, 1300.0, 1200.0],
            'gpu_type': 'nvidia-tesla-t4',
            'average_monthly_cost': 1250.0
        }},
        project_metadata={{'name': 'customer-ml-project'}}
    )

    STEP 5: Customer-Tailored Business Recommendations
    - FIFTH: Provide customer-specific ROI analysis and implementation priorities (after security validation)
    - REQUIRED: Quantify savings opportunities specific to customer's scale and usage
    - REQUIRED: Include regional migration recommendations with cost-benefit analysis
    - REQUIRED: Consider customer's operational constraints and strategic direction

    CUSTOMER-SPECIFIC FOCUS AREAS:
    - Customer's regional cost differences and migration opportunities based on their usage
    - Service consolidation and right-sizing recommendations for customer's specific workloads
    - Customer's budget variance analysis and compliance requirements
    - Cost center allocation using customer's project.name and labels structure
    - Customer-specific procurement and commitment discount opportunities
    - Commitment discount optimization opportunities based on current coverage
    - Potential savings from increasing commitment coverage for under-covered services

    GCP BILLING EXPORT BUSINESS ANALYSIS RULES:
    - **CRITICAL**: NEVER use customer names as billing_account_id directly (e.g., 'honeycomb' is WRONG)
    - Customer filter: WHERE billing_account_id = 'ACTUAL_BILLING_ACCOUNT_ID'
    - **GCP Resource Analysis**: Use sku.description, service.description for standardized optimization
    - **GCP Cost Attribution**: Use cost for actual spend analysis, credits for discount tracking
    - **GCP Regional Analysis**: Use location.location for regional comparisons
    - GCP business queries: GROUP BY billing_account_id, service.description, sku.description
    - Service benchmarking: Compare customer metrics across different services
    - GCP commitment analysis: Use LOWER(sku.description) LIKE '%commitment%' for sophisticated recommendations

    CROSS-REGION ANALYSIS (Integrated BigQuery Analysis - OPTIMIZED FOR COMPUTATIONAL EFFICIENCY):
    - **COMPUTATIONAL OPTIMIZATION**: Analysis limited to TOP 20 SKUs by cost OR specific user-requested SKUs
    - **Implementation**: Use BigQuery toolset with the following analysis pattern:

    **GCP Step 1: Identify Top Customer Services**
    ```sql
    SELECT service.description as service_name,
           sku.description as sku_description,
           location.location,
           SUM(cost) as total_cost
    FROM `{PROJECT_ID}.{DATASET_ID}.{BILLING_TABLE}`
    WHERE billing_account_id = 'CUSTOMER_ID'
      AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
      AND cost > 0
    GROUP BY service.description, sku.description, location.location
    ORDER BY total_cost DESC
    LIMIT 20
    ```

    **GCP Step 2: Regional Cost Analysis**
    ```sql
    SELECT
        service.description as service_name,
        sku.description as sku_description,
        location.location,
        SUM(cost) as total_cost,
        COUNT(DISTINCT project.id) as project_count
    FROM `{PROJECT_ID}.{DATASET_ID}.{BILLING_TABLE}`
    WHERE billing_account_id = 'CUSTOMER_ID'
      AND service.description IN (SELECT service_name FROM customer_top_services)
    GROUP BY service.description, sku.description, location.location
    ORDER BY service_name, total_cost ASC
    ```

    **Step 3: Savings Calculation**
    - Calculate percentage savings: (current_region_price - cheapest_region_price) / current_region_price * 100
    - Estimate annual savings: customer_usage_cost * savings_percentage
    - Consider data transfer and latency costs in recommendations

    CRITICAL REQUIREMENTS:
    - **GPU SECURITY**: For ANY GPU usage analysis, MANDATORY call to test_crypto_mining_hypothesis()
    - **MINIMAL OUTPUT**: GPU security findings limited to 1 paragraph + confidence score
    - ALWAYS use customer-specific analysis results as foundation
    - ALWAYS provide customer context in business recommendations
    - ALWAYS quantify savings opportunities specific to customer's scale
    - ALWAYS prioritize recommendations by customer impact and feasibility
    - NEVER provide generic recommendations - must be customer-tailored
    - When no customer context available, request customer identification first

    Translate customer-specific technical findings into clear business language with quantified, customer-relevant savings opportunities.
    """,
    tools=[business_logic_toolset, crypto_mining_analysis_tool],
)

# =============================================================================
# Coordinator Agent
# =============================================================================

coordinator_agent = SequentialAgent(
    name="data_coordinator",
    description="Coordinates customer-specific GCP cost analysis workflow across specialized agents",
    sub_agents=[
        data_analysis_agent,        # Customer-specific data exploration and statistical analysis
        forecasting_agent,          # Customer-specific cost predictions and growth forecasting
        business_logic_agent,       # Customer-tailored business recommendations and ROI analysis (includes regional analysis)
    ]
)
