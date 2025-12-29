"""
Shared services for the multi-agent BigQuery system.

This module provides common functionality used across multiple agents,
eliminating code redundancy and improving maintainability.

Configuration is loaded from environment variables for deployment flexibility.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from google.adk.tools.bigquery import BigQueryToolset
import logging

# Import cross-region analyzer for regional pricing analysis
try:
    from .cross_region_analyzer import CrossRegionAnalyzer
except ImportError:
    # Fallback if cross_region_analyzer is not available
    CrossRegionAnalyzer = None

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration from Environment Variables
# =============================================================================

def get_config() -> Dict[str, str]:
    """Get configuration from environment variables."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")

    dataset_id = os.environ.get("BQ_DATASET_ID", "detailed_billing")
    billing_table = os.environ.get("BQ_BILLING_TABLE")
    if not billing_table:
        raise ValueError("BQ_BILLING_TABLE environment variable is required")

    pricing_table = os.environ.get("BQ_PRICING_TABLE", "cloud_pricing_export")
    customers_table = os.environ.get("BQ_CUSTOMERS_TABLE", "gcp_customers")

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "billing_table": billing_table,
        "pricing_table": pricing_table,
        "customers_table": customers_table,
    }

# Global instance for cross-region analysis to avoid repeated initialization
_cross_region_analyzer = None


def analyze_cross_region_opportunities(billing_account_id: str,
                                     time_range: int = 30,
                                     current_date: str = None,
                                     max_skus: int = 20,
                                     specific_skus: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Wrapper function for CrossRegionAnalyzer.analyze_cross_region_opportunities
    that can be registered as an ADK tool.

    This function creates a global CrossRegionAnalyzer instance and delegates
    the call to its analyze_cross_region_opportunities method.

    Args:
        billing_account_id: Customer's billing account ID
        time_range: Number of days to analyze (default: 30)
        current_date: Current date for analysis (auto-detected if None)
        max_skus: Maximum number of top SKUs to analyze (default: 20)
        specific_skus: Optional list of specific SKU IDs to analyze

    Returns:
        Dictionary containing cross-region analysis results
    """
    global _cross_region_analyzer

    try:
        # Initialize analyzer if not already done
        if _cross_region_analyzer is None and CrossRegionAnalyzer is not None:
            config = get_config()
            project_id = config["project_id"]
            dataset_id = config["dataset_id"]
            billing_table = f"{project_id}.{dataset_id}.{config['billing_table']}"
            pricing_table = f"{project_id}.{dataset_id}.{config['pricing_table']}"

            # We'll need to create a minimal toolset for the analyzer
            import google.auth
            from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset
            from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode

            credentials, _ = google.auth.default()
            creds_config = BigQueryCredentialsConfig(credentials=credentials)
            tool_config = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)
            toolset = BigQueryToolset(credentials_config=creds_config, bigquery_tool_config=tool_config)

            _cross_region_analyzer = CrossRegionAnalyzer(
                toolset=toolset,
                project_id=project_id,
                dataset_id=dataset_id,
                billing_table=billing_table,
                pricing_table=pricing_table
            )

        # Call the actual analyzer method
        if _cross_region_analyzer is not None:
            return _cross_region_analyzer.analyze_cross_region_opportunities(
                billing_account_id=billing_account_id,
                time_range=time_range,
                current_date=current_date,
                max_skus=max_skus,
                specific_skus=specific_skus
            )
        else:
            return {
                'status': 'error',
                'message': 'CrossRegionAnalyzer not available - ensure all dependencies are installed',
                'recommendations': []
            }

    except Exception as e:
        logger.error(f"Error in analyze_cross_region_opportunities wrapper: {e}")
        return {
            'status': 'error',
            'message': f'Cross-region analysis failed: {str(e)}',
            'recommendations': []
        }


async def get_current_date_from_billing_data(toolset: BigQueryToolset,
                                             project_id: str = None,
                                             dataset_id: str = None,
                                             billing_table: str = None) -> str:
    """
    Get the current date by querying the MAX(usage_start_time) from the billing table.

    Args:
        toolset: BigQuery toolset for database operations
        project_id: GCP project ID (defaults to env var)
        dataset_id: BigQuery dataset ID (defaults to env var)
        billing_table: Billing table name (defaults to env var)

    Returns:
        Current date string in format "Month Day, Year" (e.g., "September 1, 2025")
    """
    try:
        # Get config from environment if not provided
        if not all([project_id, dataset_id, billing_table]):
            config = get_config()
            project_id = project_id or config["project_id"]
            dataset_id = dataset_id or config["dataset_id"]
            billing_table = billing_table or config["billing_table"]

        full_table_name = f"{project_id}.{dataset_id}.{billing_table}"

        query = f"""
        SELECT
            MAX(usage_start_time) as latest_date,
            FORMAT_DATE('%B %e, %Y', DATE(MAX(usage_start_time))) as formatted_date
        FROM `{full_table_name}`
        WHERE usage_start_time IS NOT NULL
        """

        # Execute query using the toolset's execute_sql tool
        tools = await toolset.get_tools()
        execute_sql_tools = [tool for tool in tools if tool.tool_name == 'execute_sql']
        if not execute_sql_tools:
            logger.error("execute_sql tool not found in toolset")
            from datetime import datetime
            return datetime.now().strftime('%B %-d, %Y')

        execute_sql_tool = execute_sql_tools[0]
        result = await execute_sql_tool.run(sql=query)

        if result and hasattr(result, 'data') and result.data:
            formatted_date = result.data[0].get('formatted_date')
            if formatted_date:
                logger.info(f"Retrieved current date from billing data: {formatted_date}")
                return formatted_date

        logger.warning("No date found in billing data, using system date fallback")
        from datetime import datetime
        return datetime.now().strftime('%B %-d, %Y')

    except Exception as e:
        logger.error(f"Error retrieving current date from billing data: {str(e)}")
        from datetime import datetime
        return datetime.now().strftime('%B %-d, %Y')


def get_current_date_sync(project_id: str = None,
                         dataset_id: str = None,
                         billing_table: str = None) -> str:
    """
    Synchronous version that gets current date from billing data.
    This creates its own toolset to query the current date.

    Returns:
        Current date string in format "Month Day, Year" (e.g., "September 1, 2025")
    """
    try:
        import asyncio
        import google.auth
        from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset
        from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode

        # Get config from environment if not provided
        if not all([project_id, dataset_id, billing_table]):
            config = get_config()
            project_id = project_id or config["project_id"]
            dataset_id = dataset_id or config["dataset_id"]
            billing_table = billing_table or config["billing_table"]

        # Setup BigQuery connection
        credentials, _ = google.auth.default()
        creds_config = BigQueryCredentialsConfig(credentials=credentials)
        tool_config = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)
        toolset = BigQueryToolset(credentials_config=creds_config, bigquery_tool_config=tool_config)

        # Run the async function
        return asyncio.run(get_current_date_from_billing_data(toolset, project_id, dataset_id, billing_table))

    except Exception as e:
        logger.error(f"Error in sync date retrieval: {str(e)}")
        from datetime import datetime
        return datetime.now().strftime('%B %-d, %Y')


class CustomerLookupService:
    """
    Centralized customer lookup service that eliminates redundant
    customer identification code across all agents.
    """

    def __init__(self, toolset: BigQueryToolset, project_id: str = None, dataset_id: str = None):
        """
        Initialize the CustomerLookupService.

        Args:
            toolset: BigQuery toolset for database operations
            project_id: GCP project ID (defaults to env var)
            dataset_id: BigQuery dataset ID (defaults to env var)
        """
        # Get config from environment if not provided
        if not all([project_id, dataset_id]):
            config = get_config()
            project_id = project_id or config["project_id"]
            dataset_id = dataset_id or config["dataset_id"]
            billing_table = config["billing_table"]
            customers_table = config["customers_table"]
        else:
            billing_table = os.environ.get("BQ_BILLING_TABLE", "")
            customers_table = os.environ.get("BQ_CUSTOMERS_TABLE", "gcp_customers")

        self.toolset = toolset
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.customers_table = f"{project_id}.{dataset_id}.{customers_table}"
        self.billing_table = f"{project_id}.{dataset_id}.{billing_table}"

        # Cache for customer lookup results
        self._customer_cache: Dict[str, Dict[str, Any]] = {}

    def lookup_customer(self, customer_name: str) -> Dict[str, Any]:
        """
        Look up customer by name with intelligent partial matching.

        Args:
            customer_name: Full or partial customer name

        Returns:
            Dictionary containing customer information:
            - status: 'single_match', 'multiple_matches', 'no_match'
            - customers: List of matching customer records
            - billing_account_id: Single billing account (if single match)
            - error: Error message if lookup failed
        """

        # Check cache first
        cache_key = customer_name.lower().strip()
        if cache_key in self._customer_cache:
            logger.info(f"Customer lookup cache hit for: {customer_name}")
            return self._customer_cache[cache_key]

        result = {
            'status': 'no_match',
            'customers': [],
            'billing_account_id': None,
            'customer_name': None,
            'error': None
        }

        try:
            # Step 1: Primary lookup in gcp_customers table
            customer_query = f"""
            SELECT
              billing_account_id,
              billing_account_name
            FROM `{self.customers_table}`
            WHERE LOWER(billing_account_name) LIKE LOWER('%{customer_name}%')
            ORDER BY billing_account_name
            """

            customer_results = self._execute_query(customer_query)

            if customer_results and len(customer_results) > 0:
                result['customers'] = customer_results

                if len(customer_results) == 1:
                    # Single match - perfect scenario
                    result['status'] = 'single_match'
                    result['billing_account_id'] = customer_results[0]['billing_account_id']
                    result['customer_name'] = customer_results[0]['billing_account_name']
                    logger.info(f"Single customer match found: {result['customer_name']}")
                else:
                    # Multiple matches - user needs to specify
                    result['status'] = 'multiple_matches'
                    logger.info(f"Multiple customer matches found: {len(customer_results)} matches")
            else:
                # No matches in customers table - try fallback methods
                result = self._fallback_customer_lookup(customer_name, result)

        except Exception as e:
            result['error'] = f"Customer lookup failed: {str(e)}"
            logger.error(f"Customer lookup error for {customer_name}: {str(e)}")

        # Cache the result
        self._customer_cache[cache_key] = result

        return result

    def _fallback_customer_lookup(self, customer_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback customer lookup methods when primary lookup fails.

        Args:
            customer_name: Customer name to search for
            result: Existing result dictionary to update

        Returns:
            Updated result dictionary
        """
        try:
            # Fallback 1: Search in billing table project names
            fallback_query1 = f"""
            SELECT DISTINCT
              billing_account_id,
              project.name as project_name
            FROM `{self.billing_table}`
            WHERE LOWER(project.name) LIKE LOWER('%{customer_name}%')
            LIMIT 10
            """

            fallback_results = self._execute_query(fallback_query1)

            if fallback_results and len(fallback_results) > 0:
                result['status'] = 'fallback_match' if len(fallback_results) == 1 else 'multiple_fallback_matches'
                result['customers'] = fallback_results

                if len(fallback_results) == 1:
                    result['billing_account_id'] = fallback_results[0]['billing_account_id']
                    result['customer_name'] = fallback_results[0]['project_name']
                    logger.info(f"Fallback customer match found in billing table: {result['customer_name']}")

                return result

            # Fallback 2: Direct billing account ID search
            if customer_name.replace('-', '').replace('_', '').isalnum():
                fallback_query2 = f"""
                SELECT DISTINCT
                  billing_account_id
                FROM `{self.billing_table}`
                WHERE billing_account_id LIKE '%{customer_name}%'
                LIMIT 5
                """

                fallback_results2 = self._execute_query(fallback_query2)

                if fallback_results2 and len(fallback_results2) > 0:
                    result['status'] = 'billing_id_match'
                    result['customers'] = fallback_results2

                    if len(fallback_results2) == 1:
                        result['billing_account_id'] = fallback_results2[0]['billing_account_id']
                        result['customer_name'] = f"Account {fallback_results2[0]['billing_account_id']}"
                        logger.info(f"Billing ID match found: {result['billing_account_id']}")

        except Exception as e:
            result['error'] = f"Fallback customer lookup failed: {str(e)}"
            logger.error(f"Fallback lookup error for {customer_name}: {str(e)}")

        return result

    def validate_customer_access(self, billing_account_id: str) -> bool:
        """
        Validate that the billing account has accessible data.

        Args:
            billing_account_id: Billing account ID to validate

        Returns:
            True if customer has accessible billing data, False otherwise
        """
        try:
            validation_query = f"""
            SELECT COUNT(*) as record_count
            FROM `{self.billing_table}`
            WHERE billing_account_id = '{billing_account_id}'
            LIMIT 1
            """

            result = self._execute_query(validation_query)

            if result and len(result) > 0:
                record_count = result[0].get('record_count', 0)
                return record_count > 0

        except Exception as e:
            logger.error(f"Customer validation error for {billing_account_id}: {str(e)}")

        return False

    def get_customer_summary(self, billing_account_id: str) -> Dict[str, Any]:
        """
        Get basic customer summary information.

        Args:
            billing_account_id: Customer's billing account ID

        Returns:
            Dictionary with customer summary data
        """
        try:
            summary_query = f"""
            SELECT
              billing_account_id,
              COUNT(DISTINCT project.id) as project_count,
              COUNT(DISTINCT service.id) as service_count,
              SUM(cost) as total_cost,
              MIN(usage_start_time) as earliest_usage,
              MAX(usage_start_time) as latest_usage
            FROM `{self.billing_table}`
            WHERE billing_account_id = '{billing_account_id}'
              AND cost IS NOT NULL
            GROUP BY billing_account_id
            """

            result = self._execute_query(summary_query)

            if result and len(result) > 0:
                return {
                    'status': 'success',
                    'summary': result[0],
                    'error': None
                }

        except Exception as e:
            logger.error(f"Customer summary error for {billing_account_id}: {str(e)}")

        return {
            'status': 'error',
            'summary': None,
            'error': f"Failed to get customer summary"
        }

    def _execute_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a BigQuery query using the toolset.

        Args:
            query: SQL query to execute

        Returns:
            Query results as list of dictionaries, or None if failed
        """
        try:
            logger.debug(f"Executing query: {query}")
            result = self.toolset.execute_sql(query)
            return result

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return None

    def clear_cache(self):
        """Clear the customer lookup cache."""
        self._customer_cache.clear()
        logger.info("Customer lookup cache cleared")


class QueryRouter:
    """
    Intelligent query routing service to determine optimal agent workflow.
    """

    def __init__(self):
        """Initialize the QueryRouter."""
        pass

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify incoming query to determine optimal processing workflow.

        Args:
            query: User query to classify

        Returns:
            Dictionary with routing recommendations:
            - workflow_type: 'sequential', 'parallel', 'single_agent'
            - primary_agent: Main agent to handle the query
            - supporting_agents: List of additional agents needed
            - estimated_complexity: 'low', 'medium', 'high'
        """

        query_lower = query.lower()

        # Keywords for different agent types
        data_keywords = ['analyze', 'statistics', 'data', 'pattern', 'trend', 'explore']
        forecast_keywords = ['predict', 'forecast', 'future', 'projection', 'growth', 'trend']
        business_keywords = ['cost', 'savings', 'optimization', 'recommendation', 'business']
        regional_keywords = ['region', 'location', 'geography', 'compare regions', 'pricing']

        # Analyze query content
        has_data_focus = any(keyword in query_lower for keyword in data_keywords)
        has_forecast_focus = any(keyword in query_lower for keyword in forecast_keywords)
        has_business_focus = any(keyword in query_lower for keyword in business_keywords)
        has_regional_focus = any(keyword in query_lower for keyword in regional_keywords)

        # Determine workflow type
        focus_count = sum([has_data_focus, has_forecast_focus, has_business_focus, has_regional_focus])

        if focus_count >= 3:
            workflow_type = 'sequential'
            complexity = 'high'
        elif focus_count == 2:
            workflow_type = 'parallel'
            complexity = 'medium'
        else:
            workflow_type = 'single_agent'
            complexity = 'low'

        # Determine primary agent
        if has_forecast_focus:
            primary_agent = 'forecasting_agent'
        elif has_regional_focus:
            primary_agent = 'business_logic_agent'
        elif has_business_focus:
            primary_agent = 'business_logic_agent'
        else:
            primary_agent = 'data_analysis_agent'

        # Determine supporting agents
        supporting_agents = []
        if workflow_type != 'single_agent':
            if has_data_focus and primary_agent != 'data_analysis_agent':
                supporting_agents.append('data_analysis_agent')
            if has_forecast_focus and primary_agent != 'forecasting_agent':
                supporting_agents.append('forecasting_agent')
            if has_business_focus and primary_agent != 'business_logic_agent':
                supporting_agents.append('business_logic_agent')

        return {
            'workflow_type': workflow_type,
            'primary_agent': primary_agent,
            'supporting_agents': supporting_agents,
            'estimated_complexity': complexity,
            'analysis': {
                'data_focus': has_data_focus,
                'forecast_focus': has_forecast_focus,
                'business_focus': has_business_focus,
                'regional_focus': has_regional_focus
            }
        }


class ResultSynthesizer:
    """
    Lightweight result synthesis service to replace the heavy visualization agent.
    """

    def __init__(self, customer_lookup_service: CustomerLookupService):
        """
        Initialize the ResultSynthesizer.

        Args:
            customer_lookup_service: CustomerLookupService instance for customer context
        """
        self.customer_lookup = customer_lookup_service

    def synthesize_results(self, agent_results: Dict[str, Any], customer_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Synthesize results from multiple agents into a cohesive report.

        Args:
            agent_results: Dictionary of results from various agents
            customer_context: Customer context information

        Returns:
            Synthesized report dictionary
        """

        synthesis = {
            'executive_summary': self._create_executive_summary(agent_results, customer_context),
            'key_findings': self._extract_key_findings(agent_results),
            'recommendations': self._compile_recommendations(agent_results),
            'customer_context': customer_context,
            'data_sources': list(agent_results.keys())
        }

        return synthesis

    def _create_executive_summary(self, agent_results: Dict[str, Any], customer_context: Dict[str, Any]) -> str:
        """Create executive summary from agent results."""

        customer_name = customer_context.get('customer_name', 'Unknown Customer') if customer_context else 'All Customers'

        summary_parts = [
            f"Analysis Summary for {customer_name}",
            "=" * (len(f"Analysis Summary for {customer_name}"))
        ]

        if 'data_analysis_results' in agent_results:
            summary_parts.append("• Data analysis completed with statistical insights")

            analysis = agent_results['data_analysis_results']
            if isinstance(analysis, dict) and 'commitment_coverage' in analysis:
                coverage = analysis['commitment_coverage']
                if 'overall_coverage_percentage' in coverage:
                    summary_parts.append(f"• Commitment coverage analysis completed: {coverage['overall_coverage_percentage']:.1f}% coverage")

        if 'forecasting_results' in agent_results:
            summary_parts.append("• Future cost predictions generated")

        if 'business_analysis_results' in agent_results:
            summary_parts.append("• Business optimization recommendations identified")

        if 'regional_comparison_results' in agent_results:
            summary_parts.append("• Regional pricing analysis completed")

        return "\n".join(summary_parts)

    def _extract_key_findings(self, agent_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from agent results."""

        findings = []

        for agent_name, results in agent_results.items():
            if isinstance(results, dict) and 'key_insights' in results:
                findings.extend(results['key_insights'])
            elif isinstance(results, str) and len(results) > 0:
                first_sentence = results.split('.')[0] + '.'
                findings.append(f"{agent_name}: {first_sentence}")

        return findings

    def _compile_recommendations(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile actionable recommendations from agent results."""

        recommendations = []

        if 'business_analysis_results' in agent_results:
            business_results = agent_results['business_analysis_results']
            if isinstance(business_results, dict) and 'recommendations' in business_results:
                recommendations.extend(business_results['recommendations'])

        if 'forecasting_results' in agent_results:
            forecast_results = agent_results['forecasting_results']
            if isinstance(forecast_results, dict) and 'recommendations' in forecast_results:
                recommendations.extend(forecast_results['recommendations'])

        return recommendations


class CommitmentCoverageAnalyzer:
    """
    Analyzes commitment discount coverage for services using sku.description partial matching.
    """

    def __init__(self, toolset: Any = None, project_id: str = None,
                 dataset_id: str = None, cache_manager: Any = None):
        """
        Initialize the CommitmentCoverageAnalyzer.

        Args:
            toolset: BigQuery toolset for database operations
            project_id: GCP project ID (defaults to env var)
            dataset_id: BigQuery dataset ID (defaults to env var)
            cache_manager: Cache manager for performance optimization
        """
        # Get config from environment if not provided
        if not all([project_id, dataset_id]):
            config = get_config()
            project_id = project_id or config["project_id"]
            dataset_id = dataset_id or config["dataset_id"]
            billing_table = config["billing_table"]
        else:
            billing_table = os.environ.get("BQ_BILLING_TABLE", "")

        self.toolset = toolset
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.billing_table = f"{project_id}.{dataset_id}.{billing_table}"
        self.cache_manager = cache_manager

        # Cache for commitment analysis results
        self._commitment_cache: Dict[str, Dict[str, Any]] = {}

    def analyze_commitment_coverage(self, billing_account_id: str, time_range: int = 30,
                                  current_date: str = None) -> Dict[str, Any]:
        """
        Calculate commitment coverage percentage by service for a specific customer.

        Args:
            billing_account_id: Customer's billing account ID
            time_range: Analysis time range in days (default: 30)
            current_date: Current date for relative calculations

        Returns:
            Dictionary containing commitment coverage analysis
        """

        # Get current date dynamically if not provided
        if current_date is None:
            try:
                current_date = get_current_date_sync()
            except Exception as e:
                logger.warning(f"Could not retrieve dynamic date: {e}")
                from datetime import datetime
                current_date = datetime.now().strftime('%B %-d, %Y')

        # Check cache first
        cache_key = f"{billing_account_id}_{time_range}_{current_date}"
        if cache_key in self._commitment_cache:
            logger.info(f"Commitment coverage cache hit for: {billing_account_id}")
            return self._commitment_cache[cache_key]

        try:
            coverage_query = self._build_commitment_coverage_query(
                billing_account_id, time_range, current_date
            )

            if self.toolset:
                raw_results = self._execute_query(coverage_query)
            else:
                raw_results = self._generate_mock_commitment_data(billing_account_id)

            coverage_analysis = self._process_commitment_results(raw_results)

            coverage_analysis['recommendations'] = self._generate_commitment_recommendations(
                coverage_analysis['service_breakdown']
            )

            self._commitment_cache[cache_key] = coverage_analysis

            logger.info(f"Commitment coverage analysis completed for {billing_account_id}")
            return coverage_analysis

        except Exception as e:
            logger.error(f"Commitment coverage analysis failed for {billing_account_id}: {str(e)}")
            return {
                'error': str(e),
                'total_commitment_cost': 0,
                'total_eligible_cost': 0,
                'overall_coverage_percentage': 0,
                'service_breakdown': [],
                'recommendations': ['Unable to analyze commitment coverage due to data access issues']
            }

    def _build_commitment_coverage_query(self, billing_account_id: str, time_range: int,
                                       current_date: str) -> str:
        """Build the BigQuery SQL for commitment coverage analysis."""

        query = f"""
        WITH service_totals AS (
            SELECT
                service.description as service_name,
                SUM(cost) as total_cost,
                COUNT(*) as total_line_items
            FROM `{self.billing_table}`
            WHERE billing_account_id = '{billing_account_id}'
                AND usage_start_time >= TIMESTAMP_SUB(TIMESTAMP('{current_date}'), INTERVAL {time_range} DAY)
                AND usage_start_time <= TIMESTAMP('{current_date}')
                AND cost IS NOT NULL
                AND cost > 0
            GROUP BY service.description
        ),
        commitment_costs AS (
            SELECT
                service.description as service_name,
                SUM(cost) as commitment_cost,
                COUNT(*) as commitment_line_items,
                ARRAY_AGG(DISTINCT sku.description ORDER BY sku.description) as commitment_types
            FROM `{self.billing_table}`
            WHERE billing_account_id = '{billing_account_id}'
                AND usage_start_time >= TIMESTAMP_SUB(TIMESTAMP('{current_date}'), INTERVAL {time_range} DAY)
                AND usage_start_time <= TIMESTAMP('{current_date}')
                AND LOWER(sku.description) LIKE '%commitment%'
                AND cost IS NOT NULL
                AND cost > 0
            GROUP BY service.description
        ),
        coverage_analysis AS (
            SELECT
                st.service_name,
                st.total_cost,
                st.total_line_items,
                COALESCE(cc.commitment_cost, 0) as commitment_cost,
                COALESCE(cc.commitment_line_items, 0) as commitment_line_items,
                COALESCE(cc.commitment_types, []) as commitment_types,
                CASE
                    WHEN st.total_cost > 0 THEN
                        ROUND((COALESCE(cc.commitment_cost, 0) / st.total_cost) * 100, 2)
                    ELSE 0
                END as coverage_percentage,
                st.total_cost - COALESCE(cc.commitment_cost, 0) as uncommitted_cost
            FROM service_totals st
            LEFT JOIN commitment_costs cc ON st.service_name = cc.service_name
        )
        SELECT
            service_name,
            total_cost,
            commitment_cost,
            coverage_percentage,
            uncommitted_cost,
            total_line_items,
            commitment_line_items,
            commitment_types,
            CASE
                WHEN coverage_percentage = 0 AND total_cost > 1000 THEN 5
                WHEN coverage_percentage < 25 AND total_cost > 500 THEN 4
                WHEN coverage_percentage < 50 AND total_cost > 100 THEN 3
                ELSE 1
            END as optimization_priority
        FROM coverage_analysis
        ORDER BY total_cost DESC, optimization_priority DESC
        """

        return query

    def _process_commitment_results(self, raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw query results into structured commitment analysis."""

        if not raw_results:
            return {
                'total_commitment_cost': 0,
                'total_eligible_cost': 0,
                'overall_coverage_percentage': 0,
                'service_breakdown': [],
                'coverage_summary': 'No data available for commitment analysis'
            }

        total_eligible_cost = sum(row['total_cost'] for row in raw_results)
        total_commitment_cost = sum(row['commitment_cost'] for row in raw_results)
        overall_coverage_percentage = (
            (total_commitment_cost / total_eligible_cost * 100)
            if total_eligible_cost > 0 else 0
        )

        service_breakdown = []
        for row in raw_results:
            service_detail = {
                'service': row['service_name'],
                'total_cost': float(row['total_cost']),
                'commitment_cost': float(row['commitment_cost']),
                'coverage_percentage': float(row['coverage_percentage']),
                'uncommitted_cost': float(row['uncommitted_cost']),
                'optimization_priority': row.get('optimization_priority', 1),
                'commitment_types': row.get('commitment_types', []),
                'line_items': {
                    'total': row.get('total_line_items', 0),
                    'commitment': row.get('commitment_line_items', 0)
                }
            }
            service_breakdown.append(service_detail)

        well_covered_services = [s for s in service_breakdown if s['coverage_percentage'] >= 50]
        under_covered_services = [s for s in service_breakdown if s['coverage_percentage'] < 25]

        coverage_summary = f"Overall {overall_coverage_percentage:.1f}% coverage. "
        coverage_summary += f"{len(well_covered_services)} services well-covered (>=50%), "
        coverage_summary += f"{len(under_covered_services)} services under-covered (<25%)"

        return {
            'total_commitment_cost': round(total_commitment_cost, 2),
            'total_eligible_cost': round(total_eligible_cost, 2),
            'overall_coverage_percentage': round(overall_coverage_percentage, 2),
            'service_breakdown': service_breakdown,
            'coverage_summary': coverage_summary,
            'analysis_metadata': {
                'total_services_analyzed': len(raw_results),
                'services_with_commitments': len([s for s in service_breakdown if s['commitment_cost'] > 0]),
                'highest_priority_service': service_breakdown[0]['service'] if service_breakdown else None
            }
        }

    def _generate_commitment_recommendations(self, service_breakdown: List[Dict[str, Any]]) -> List[str]:
        """Generate commitment optimization recommendations based on coverage analysis."""

        recommendations = []

        if not service_breakdown:
            return ["No service data available for commitment recommendations"]

        prioritized_services = sorted(
            service_breakdown,
            key=lambda x: (x['optimization_priority'], x['uncommitted_cost']),
            reverse=True
        )

        high_priority = [s for s in prioritized_services if s['optimization_priority'] >= 4]
        if high_priority:
            for service in high_priority[:3]:
                if service['coverage_percentage'] == 0:
                    recommendations.append(
                        f"URGENT: Consider commitment discounts for {service['service']} "
                        f"({service['coverage_percentage']:.1f}% coverage, ${service['uncommitted_cost']:,.2f} potential)"
                    )
                else:
                    recommendations.append(
                        f"HIGH: Expand commitment coverage for {service['service']} "
                        f"({service['coverage_percentage']:.1f}% coverage, ${service['uncommitted_cost']:,.2f} additional potential)"
                    )

        medium_priority = [s for s in prioritized_services if s['optimization_priority'] == 3]
        if medium_priority:
            for service in medium_priority[:2]:
                recommendations.append(
                    f"MEDIUM: Optimize commitment coverage for {service['service']} "
                    f"({service['coverage_percentage']:.1f}% coverage, ${service['uncommitted_cost']:,.2f} opportunity)"
                )

        well_covered = [s for s in service_breakdown if s['coverage_percentage'] >= 75]
        if well_covered:
            top_covered = well_covered[0]
            recommendations.append(
                f"GOOD: {top_covered['service']} has excellent commitment coverage "
                f"({top_covered['coverage_percentage']:.1f}%)"
            )

        total_uncommitted = sum(s['uncommitted_cost'] for s in service_breakdown)
        if total_uncommitted > 1000:
            recommendations.append(
                f"STRATEGIC: Total commitment opportunity of ${total_uncommitted:,.2f} identified across all services"
            )

        return recommendations[:5]

    def _generate_mock_commitment_data(self, billing_account_id: str) -> List[Dict[str, Any]]:
        """Generate mock commitment data for testing/demonstration purposes."""

        mock_data = [
            {
                'service_name': 'Compute Engine',
                'total_cost': 8934.21,
                'commitment_cost': 4521.33,
                'coverage_percentage': 50.6,
                'uncommitted_cost': 4412.88,
                'optimization_priority': 3,
                'commitment_types': ['Compute Engine Committed Use', 'CPU Commitment'],
                'total_line_items': 1250,
                'commitment_line_items': 632
            },
            {
                'service_name': 'BigQuery',
                'total_cost': 2456.78,
                'commitment_cost': 0,
                'coverage_percentage': 0.0,
                'uncommitted_cost': 2456.78,
                'optimization_priority': 5,
                'commitment_types': [],
                'total_line_items': 890,
                'commitment_line_items': 0
            },
            {
                'service_name': 'Cloud Storage',
                'total_cost': 1847.63,
                'commitment_cost': 467.28,
                'coverage_percentage': 25.3,
                'uncommitted_cost': 1380.35,
                'optimization_priority': 4,
                'commitment_types': ['Storage Commitment'],
                'total_line_items': 3400,
                'commitment_line_items': 860
            },
            {
                'service_name': 'Cloud Functions',
                'total_cost': 1234.56,
                'commitment_cost': 185.18,
                'coverage_percentage': 15.0,
                'uncommitted_cost': 1049.38,
                'optimization_priority': 4,
                'commitment_types': ['Function Commitment'],
                'total_line_items': 567,
                'commitment_line_items': 85
            },
            {
                'service_name': 'Cloud SQL',
                'total_cost': 892.45,
                'commitment_cost': 669.34,
                'coverage_percentage': 75.0,
                'uncommitted_cost': 223.11,
                'optimization_priority': 1,
                'commitment_types': ['Database Commitment', 'SQL Instance Commitment'],
                'total_line_items': 234,
                'commitment_line_items': 175
            }
        ]

        return mock_data

    def _execute_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Execute a BigQuery query using the toolset."""
        try:
            logger.debug(f"Executing commitment coverage query")

            if hasattr(self.toolset, 'execute_sql'):
                result = self.toolset.execute_sql(query)
                return result
            else:
                logger.warning("Toolset does not support execute_sql, using mock data")
                return None

        except Exception as e:
            logger.error(f"Commitment coverage query execution failed: {str(e)}")
            return None

    def get_commitment_summary(self, billing_account_id: str, time_range: int = 30) -> str:
        """Get a concise commitment coverage summary for reporting."""

        analysis = self.analyze_commitment_coverage(billing_account_id, time_range)

        if analysis.get('error'):
            return f"Commitment analysis unavailable: {analysis['error']}"

        summary = f"Commitment Coverage: {analysis['overall_coverage_percentage']:.1f}% "
        summary += f"(${analysis['total_commitment_cost']:,.2f} of ${analysis['total_eligible_cost']:,.2f})"

        if analysis['service_breakdown']:
            top_opportunity = max(
                analysis['service_breakdown'],
                key=lambda x: x['uncommitted_cost']
            )
            summary += f" | Top opportunity: {top_opportunity['service']} "
            summary += f"({top_opportunity['coverage_percentage']:.1f}% coverage)"

        return summary

    def clear_cache(self):
        """Clear the commitment analysis cache."""
        self._commitment_cache.clear()
        logger.info("Commitment coverage cache cleared")


def test_crypto_mining_hypothesis(usage_data: Dict[str, Any],
                                 project_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Scientific hypothesis testing for crypto-mining detection.

    H0 (Null): GPU usage is legitimate business activity
    H1 (Alternative): GPU usage is unauthorized crypto-mining

    Args:
        usage_data: Dictionary containing usage patterns, costs, GPU types, etc.
        project_metadata: Optional project information for context

    Returns:
        Dict with keys: conclusion, confidence, evidence, summary
    """
    evidence = []
    confidence_score = 0.0

    project_score = _test_project_identity(usage_data, project_metadata, evidence)
    pattern_score = _test_usage_patterns(usage_data, evidence)
    gpu_score = _test_gpu_selection(usage_data, evidence)
    efficiency_score = _test_cost_efficiency(usage_data, evidence)

    confidence_score = (
        project_score * 0.3 +
        pattern_score * 0.3 +
        gpu_score * 0.25 +
        efficiency_score * 0.15
    )

    if confidence_score >= 0.7:
        conclusion = "ALERT"
        summary = f"High confidence ({confidence_score:.2f}) crypto-mining detected. Investigation recommended."
    elif confidence_score >= 0.3:
        conclusion = "INSUFFICIENT_DATA"
        summary = f"Moderate confidence ({confidence_score:.2f}). Additional evidence needed for determination."
    else:
        conclusion = "RULED_OUT"
        summary = f"Low mining probability ({confidence_score:.2f}). Usage patterns consistent with legitimate business operations."

    return {
        "conclusion": conclusion,
        "confidence": confidence_score,
        "evidence": evidence,
        "summary": summary
    }


def _test_project_identity(usage_data: Dict[str, Any],
                          project_metadata: Optional[Dict[str, Any]],
                          evidence: List[str]) -> float:
    """Test project identity for suspicious patterns."""
    score = 0.0

    if not project_metadata:
        evidence.append("Project metadata unavailable - cannot verify business purpose")
        return 0.4

    project_name = project_metadata.get('name', '').lower()
    suspicious_names = ['gpu', 'mining', 'compute', 'worker', 'node', 'temp', 'test']

    if any(name in project_name for name in suspicious_names):
        score += 0.3
        evidence.append(f"Suspicious project name pattern: {project_name}")
    else:
        evidence.append(f"Project name appears business-focused: {project_name}")

    return score


def _test_usage_patterns(usage_data: Dict[str, Any], evidence: List[str]) -> float:
    """Test usage patterns - mining is flat, business has variance."""
    score = 0.0

    monthly_costs = usage_data.get('monthly_costs', [])
    if len(monthly_costs) < 3:
        evidence.append("Insufficient historical data for pattern analysis")
        return 0.3

    import statistics
    if len(monthly_costs) > 1:
        mean_cost = statistics.mean(monthly_costs)
        stdev_cost = statistics.stdev(monthly_costs)
        cv = stdev_cost / mean_cost if mean_cost > 0 else 0

        if cv < 0.1:
            score += 0.4
            evidence.append(f"Unusually consistent usage (CV: {cv:.3f}) - typical of 24/7 operations")
        elif cv > 0.25:
            evidence.append(f"Variable usage patterns (CV: {cv:.3f}) - consistent with business workflows")
        else:
            evidence.append(f"Moderate usage variance (CV: {cv:.3f}) - within normal business range")

    return score


def _test_gpu_selection(usage_data: Dict[str, Any], evidence: List[str]) -> float:
    """Test GPU type selection - T4s are poor for mining."""
    score = 0.0

    gpu_type = usage_data.get('gpu_type', '').lower()

    if 't4' in gpu_type:
        evidence.append("Tesla T4 GPUs detected - suboptimal for crypto-mining, good for ML inference")
        score = 0.0
    elif any(gpu in gpu_type for gpu in ['v100', 'a100', 'rtx']):
        evidence.append(f"{gpu_type} GPUs could be used for mining but also common for ML training")
        score = 0.2
    else:
        evidence.append(f"GPU type: {gpu_type} - mining suitability unknown")
        score = 0.3

    return score


def _test_cost_efficiency(usage_data: Dict[str, Any], evidence: List[str]) -> float:
    """Test cost efficiency - miners optimize for cheapest compute."""
    score = 0.0

    monthly_cost = usage_data.get('average_monthly_cost', 0)

    if monthly_cost < 2000:
        evidence.append(f"Monthly cost ${monthly_cost:,.0f} suggests small-scale operation, not industrial mining")
        score = 0.1
    elif monthly_cost > 10000:
        evidence.append(f"High monthly cost ${monthly_cost:,.0f} could indicate large-scale mining operation")
        score = 0.3
    else:
        evidence.append(f"Moderate monthly cost ${monthly_cost:,.0f} within range for both mining and business use")
        score = 0.2

    return score
