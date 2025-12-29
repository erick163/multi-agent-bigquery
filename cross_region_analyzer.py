"""
Cross-Region Price Comparison Analyzer

This module implements the correct cross-region comparison logic using product_taxonomy
to find equivalent services across different regions and compare unit costs.

Implements the required 4-step method:
1. Billing SKU → Pricing SKU mapping with product_taxonomy
2. Use product_taxonomy to find equivalent services in other regions  
3. Compare unit costs (list_price.tiered_rates.usd_amount) across regions
4. Generate summary and recommendations for cross-region savings
"""

from typing import Dict, List, Any, Optional, Tuple
from google.adk.tools.bigquery import BigQueryToolset
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CrossRegionAnalyzer:
    """
    Analyzes cross-region pricing opportunities using product taxonomy matching.
    
    This analyzer identifies equivalent services across different regions and
    compares unit costs to find potential arbitrage opportunities.
    """
    
    def __init__(self, toolset: BigQueryToolset, project_id: str, dataset_id: str, 
                 billing_table: str, pricing_table: str):
        """
        Initialize the CrossRegionAnalyzer.
        
        Args:
            toolset: BigQuery toolset for database operations
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            billing_table: Name of the billing export table
            pricing_table: Name of the cloud pricing export table
        """
        self.toolset = toolset
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.billing_table = f"{project_id}.{dataset_id}.{billing_table}"
        self.pricing_table = f"{project_id}.{dataset_id}.{pricing_table}"
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
    
    def analyze_cross_region_opportunities(self, billing_account_id: str, 
                                         time_range: int = 30,
                                         current_date: str = None,
                                         max_skus: int = 20,
                                         specific_skus: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform optimized cross-region pricing analysis with computational cost controls.
        
        Args:
            billing_account_id: Customer's billing account ID
            time_range: Number of days to analyze (default: 30)
            current_date: Current date for analysis
            max_skus: Maximum number of top SKUs to analyze (default: 20, reduces computational cost)
            specific_skus: Optional list of specific SKU IDs to analyze (user-requested analysis)
            
        Returns:
            Dictionary containing cross-region analysis results
        """
        try:
            # Get current date dynamically if not provided
            if current_date is None:
                from .shared_services import get_current_date_sync
                try:
                    current_date = get_current_date_sync()
                except Exception as e:
                    logger.warning(f"Could not retrieve dynamic date: {e}")
                    from datetime import datetime
                    current_date = datetime.now().strftime('%B %d, %Y')  # Fallback to system date
            
            # Check cache first (include optimization parameters in cache key)
            cache_key = f"{billing_account_id}_{time_range}_{current_date}_{max_skus}_{len(specific_skus or [])}"
            if cache_key in self._analysis_cache:
                logger.info(f"Returning cached cross-region analysis for {billing_account_id}")
                return self._analysis_cache[cache_key]
            
            # Determine analysis scope for computational optimization
            analysis_scope = "specific_skus" if specific_skus else f"top_{max_skus}_skus"
            logger.info(f"Starting optimized cross-region analysis for {billing_account_id} (scope: {analysis_scope})")
            
            # Step 1: Get customer SKUs with pricing and product taxonomy (optimized)
            customer_skus = self._get_customer_sku_pricing(
                billing_account_id, time_range, current_date, max_skus, specific_skus
            )
            
            if not customer_skus:
                logger.warning(f"No SKU data found for billing_account_id: {billing_account_id}")
                return self._generate_empty_analysis()
            
            # Step 2: Find regional alternatives using product_taxonomy
            regional_alternatives = self._find_regional_alternatives(customer_skus)
            
            # Step 3: Calculate unit cost comparisons
            cost_comparisons = self._calculate_cost_comparisons(regional_alternatives)
            
            # Step 4: Generate summary and recommendations
            analysis_result = self._generate_cross_region_summary(cost_comparisons, billing_account_id)
            
            # Cache results
            self._analysis_cache[cache_key] = analysis_result
            
            logger.info(f"Cross-region analysis completed for {billing_account_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Cross-region analysis failed for {billing_account_id}: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _get_customer_sku_pricing(self, billing_account_id: str, time_range: int, 
                                current_date: str, max_skus: int = 20,
                                specific_skus: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Step 1: Get customer's SKUs with pricing and product taxonomy (optimized).
        
        Optimization strategies:
        1. If specific_skus provided: Only analyze those specific SKUs
        2. Otherwise: Limit to top N SKUs by cost to reduce computational load
        """
        query = self._build_customer_sku_query(
            billing_account_id, time_range, current_date, max_skus, specific_skus
        )
        
        try:
            result = self.toolset.execute_sql(query)
            scope = f"{len(specific_skus)} specific SKUs" if specific_skus else f"top {max_skus} SKUs by cost"
            logger.info(f"Found {len(result)} SKUs for optimized analysis (scope: {scope})")
            return result
        except Exception as e:
            logger.error(f"Failed to get customer SKU pricing: {str(e)}")
            return []
    
    def _build_customer_sku_query(self, billing_account_id: str, time_range: int, 
                                current_date: str, max_skus: int = 20,
                                specific_skus: Optional[List[str]] = None) -> str:
        """
        Build optimized SQL query for Step 1: Customer SKU to Pricing mapping.
        
        Optimization: Limits analysis to top N SKUs by cost or specific user-requested SKUs.
        """
        # Build specific SKU filter if provided
        sku_filter = ""
        if specific_skus:
            sku_list = "', '".join(specific_skus)
            sku_filter = f"AND sku.id IN ('{sku_list}')"
        
        # Build LIMIT clause for top N SKUs optimization
        limit_clause = f"LIMIT {max_skus}" if not specific_skus else ""
        
        return f"""
        -- Step 1: Get customer SKUs with pricing and product_taxonomy (OPTIMIZED)
        WITH customer_usage AS (
            SELECT DISTINCT 
                sku.id as customer_sku_id,
                service.description as service_name,
                SUM(cost) as total_usage_cost,
                SUM(usage.amount) as total_usage_amount,
                usage.unit as usage_unit
            FROM `{self.billing_table}`
            WHERE billing_account_id = '{billing_account_id}'
                AND usage_start_time >= TIMESTAMP_SUB(TIMESTAMP('{current_date}'), INTERVAL {time_range} DAY)
                AND usage_start_time <= TIMESTAMP('{current_date}')
                AND cost IS NOT NULL 
                AND cost > 0
                {sku_filter}  -- ✅ OPTIMIZATION: Specific SKUs filter
            GROUP BY sku.id, service.description, usage.unit
        ),
        customer_sku_pricing AS (
            SELECT 
                cu.customer_sku_id,
                cu.service_name,
                cu.total_usage_cost,
                cu.total_usage_amount,
                cu.usage_unit,
                p.list_price.tiered_rates[OFFSET(0)].usd_amount as current_unit_cost,
                p.product_taxonomy,
                p.geo_taxonomy.regions as current_regions
            FROM customer_usage cu
            JOIN `{self.pricing_table}` p ON cu.customer_sku_id = p.sku.id
            WHERE p._PARTITIONTIME = (SELECT MAX(_PARTITIONTIME) FROM `{self.pricing_table}`)
                AND p.list_price.tiered_rates[OFFSET(0)].usd_amount IS NOT NULL
                AND p.list_price.tiered_rates[OFFSET(0)].usd_amount > 0
        )
        SELECT 
            customer_sku_id,
            service_name,
            total_usage_cost,
            total_usage_amount,
            usage_unit,
            current_unit_cost,
            product_taxonomy,
            current_regions
        FROM customer_sku_pricing
        ORDER BY total_usage_cost DESC  -- ✅ OPTIMIZATION: Order by cost for top N selection
        {limit_clause}  -- ✅ OPTIMIZATION: Limit to top N SKUs by cost
        """
    
    def _find_regional_alternatives(self, customer_skus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 2: Find equivalent services in other regions using product_taxonomy.
        """
        if not customer_skus:
            return []
        
        # Build product taxonomy list for matching
        taxonomies = [sku.get('product_taxonomy', []) for sku in customer_skus if sku.get('product_taxonomy')]
        
        if not taxonomies:
            logger.warning("No product taxonomies found for regional matching")
            return []
        
        query = self._build_regional_alternatives_query(customer_skus)
        
        try:
            result = self.toolset.execute_sql(query)
            logger.info(f"Found {len(result)} regional alternatives")
            return result
        except Exception as e:
            logger.error(f"Failed to find regional alternatives: {str(e)}")
            return []
    
    def _build_regional_alternatives_query(self, customer_skus: List[Dict[str, Any]]) -> str:
        """
        Build SQL query for Step 2: Regional alternatives discovery.
        """
        # Create customer SKUs CTE (already optimized by Step 1 filtering)
        customer_skus_cte = []
        for i, sku in enumerate(customer_skus):  # Use all SKUs from Step 1 (already limited to top N or specific SKUs)
            taxonomy_str = str(sku.get('product_taxonomy', [])).replace("'", "\\'")
            customer_skus_cte.append(f"""
            SELECT 
                '{sku.get('customer_sku_id', '')}' as customer_sku_id,
                '{sku.get('service_name', '').replace("'", "\\'")}' as service_name,
                {sku.get('current_unit_cost', 0)} as current_unit_cost,
                {sku.get('total_usage_cost', 0)} as total_usage_cost,
                {sku.get('total_usage_amount', 0)} as total_usage_amount,
                '{taxonomy_str}' as product_taxonomy_str
            """)
        
        customer_skus_union = " UNION ALL ".join(customer_skus_cte)
        
        return f"""
        -- Step 2: Find regional alternatives using product_taxonomy
        WITH customer_skus_data AS (
            {customer_skus_union}
        ),
        regional_alternatives AS (
            SELECT 
                csd.customer_sku_id,
                csd.service_name,
                csd.current_unit_cost,
                csd.total_usage_cost,
                csd.total_usage_amount,
                p.sku.id as alternative_sku_id,
                p.list_price.tiered_rates[OFFSET(0)].usd_amount as alternative_unit_cost,
                region as alternative_region
            FROM customer_skus_data csd
            JOIN `{self.pricing_table}` p ON TO_JSON_STRING(p.product_taxonomy) = csd.product_taxonomy_str
            CROSS JOIN UNNEST(p.geo_taxonomy.regions) as region
            WHERE p._PARTITIONTIME = (SELECT MAX(_PARTITIONTIME) FROM `{self.pricing_table}`)
                AND p.sku.id != csd.customer_sku_id
                AND p.list_price.tiered_rates[OFFSET(0)].usd_amount IS NOT NULL
                AND p.list_price.tiered_rates[OFFSET(0)].usd_amount > 0
        )
        SELECT 
            customer_sku_id,
            service_name,
            current_unit_cost,
            total_usage_cost,
            total_usage_amount,
            alternative_sku_id,
            alternative_unit_cost,
            alternative_region,
            ROUND((current_unit_cost - alternative_unit_cost) / current_unit_cost * 100, 2) as savings_percent
        FROM regional_alternatives
        WHERE alternative_unit_cost < current_unit_cost  -- Only cost-saving opportunities
        ORDER BY savings_percent DESC, total_usage_cost DESC
        """
    
    def _calculate_cost_comparisons(self, regional_alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: Calculate detailed cost comparisons and potential savings.
        """
        cost_comparisons = []
        
        for alt in regional_alternatives:
            try:
                current_cost = float(alt.get('current_unit_cost', 0))
                alternative_cost = float(alt.get('alternative_unit_cost', 0))
                total_usage_cost = float(alt.get('total_usage_cost', 0))
                total_usage_amount = float(alt.get('total_usage_amount', 0))
                
                if current_cost > 0 and alternative_cost > 0:
                    savings_per_unit = current_cost - alternative_cost
                    savings_percent = (savings_per_unit / current_cost) * 100
                    
                    # Calculate potential annual savings based on current usage
                    if total_usage_amount > 0:
                        potential_monthly_savings = savings_per_unit * total_usage_amount
                        potential_annual_savings = potential_monthly_savings * 12
                    else:
                        # Estimate based on cost if usage amount not available
                        cost_ratio = alternative_cost / current_cost
                        potential_monthly_savings = total_usage_cost * (1 - cost_ratio)
                        potential_annual_savings = potential_monthly_savings * 12
                    
                    comparison = {
                        'customer_sku_id': alt.get('customer_sku_id'),
                        'service_name': alt.get('service_name'),
                        'current_unit_cost': current_cost,
                        'alternative_unit_cost': alternative_cost,
                        'alternative_region': alt.get('alternative_region'),
                        'savings_per_unit': savings_per_unit,
                        'savings_percent': round(savings_percent, 2),
                        'total_usage_cost': total_usage_cost,
                        'total_usage_amount': total_usage_amount,
                        'potential_monthly_savings': round(potential_monthly_savings, 2),
                        'potential_annual_savings': round(potential_annual_savings, 2),
                        'recommendation_priority': self._calculate_priority(
                            savings_percent, potential_annual_savings, total_usage_cost
                        )
                    }
                    cost_comparisons.append(comparison)
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Error calculating comparison for SKU {alt.get('customer_sku_id')}: {str(e)}")
                continue
        
        # Sort by priority and potential savings
        cost_comparisons.sort(key=lambda x: (x['recommendation_priority'], x['potential_annual_savings']), reverse=True)
        
        return cost_comparisons
    
    def _calculate_priority(self, savings_percent: float, potential_annual_savings: float, 
                          total_usage_cost: float) -> int:
        """
        Calculate recommendation priority (1-5, higher is better).
        """
        priority = 1
        
        # High savings percentage
        if savings_percent >= 30:
            priority += 2
        elif savings_percent >= 15:
            priority += 1
        
        # High potential savings amount
        if potential_annual_savings >= 10000:
            priority += 2
        elif potential_annual_savings >= 1000:
            priority += 1
        
        # High usage cost (more impact)
        if total_usage_cost >= 5000:
            priority += 1
        
        return min(priority, 5)  # Cap at 5
    
    def _generate_cross_region_summary(self, cost_comparisons: List[Dict[str, Any]], 
                                     billing_account_id: str) -> Dict[str, Any]:
        """
        Step 4: Generate comprehensive summary and recommendations.
        """
        if not cost_comparisons:
            return self._generate_empty_analysis()
        
        # Calculate overall metrics
        total_potential_monthly_savings = sum(comp['potential_monthly_savings'] for comp in cost_comparisons)
        total_potential_annual_savings = sum(comp['potential_annual_savings'] for comp in cost_comparisons)
        
        # Get top opportunities
        top_opportunities = cost_comparisons[:10]  # Top 10 opportunities
        
        # Group by service
        service_summary = {}
        region_summary = {}
        
        for comp in cost_comparisons:
            service = comp['service_name']
            region = comp['alternative_region']
            
            # Service summary
            if service not in service_summary:
                service_summary[service] = {
                    'service_name': service,
                    'opportunities_count': 0,
                    'total_potential_savings': 0,
                    'best_savings_percent': 0,
                    'best_region': ''
                }
            
            service_summary[service]['opportunities_count'] += 1
            service_summary[service]['total_potential_savings'] += comp['potential_annual_savings']
            
            if comp['savings_percent'] > service_summary[service]['best_savings_percent']:
                service_summary[service]['best_savings_percent'] = comp['savings_percent']
                service_summary[service]['best_region'] = region
            
            # Region summary
            if region not in region_summary:
                region_summary[region] = {
                    'region': region,
                    'opportunities_count': 0,
                    'total_potential_savings': 0,
                    'average_savings_percent': 0
                }
            
            region_summary[region]['opportunities_count'] += 1
            region_summary[region]['total_potential_savings'] += comp['potential_annual_savings']
        
        # Calculate average savings percent by region
        for region_data in region_summary.values():
            region_comps = [c for c in cost_comparisons if c['alternative_region'] == region_data['region']]
            region_data['average_savings_percent'] = round(
                sum(c['savings_percent'] for c in region_comps) / len(region_comps), 2
            ) if region_comps else 0
        
        # Sort summaries
        service_summary_list = sorted(service_summary.values(), 
                                    key=lambda x: x['total_potential_savings'], reverse=True)
        region_summary_list = sorted(region_summary.values(), 
                                   key=lambda x: x['total_potential_savings'], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(cost_comparisons, service_summary_list, region_summary_list)
        
        return {
            'billing_account_id': billing_account_id,
            'analysis_date': datetime.now().isoformat(),
            'total_opportunities': len(cost_comparisons),
            'total_potential_monthly_savings': round(total_potential_monthly_savings, 2),
            'total_potential_annual_savings': round(total_potential_annual_savings, 2),
            'top_opportunities': top_opportunities,
            'service_summary': service_summary_list,
            'region_summary': region_summary_list,
            'recommendations': recommendations,
            'methodology': 'product_taxonomy_based_cross_region_comparison',
            'status': 'completed'
        }
    
    def _generate_recommendations(self, cost_comparisons: List[Dict[str, Any]], 
                                service_summary: List[Dict[str, Any]], 
                                region_summary: List[Dict[str, Any]]) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        """
        recommendations = []
        
        if not cost_comparisons:
            return ["No cross-region savings opportunities identified"]
        
        # Top service opportunity
        if service_summary:
            top_service = service_summary[0]
            recommendations.append(
                f"Priority: {top_service['service_name']} migration to {top_service['best_region']} "
                f"({top_service['best_savings_percent']:.1f}% savings, "
                f"${top_service['total_potential_savings']:,.0f} annual potential)"
            )
        
        # Top region opportunity
        if region_summary:
            top_region = region_summary[0]
            recommendations.append(
                f"Strategic: Consider {top_region['region']} as primary region "
                f"({top_region['average_savings_percent']:.1f}% average savings, "
                f"${top_region['total_potential_savings']:,.0f} annual potential)"
            )
        
        # High-priority individual opportunities
        high_priority = [c for c in cost_comparisons if c['recommendation_priority'] >= 4]
        if high_priority:
            recommendations.append(
                f"Immediate: {len(high_priority)} high-priority opportunities "
                f"(${sum(c['potential_annual_savings'] for c in high_priority):,.0f} annual potential)"
            )
        
        # Quick wins
        quick_wins = [c for c in cost_comparisons if c['savings_percent'] >= 25]
        if quick_wins:
            recommendations.append(
                f"Quick Wins: {len(quick_wins)} services with >25% savings potential "
                f"(${sum(c['potential_annual_savings'] for c in quick_wins):,.0f} annual potential)"
            )
        
        return recommendations
    
    def _generate_empty_analysis(self) -> Dict[str, Any]:
        """Generate empty analysis result when no data found."""
        return {
            'billing_account_id': '',
            'analysis_date': datetime.now().isoformat(),
            'total_opportunities': 0,
            'total_potential_monthly_savings': 0,
            'total_potential_annual_savings': 0,
            'top_opportunities': [],
            'service_summary': [],
            'region_summary': [],
            'recommendations': ["No cross-region pricing data available for analysis"],
            'methodology': 'product_taxonomy_based_cross_region_comparison',
            'status': 'no_data'
        }
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            'billing_account_id': '',
            'analysis_date': datetime.now().isoformat(),
            'total_opportunities': 0,
            'total_potential_monthly_savings': 0,
            'total_potential_annual_savings': 0,
            'top_opportunities': [],
            'service_summary': [],
            'region_summary': [],
            'recommendations': [f"Analysis failed: {error_message}"],
            'methodology': 'product_taxonomy_based_cross_region_comparison',
            'status': 'error',
            'error': error_message
        }
    
    def generate_mock_cross_region_data(self, billing_account_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive mock data for testing and validation.
        """
        logger.info(f"Generating mock cross-region analysis for {billing_account_id}")
        
        mock_opportunities = [
            {
                'customer_sku_id': 'CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-4-US-CENTRAL1',
                'service_name': 'Compute Engine',
                'current_unit_cost': 0.1900,
                'alternative_unit_cost': 0.1596,
                'alternative_region': 'us-south1',
                'savings_per_unit': 0.0304,
                'savings_percent': 16.0,
                'total_usage_cost': 8934.21,
                'total_usage_amount': 47022.16,
                'potential_monthly_savings': 1429.47,
                'potential_annual_savings': 17153.68,
                'recommendation_priority': 5
            },
            {
                'customer_sku_id': 'CP-BIGQUERY-ANALYSIS-US-CENTRAL1',
                'service_name': 'BigQuery',
                'current_unit_cost': 5.00,
                'alternative_unit_cost': 4.20,
                'alternative_region': 'us-west1',
                'savings_per_unit': 0.80,
                'savings_percent': 16.0,
                'total_usage_cost': 2456.78,
                'total_usage_amount': 491.36,
                'potential_monthly_savings': 392.93,
                'potential_annual_savings': 4715.14,
                'recommendation_priority': 4
            },
            {
                'customer_sku_id': 'CP-CLOUD-STORAGE-STANDARD-US-CENTRAL1',
                'service_name': 'Cloud Storage',
                'current_unit_cost': 0.020,
                'alternative_unit_cost': 0.018,
                'alternative_region': 'europe-west4',
                'savings_per_unit': 0.002,
                'savings_percent': 10.0,
                'total_usage_cost': 1847.63,
                'total_usage_amount': 92381.5,
                'potential_monthly_savings': 184.76,
                'potential_annual_savings': 2217.16,
                'recommendation_priority': 3
            }
        ]
        
        service_summary = [
            {
                'service_name': 'Compute Engine',
                'opportunities_count': 1,
                'total_potential_savings': 17153.68,
                'best_savings_percent': 16.0,
                'best_region': 'us-south1'
            },
            {
                'service_name': 'BigQuery',
                'opportunities_count': 1,
                'total_potential_savings': 4715.14,
                'best_savings_percent': 16.0,
                'best_region': 'us-west1'
            },
            {
                'service_name': 'Cloud Storage',
                'opportunities_count': 1,
                'total_potential_savings': 2217.16,
                'best_savings_percent': 10.0,
                'best_region': 'europe-west4'
            }
        ]
        
        region_summary = [
            {
                'region': 'us-south1',
                'opportunities_count': 1,
                'total_potential_savings': 17153.68,
                'average_savings_percent': 16.0
            },
            {
                'region': 'us-west1',
                'opportunities_count': 1,
                'total_potential_savings': 4715.14,
                'average_savings_percent': 16.0
            },
            {
                'region': 'europe-west4',
                'opportunities_count': 1,
                'total_potential_savings': 2217.16,
                'average_savings_percent': 10.0
            }
        ]
        
        recommendations = [
            "Priority: Compute Engine migration to us-south1 (16.0% savings, $17,154 annual potential)",
            "Strategic: Consider us-south1 as primary region (16.0% average savings, $17,154 annual potential)",
            "Immediate: 2 high-priority opportunities ($21,869 annual potential)",
            "Quick Wins: 2 services with >15% savings potential ($21,869 annual potential)"
        ]
        
        return {
            'billing_account_id': billing_account_id,
            'analysis_date': datetime.now().isoformat(),
            'total_opportunities': len(mock_opportunities),
            'total_potential_monthly_savings': 2007.16,
            'total_potential_annual_savings': 24085.98,
            'top_opportunities': mock_opportunities,
            'service_summary': service_summary,
            'region_summary': region_summary,
            'recommendations': recommendations,
            'methodology': 'product_taxonomy_based_cross_region_comparison',
            'status': 'completed_mock'
        }