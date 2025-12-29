"""
Custom tools for the multi-agent BigQuery system.

Since the ADK BaseTool API is complex and experimental, this module provides
a simple wrapper function approach that avoids ADK tool registration issues.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Simple function wrapper for cross-region analysis
def create_cross_region_function():
    """
    Create a simple function that can be called by agents.
    This avoids the complexity of ADK tool registration.
    """
    try:
        from .shared_services import analyze_cross_region_opportunities
        return analyze_cross_region_opportunities
    except ImportError as e:
        logger.error(f"Could not import analyze_cross_region_opportunities: {e}")
        
        # Return a fallback function
        def fallback_function(billing_account_id: str, **kwargs) -> Dict[str, Any]:
            return {
                'status': 'error',
                'message': 'Cross-region analysis not available - import error',
                'billing_account_id': billing_account_id,
                'recommendations': []
            }
        return fallback_function


# Create the function instance
cross_region_analysis_function = create_cross_region_function()


# For backwards compatibility, create a simple mock tool object
class MockCrossRegionTool:
    """
    Simple mock tool that provides the function without ADK complexity.
    """
    def __init__(self):
        self.name = "analyze_cross_region_opportunities"
        self.description = "Cross-region pricing analysis for cost optimization"
    
    def run(self, **kwargs):
        return cross_region_analysis_function(**kwargs)
    
    def __call__(self, **kwargs):
        return self.run(**kwargs)


# Create the mock tool instance
cross_region_analysis_tool = MockCrossRegionTool()


# Simple function wrapper for crypto-mining hypothesis testing
def create_crypto_mining_function():
    """
    Create a simple function that can be called by agents for crypto-mining detection.
    This avoids the complexity of ADK tool registration.
    """
    try:
        from .shared_services import test_crypto_mining_hypothesis
        return test_crypto_mining_hypothesis
    except ImportError as e:
        logger.error(f"Could not import test_crypto_mining_hypothesis: {e}")

        # Return a fallback function
        def fallback_function(usage_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            return {
                'conclusion': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'evidence': ['Crypto-mining analysis not available - import error'],
                'summary': 'Security analysis unavailable due to system error.'
            }
        return fallback_function


# Create the function instance
crypto_mining_analysis_function = create_crypto_mining_function()


# Mock tool for crypto-mining hypothesis testing
class MockCryptoMiningTool:
    """
    Simple mock tool that provides crypto-mining hypothesis testing without ADK complexity.
    """
    def __init__(self):
        self.name = "test_crypto_mining_hypothesis"
        self.description = "Scientific hypothesis testing for crypto-mining detection"

    def run(self, **kwargs):
        return crypto_mining_analysis_function(**kwargs)

    def __call__(self, **kwargs):
        return self.run(**kwargs)


# Create the mock tool instance
crypto_mining_analysis_tool = MockCryptoMiningTool()