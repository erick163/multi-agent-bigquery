"""
Configuration utilities for dynamic date replacement in agent configs.
"""

import yaml
import os
from typing import Dict, Any
from .shared_services import get_current_date_sync


def load_agent_config_with_dynamic_date(config_path: str) -> Dict[str, Any]:
    """
    Load an agent configuration file and replace {CURRENT_DATE} placeholders 
    with the current date from billing data.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with configuration data and dynamic date substitution
    """
    try:
        # Get current date dynamically
        current_date = get_current_date_sync()
        
        # Read and parse YAML file
        with open(config_path, 'r', encoding='utf-8') as file:
            config_content = file.read()
        
        # Replace {CURRENT_DATE} placeholder
        config_content = config_content.replace('{CURRENT_DATE}', current_date)
        
        # Parse the updated YAML
        config_dict = yaml.safe_load(config_content)
        
        return config_dict
        
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        # Fallback: load without substitution
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)


def get_config_directory() -> str:
    """Get the path to the configs directory."""
    return os.path.join(os.path.dirname(__file__), 'configs')


def load_all_agent_configs_with_dynamic_date() -> Dict[str, Dict[str, Any]]:
    """
    Load all agent configuration files with dynamic date replacement.
    
    Returns:
        Dictionary mapping agent names to their configuration data
    """
    config_dir = get_config_directory()
    configs = {}
    
    # Main agent configs
    main_configs = [
        'data_analysis_agent.yaml',
        'forecasting_agent.yaml', 
        'business_logic_agent.yaml'
    ]
    
    for config_file in main_configs:
        config_path = os.path.join(config_dir, config_file)
        if os.path.exists(config_path):
            agent_name = config_file.replace('.yaml', '').replace('_agent', '')
            configs[agent_name] = load_agent_config_with_dynamic_date(config_path)
    
    # Subagent configs
    subagent_dir = os.path.join(config_dir, 'subagents')
    if os.path.exists(subagent_dir):
        for config_file in os.listdir(subagent_dir):
            if config_file.endswith('.yaml'):
                config_path = os.path.join(subagent_dir, config_file)
                agent_name = f"subagent_{config_file.replace('.yaml', '')}"
                configs[agent_name] = load_agent_config_with_dynamic_date(config_path)
    
    return configs