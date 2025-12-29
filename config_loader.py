"""
Configuration loader for YAML-based agent definitions.

This module provides utilities to load and manage YAML-based agent configurations
following Google ADK best practices for config-driven architecture.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LlmAgent
from google.adk.tools.bigquery import BigQueryToolset, BigQueryCredentialsConfig
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
import google.auth
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages YAML-based agent configurations."""
    
    def __init__(self, config_directory: str = None):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_directory: Path to the directory containing YAML configs
        """
        if config_directory is None:
            config_directory = os.path.join(os.path.dirname(__file__), 'configs')
        
        self.config_directory = Path(config_directory)
        self.loaded_agents: Dict[str, Agent] = {}
        self.loaded_configs: Dict[str, Dict[str, Any]] = {}
        
        # Setup shared BigQuery configuration
        self._setup_shared_config()
    
    def _setup_shared_config(self):
        """Setup shared BigQuery configuration for all agents."""
        tool_config = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)
        application_default_credentials, _ = google.auth.default()
        self.credentials_config = BigQueryCredentialsConfig(
            credentials=application_default_credentials
        )
        self.bigquery_tool_config = tool_config
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path: Path to the YAML config file (relative to config_directory)
            
        Returns:
            Loaded configuration dictionary
        """
        full_path = self.config_directory / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {full_path}")
        
        # Check cache first
        if str(full_path) in self.loaded_configs:
            return self.loaded_configs[str(full_path)]
        
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Cache the loaded config
            self.loaded_configs[str(full_path)] = config
            
            logger.info(f"Loaded configuration: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration {config_path}: {str(e)}")
            raise
    
    def create_bigquery_toolset(self, tool_config: Dict[str, Any]) -> BigQueryToolset:
        """
        Create a BigQuery toolset from configuration.
        
        Args:
            tool_config: Tool configuration dictionary
            
        Returns:
            Configured BigQueryToolset
        """
        tool_filter = tool_config.get('tool_filter', [])
        
        return BigQueryToolset(
            credentials_config=self.credentials_config,
            bigquery_tool_config=self.bigquery_tool_config,
            tool_filter=tool_filter
        )
    
    def create_agent_from_config(self, config: Dict[str, Any]) -> Agent:
        """
        Create an agent from a configuration dictionary.
        
        Args:
            config: Agent configuration dictionary
            
        Returns:
            Configured agent instance
        """
        agent_class = config.get('agent_class', 'LlmAgent')
        
        # Handle different agent types
        if agent_class == 'LlmAgent':
            return self._create_llm_agent(config)
        elif agent_class == 'SequentialAgent':
            return self._create_sequential_agent(config)
        elif agent_class == 'ParallelAgent':
            return self._create_parallel_agent(config)
        else:
            raise ValueError(f"Unsupported agent class: {agent_class}")
    
    def _create_llm_agent(self, config: Dict[str, Any]) -> LlmAgent:
        """Create an LLM agent from configuration."""
        
        # Extract basic configuration
        name = config.get('name', 'unnamed_agent')
        model = config.get('model', 'gemini-2.5-flash')
        description = config.get('description', '')
        output_key = config.get('output_key', 'agent_results')
        system_instruction = config.get('system_instruction', '')
        
        # Create tools
        tools = []
        if 'tools' in config:
            for tool_config in config['tools']:
                if tool_config.get('name') == 'BigQueryToolset':
                    tool = self.create_bigquery_toolset(tool_config.get('config', {}))
                    tools.append(tool)
        
        # Create the agent
        agent = LlmAgent(
            model=model,
            name=name,
            description=description,
            output_key=output_key,
            instruction=system_instruction,
            tools=tools
        )
        
        # Apply generation config if present
        if 'generation_config' in config:
            gen_config = config['generation_config']
            # Note: In a real implementation, you would apply these to the agent
            # This is simplified for demonstration purposes
            logger.debug(f"Generation config for {name}: {gen_config}")
        
        # Apply safety settings if present
        if 'safety_settings' in config:
            safety_config = config['safety_settings']
            # Note: In a real implementation, you would apply these to the agent
            logger.debug(f"Safety settings for {name}: {safety_config}")
        
        return agent
    
    def _create_sequential_agent(self, config: Dict[str, Any]) -> SequentialAgent:
        """Create a sequential agent from configuration."""
        
        name = config.get('name', 'unnamed_sequential_agent')
        description = config.get('description', '')
        
        # Load sub-agents
        sub_agents = []
        if 'sub_agents' in config:
            for sub_agent_config in config['sub_agents']:
                if 'config_path' in sub_agent_config:
                    sub_config = self.load_config(sub_agent_config['config_path'])
                    sub_agent = self.create_agent_from_config(sub_config)
                    sub_agents.append(sub_agent)
        
        agent = SequentialAgent(
            name=name,
            description=description,
            sub_agents=sub_agents
        )
        
        return agent
    
    def _create_parallel_agent(self, config: Dict[str, Any]) -> ParallelAgent:
        """Create a parallel agent from configuration."""
        
        name = config.get('name', 'unnamed_parallel_agent')
        description = config.get('description', '')
        
        # Load sub-agents
        sub_agents = []
        if 'sub_agents' in config:
            for sub_agent_config in config['sub_agents']:
                if 'config_path' in sub_agent_config:
                    sub_config = self.load_config(sub_agent_config['config_path'])
                    sub_agent = self.create_agent_from_config(sub_config)
                    sub_agents.append(sub_agent)
        
        agent = ParallelAgent(
            name=name,
            description=description,
            sub_agents=sub_agents
        )
        
        return agent
    
    def load_agent(self, config_path: str) -> Agent:
        """
        Load an agent from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML config file
            
        Returns:
            Configured agent instance
        """
        # Check cache first
        if config_path in self.loaded_agents:
            return self.loaded_agents[config_path]
        
        # Load configuration and create agent
        config = self.load_config(config_path)
        agent = self.create_agent_from_config(config)
        
        # Cache the created agent
        self.loaded_agents[config_path] = agent
        
        logger.info(f"Created agent from config: {config_path}")
        return agent
    
    def load_all_agents(self) -> Dict[str, Agent]:
        """
        Load all agents from the configuration directory.
        
        Returns:
            Dictionary of agent_name -> Agent instances
        """
        agents = {}
        
        # Find all YAML files in the config directory
        config_files = list(self.config_directory.glob('*.yaml'))
        config_files.extend(self.config_directory.glob('**/*.yaml'))
        
        for config_file in config_files:
            try:
                relative_path = config_file.relative_to(self.config_directory)
                agent = self.load_agent(str(relative_path))
                agents[agent.name] = agent
            except Exception as e:
                logger.warning(f"Failed to load agent from {config_file}: {str(e)}")
        
        return agents
    
    def get_available_configs(self) -> List[str]:
        """
        Get list of available configuration files.
        
        Returns:
            List of available config file paths
        """
        config_files = []
        
        # Find all YAML files
        yaml_files = list(self.config_directory.glob('*.yaml'))
        yaml_files.extend(self.config_directory.glob('**/*.yaml'))
        
        for config_file in yaml_files:
            relative_path = config_file.relative_to(self.config_directory)
            config_files.append(str(relative_path))
        
        return sorted(config_files)
    
    def validate_config(self, config_path: str) -> Dict[str, Any]:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to the config file to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'config_summary': {}
        }
        
        try:
            config = self.load_config(config_path)
            
            # Basic validation
            required_fields = ['agent_class', 'name']
            for field in required_fields:
                if field not in config:
                    validation_results['errors'].append(f"Missing required field: {field}")
                    validation_results['valid'] = False
            
            # Agent class validation
            supported_classes = ['LlmAgent', 'SequentialAgent', 'ParallelAgent']
            agent_class = config.get('agent_class')
            if agent_class not in supported_classes:
                validation_results['errors'].append(f"Unsupported agent_class: {agent_class}")
                validation_results['valid'] = False
            
            # LlmAgent specific validation
            if agent_class == 'LlmAgent':
                if 'system_instruction' not in config:
                    validation_results['warnings'].append("Missing system_instruction for LlmAgent")
            
            # Sub-agent validation for composite agents
            if agent_class in ['SequentialAgent', 'ParallelAgent']:
                if 'sub_agents' not in config:
                    validation_results['errors'].append(f"{agent_class} requires sub_agents configuration")
                    validation_results['valid'] = False
            
            # Create config summary
            validation_results['config_summary'] = {
                'agent_class': config.get('agent_class'),
                'name': config.get('name'),
                'description': config.get('description', ''),
                'has_tools': 'tools' in config,
                'has_sub_agents': 'sub_agents' in config,
                'tool_count': len(config.get('tools', [])),
                'sub_agent_count': len(config.get('sub_agents', []))
            }
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Configuration parsing error: {str(e)}")
        
        return validation_results


# Convenience functions

def load_agent_from_config(config_path: str, config_directory: str = None) -> Agent:
    """
    Load an agent from a YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file
        config_directory: Directory containing config files
        
    Returns:
        Configured agent instance
    """
    loader = ConfigLoader(config_directory)
    return loader.load_agent(config_path)


def get_all_agents(config_directory: str = None) -> Dict[str, Agent]:
    """
    Load all agents from configuration files.
    
    Args:
        config_directory: Directory containing config files
        
    Returns:
        Dictionary of agent_name -> Agent instances
    """
    loader = ConfigLoader(config_directory)
    return loader.load_all_agents()


def validate_all_configs(config_directory: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate all configuration files in a directory.
    
    Args:
        config_directory: Directory containing config files
        
    Returns:
        Dictionary of config_path -> validation_results
    """
    loader = ConfigLoader(config_directory)
    config_files = loader.get_available_configs()
    
    results = {}
    for config_file in config_files:
        results[config_file] = loader.validate_config(config_file)
    
    return results


# Example usage and testing

if __name__ == "__main__":
    # Example usage
    print("YAML Configuration Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = ConfigLoader()
    
    # Get available configs
    configs = loader.get_available_configs()
    print(f"Available configurations: {len(configs)}")
    for config in configs:
        print(f"  - {config}")
    
    # Load a specific agent
    try:
        agent = loader.load_agent('data_analysis_agent.yaml')
        print(f"\nLoaded agent: {agent.name}")
        print(f"Description: {agent.description}")
    except Exception as e:
        print(f"Error loading agent: {str(e)}")
    
    # Validate configurations
    print("\nValidating configurations...")
    for config in configs[:3]:  # Validate first 3 configs
        validation = loader.validate_config(config)
        status = "✓ Valid" if validation['valid'] else "✗ Invalid"
        print(f"  {config}: {status}")
        
        if validation['errors']:
            for error in validation['errors']:
                print(f"    Error: {error}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"    Warning: {warning}")