"""
Configuration loader for Enhanced MCTS Configuration System
Provides utilities for loading and managing MCTS configuration with environment overrides
"""

import os
from typing import Optional
from pathlib import Path

from .config import EnhancedMCTSConfig
from ...utils.logger import logger


class MCTSConfigLoader:
    """Utility class for loading MCTS configuration from various sources"""
    
    DEFAULT_CONFIG_PATHS = [
        "./mcts_config.json",
        "./config/mcts.json",
        os.path.expanduser("~/.mcts_config.json"),
    ]
    
    @classmethod
    def load_default_config(cls) -> EnhancedMCTSConfig:
        """Load configuration using default search paths and environment overrides"""
        return cls.load_config()
    
    @classmethod
    def load_config(cls, 
                   config_file: Optional[str] = None,
                   env_prefix: str = "MCTS_",
                   search_default_paths: bool = True) -> EnhancedMCTSConfig:
        """
        Load MCTS configuration with the following priority:
        1. Default configuration values
        2. Configuration file (if found)
        3. Environment variable overrides
        
        Args:
            config_file: Specific configuration file path
            env_prefix: Prefix for environment variables
            search_default_paths: Whether to search default config file locations
            
        Returns:
            EnhancedMCTSConfig: Loaded and validated configuration
        """
        config_file_path = None
        
        # Determine configuration file to use
        if config_file:
            if Path(config_file).exists():
                config_file_path = config_file
            else:
                logger.warning(f"Specified config file not found: {config_file}")
        elif search_default_paths:
            # Search for configuration file in default locations
            for path in cls.DEFAULT_CONFIG_PATHS:
                if Path(path).exists():
                    config_file_path = path
                    logger.info(f"Found MCTS configuration file: {path}")
                    break
        
        try:
            # Load configuration with overrides
            config = EnhancedMCTSConfig.load_with_overrides(
                file_path=config_file_path,
                env_prefix=env_prefix
            )
            
            logger.info("MCTS configuration loaded successfully")
            if config_file_path:
                logger.info(f"Configuration file: {config_file_path}")
            
            # Log key configuration values for debugging
            logger.debug(f"MCTS Config - Max nodes: {config.max_tree_nodes}, "
                        f"Timeout: {config.timeout_seconds}s, "
                        f"Pruning enabled: {config.enable_pruning}, "
                        f"Early stopping: {config.enable_early_stopping}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load MCTS configuration: {e}")
            logger.info("Using default MCTS configuration")
            return EnhancedMCTSConfig()
    
    @classmethod
    def create_sample_config_file(cls, file_path: str = "./mcts_config.json"):
        """Create a sample configuration file with all available options"""
        sample_config = EnhancedMCTSConfig()
        
        try:
            sample_config.save_to_file(file_path)
            logger.info(f"Sample MCTS configuration file created: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create sample configuration file: {e}")
            raise
    
    @classmethod
    def validate_environment_config(cls, env_prefix: str = "MCTS_") -> dict:
        """
        Validate environment variables for MCTS configuration
        
        Returns:
            dict: Dictionary of found environment variables and their values
        """
        found_vars = {}
        expected_vars = [
            f"{env_prefix}INITIAL_EXPLORATION_CONSTANT",
            f"{env_prefix}FINAL_EXPLORATION_CONSTANT", 
            f"{env_prefix}ADAPTIVE_EXPLORATION",
            f"{env_prefix}MAX_MEMORY_MB",
            f"{env_prefix}MAX_TREE_NODES",
            f"{env_prefix}TIMEOUT_SECONDS",
            f"{env_prefix}ENABLE_PRUNING",
            f"{env_prefix}PRUNING_START_ITERATION_RATIO",
            f"{env_prefix}PRUNING_CONFIDENCE_THRESHOLD",
            f"{env_prefix}PRUNING_AGGRESSIVENESS",
            f"{env_prefix}ENABLE_EARLY_STOPPING",
            f"{env_prefix}EARLY_STOP_SCORE_GAP",
            f"{env_prefix}EARLY_STOP_CONFIDENCE",
            f"{env_prefix}CONVERGENCE_WINDOW",
            f"{env_prefix}ENABLE_PARALLEL_PROCESSING",
            f"{env_prefix}MAX_PARALLEL_OPERATIONS",
            f"{env_prefix}BATCH_SIZE",
            f"{env_prefix}ENABLE_CHECKPOINTING",
            f"{env_prefix}CHECKPOINT_INTERVAL",
            f"{env_prefix}CHECKPOINT_PATH",
            f"{env_prefix}ENABLE_DETAILED_MONITORING",
            f"{env_prefix}LOG_PERFORMANCE_METRICS",
        ]
        
        for var in expected_vars:
            value = os.getenv(var)
            if value is not None:
                found_vars[var] = value
        
        return found_vars


# Global configuration instance
_global_mcts_config: Optional[EnhancedMCTSConfig] = None


def get_mcts_config() -> EnhancedMCTSConfig:
    """Get the global MCTS configuration instance (singleton pattern)"""
    global _global_mcts_config
    
    if _global_mcts_config is None:
        _global_mcts_config = MCTSConfigLoader.load_default_config()
    
    return _global_mcts_config


def reload_mcts_config(config_file: Optional[str] = None) -> EnhancedMCTSConfig:
    """Reload the global MCTS configuration"""
    global _global_mcts_config
    
    _global_mcts_config = MCTSConfigLoader.load_config(config_file=config_file)
    return _global_mcts_config


def set_mcts_config(config: EnhancedMCTSConfig):
    """Set the global MCTS configuration instance"""
    global _global_mcts_config
    _global_mcts_config = config