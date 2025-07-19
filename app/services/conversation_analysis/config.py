import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class EnhancedMCTSConfig:
    """Enhanced configuration for MCTS algorithm with optimization parameters"""
    
    # Exploration parameters
    initial_exploration_constant: float = 1.41
    final_exploration_constant: float = 0.5
    adaptive_exploration: bool = True
    
    # Resource limits (adjusted for high-latency LLM operations)
    max_memory_mb: int = 512
    max_tree_nodes: int = 100  # Much lower due to 10-15s per node
    timeout_seconds: int = 1800  # 30 minutes for reasonable analysis
    
    # Pruning configuration (aggressive due to expensive LLM calls)
    enable_pruning: bool = True
    pruning_start_iteration_ratio: float = 0.2  # Start pruning earlier
    pruning_confidence_threshold: float = 0.15  # More aggressive pruning
    pruning_aggressiveness: float = 1.5  # Higher aggressiveness to avoid wasteful LLM calls
    
    # Early stopping (critical to avoid wasteful expensive LLM calls)
    enable_early_stopping: bool = True
    early_stop_score_gap: float = 0.08  # Lower threshold for earlier stopping
    early_stop_confidence: float = 0.6  # Lower confidence requirement
    convergence_window: int = 5  # Smaller window for faster detection
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_parallel_operations: int = 4
    batch_size: int = 8
    
    # State persistence (important for long-running analyses with expensive LLM calls)
    enable_checkpointing: bool = True  # Default enabled due to high cost of losing progress
    checkpoint_interval: int = 10  # More frequent due to expensive operations
    checkpoint_path: str = "./mcts_checkpoints"
    
    # Monitoring
    enable_detailed_monitoring: bool = True
    log_performance_metrics: bool = True
    
    # Legacy compatibility
    max_children: int = 3
    exploration_constant: float = 1.414
    pruning_interval: int = 5
    pruning_threshold_ratio: float = 0.7
    min_visits_for_pruning: int = 5

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters with clear error messages"""
        errors = []
        
        # Exploration parameters validation
        if self.initial_exploration_constant <= 0:
            errors.append("initial_exploration_constant must be positive")
        if self.final_exploration_constant <= 0:
            errors.append("final_exploration_constant must be positive")
        if self.initial_exploration_constant < self.final_exploration_constant:
            errors.append("initial_exploration_constant should be >= final_exploration_constant")
        
        # Resource limits validation
        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        if self.max_tree_nodes <= 0:
            errors.append("max_tree_nodes must be positive")
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        # Pruning configuration validation
        if not (0.0 <= self.pruning_start_iteration_ratio <= 1.0):
            errors.append("pruning_start_iteration_ratio must be between 0.0 and 1.0")
        if self.pruning_confidence_threshold <= 0:
            errors.append("pruning_confidence_threshold must be positive")
        if self.pruning_aggressiveness <= 0:
            errors.append("pruning_aggressiveness must be positive")
        
        # Early stopping validation
        if self.early_stop_score_gap <= 0:
            errors.append("early_stop_score_gap must be positive")
        if not (0.0 <= self.early_stop_confidence <= 1.0):
            errors.append("early_stop_confidence must be between 0.0 and 1.0")
        if self.convergence_window <= 0:
            errors.append("convergence_window must be positive")
        
        # Parallel processing validation
        if self.max_parallel_operations <= 0:
            errors.append("max_parallel_operations must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        # Checkpointing validation
        if self.checkpoint_interval <= 0:
            errors.append("checkpoint_interval must be positive")
        if not self.checkpoint_path:
            errors.append("checkpoint_path cannot be empty")
        
        # Legacy compatibility validation
        if self.max_children <= 0:
            errors.append("max_children must be positive")
        if self.exploration_constant <= 0:
            errors.append("exploration_constant must be positive")
        if self.pruning_interval <= 0:
            errors.append("pruning_interval must be positive")
        if not (0.0 <= self.pruning_threshold_ratio <= 1.0):
            errors.append("pruning_threshold_ratio must be between 0.0 and 1.0")
        if self.min_visits_for_pruning < 0:
            errors.append("min_visits_for_pruning must be non-negative")
        
        if errors:
            raise ValueError(f"MCTS Configuration validation failed: {'; '.join(errors)}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedMCTSConfig':
        """Create configuration from dictionary with validation"""
        # Filter only known fields to avoid TypeError
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'EnhancedMCTSConfig':
        """Load configuration from JSON file"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {file_path}: {e}")
    
    @classmethod
    def from_environment(cls, prefix: str = "MCTS_") -> 'EnhancedMCTSConfig':
        """Load configuration from environment variables with optional prefix"""
        config_dict = {}
        
        # Map of environment variable names to config field names
        env_mapping = {
            f"{prefix}INITIAL_EXPLORATION_CONSTANT": "initial_exploration_constant",
            f"{prefix}FINAL_EXPLORATION_CONSTANT": "final_exploration_constant",
            f"{prefix}ADAPTIVE_EXPLORATION": "adaptive_exploration",
            f"{prefix}MAX_MEMORY_MB": "max_memory_mb",
            f"{prefix}MAX_TREE_NODES": "max_tree_nodes",
            f"{prefix}TIMEOUT_SECONDS": "timeout_seconds",
            f"{prefix}ENABLE_PRUNING": "enable_pruning",
            f"{prefix}PRUNING_START_ITERATION_RATIO": "pruning_start_iteration_ratio",
            f"{prefix}PRUNING_CONFIDENCE_THRESHOLD": "pruning_confidence_threshold",
            f"{prefix}PRUNING_AGGRESSIVENESS": "pruning_aggressiveness",
            f"{prefix}ENABLE_EARLY_STOPPING": "enable_early_stopping",
            f"{prefix}EARLY_STOP_SCORE_GAP": "early_stop_score_gap",
            f"{prefix}EARLY_STOP_CONFIDENCE": "early_stop_confidence",
            f"{prefix}CONVERGENCE_WINDOW": "convergence_window",
            f"{prefix}ENABLE_PARALLEL_PROCESSING": "enable_parallel_processing",
            f"{prefix}MAX_PARALLEL_OPERATIONS": "max_parallel_operations",
            f"{prefix}BATCH_SIZE": "batch_size",
            f"{prefix}ENABLE_CHECKPOINTING": "enable_checkpointing",
            f"{prefix}CHECKPOINT_INTERVAL": "checkpoint_interval",
            f"{prefix}CHECKPOINT_PATH": "checkpoint_path",
            f"{prefix}ENABLE_DETAILED_MONITORING": "enable_detailed_monitoring",
            f"{prefix}LOG_PERFORMANCE_METRICS": "log_performance_metrics",
            f"{prefix}MAX_CHILDREN": "max_children",
            f"{prefix}EXPLORATION_CONSTANT": "exploration_constant",
            f"{prefix}PRUNING_INTERVAL": "pruning_interval",
            f"{prefix}PRUNING_THRESHOLD_RATIO": "pruning_threshold_ratio",
            f"{prefix}MIN_VISITS_FOR_PRUNING": "min_visits_for_pruning",
        }
        
        for env_var, field_name in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                field_type = cls.__dataclass_fields__[field_name].type
                try:
                    if field_type == bool:
                        config_dict[field_name] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_type == int:
                        config_dict[field_name] = int(env_value)
                    elif field_type == float:
                        config_dict[field_name] = float(env_value)
                    else:
                        config_dict[field_name] = env_value
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for {env_var}: {env_value}. Expected {field_type.__name__}: {e}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_with_overrides(cls, 
                           file_path: Optional[str] = None, 
                           env_prefix: str = "MCTS_",
                           base_config: Optional['EnhancedMCTSConfig'] = None) -> 'EnhancedMCTSConfig':
        """
        Load configuration with layered overrides:
        1. Start with base_config (or defaults)
        2. Override with file configuration if provided
        3. Override with environment variables
        """
        # Start with base configuration or defaults
        if base_config:
            config_dict = {
                field.name: getattr(base_config, field.name) 
                for field in cls.__dataclass_fields__.values()
            }
        else:
            config_dict = {}
        
        # Override with file configuration
        if file_path:
            try:
                file_config = cls.from_file(file_path)
                file_dict = {
                    field.name: getattr(file_config, field.name) 
                    for field in cls.__dataclass_fields__.values()
                }
                config_dict.update(file_dict)
            except Exception as e:
                # Log warning but continue with other sources
                print(f"Warning: Could not load configuration file {file_path}: {e}")
        
        # Override with environment variables
        try:
            env_config = cls.from_environment(env_prefix)
            env_dict = {
                field.name: getattr(env_config, field.name) 
                for field in cls.__dataclass_fields__.values()
                if hasattr(env_config, field.name) and 
                   getattr(env_config, field.name) != getattr(cls(), field.name)
            }
            config_dict.update(env_dict)
        except Exception as e:
            # Log warning but continue
            print(f"Warning: Error loading environment configuration: {e}")
        
        return cls.from_dict(config_dict) if config_dict else cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(self.to_json())


# Legacy configuration class for backward compatibility
@dataclass
class MCTSConfig:
    """Legacy configuration for MCTS algorithm - maintained for backward compatibility"""

    DEFAULT_MAX_CHILDREN = 3
    DEFAULT_EXPLORATION_CONSTANT = 1.414
    PRUNING_INTERVAL = 5
    PRUNING_THRESHOLD_RATIO = 0.7
    MIN_VISITS_FOR_PRUNING = 5


@dataclass
class ScoringConfig:
    """Configuration for scoring and evaluation"""

    SCORE_WEIGHT_QUALITY = 0.7
    SCORE_WEIGHT_VISITS = 0.3

    GENERAL_METRICS = [
        "clarity",
        "relevance",
        "engagement",
        "authenticity",
        "coherence",
        "respectfulness",
    ]


@dataclass
class ResponseConfig:
    """Configuration for response generation"""

    TOKEN_MULTIPLIER_INITIAL = 2
    TOKEN_MULTIPLIER_SIMULATION = 3
    TOKEN_MULTIPLIER_ANALYSIS = 2

    DEFAULT_RESPONSES = [
        "I understand you're going through a difficult time. Let's talk about what you're feeling.",
        "That sounds challenging. Can you tell me more about what happened?",
        "I'm here to listen and support you. What aspect of this situation is bothering you the most?",
    ]
