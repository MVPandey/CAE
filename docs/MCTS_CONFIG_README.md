# Enhanced MCTS Configuration System

This document describes the enhanced MCTS (Monte Carlo Tree Search) configuration system that provides comprehensive optimization parameters, environment variable support, and flexible configuration loading.

## Overview

The Enhanced MCTS Configuration System provides:

- **Comprehensive optimization parameters** for fine-tuning MCTS performance
- **Environment variable overrides** for deployment flexibility
- **Configuration file support** with JSON format
- **Layered configuration loading** (defaults → file → environment)
- **Validation with clear error messages** for all parameters
- **Backward compatibility** with existing MCTS configuration

## Configuration Parameters

### Exploration Parameters
- `initial_exploration_constant` (float, default: 1.41): Starting exploration constant
- `final_exploration_constant` (float, default: 0.5): Final exploration constant
- `adaptive_exploration` (bool, default: True): Enable adaptive exploration

### Resource Limits
- `max_memory_mb` (int, default: 512): Maximum memory usage in MB
- `max_tree_nodes` (int, default: 100): Maximum number of tree nodes
- `timeout_seconds` (int, default: 1800): Analysis timeout in seconds

### Pruning Configuration
- `enable_pruning` (bool, default: True): Enable tree pruning
- `pruning_start_iteration_ratio` (float, default: 0.2): When to start pruning
- `pruning_confidence_threshold` (float, default: 0.15): Confidence threshold for pruning
- `pruning_aggressiveness` (float, default: 1.5): Pruning aggressiveness level

### Early Stopping
- `enable_early_stopping` (bool, default: True): Enable early stopping
- `early_stop_score_gap` (float, default: 0.08): Score gap threshold
- `early_stop_confidence` (float, default: 0.6): Confidence threshold
- `convergence_window` (int, default: 5): Convergence detection window

### Parallel Processing
- `enable_parallel_processing` (bool, default: True): Enable parallel processing
- `max_parallel_operations` (int, default: 4): Maximum parallel operations
- `batch_size` (int, default: 8): Batch size for operations

### State Persistence
- `enable_checkpointing` (bool, default: True): Enable checkpointing
- `checkpoint_interval` (int, default: 10): Checkpoint frequency
- `checkpoint_path` (str, default: "./mcts_checkpoints"): Checkpoint directory

### Monitoring
- `enable_detailed_monitoring` (bool, default: True): Enable detailed monitoring
- `log_performance_metrics` (bool, default: True): Log performance metrics

## Usage

### Basic Usage

```python
from app.services.conversation_analysis.config_loader import get_mcts_config

# Get the global configuration instance
config = get_mcts_config()
print(f"Max tree nodes: {config.max_tree_nodes}")
```

### Loading from Environment Variables

Set environment variables with the `MCTS_` prefix:

```bash
export MCTS_MAX_TREE_NODES=200
export MCTS_TIMEOUT_SECONDS=3600
export MCTS_ENABLE_PRUNING=false
```

### Loading from Configuration File

Create a JSON configuration file:

```json
{
  "max_tree_nodes": 150,
  "timeout_seconds": 2400,
  "enable_early_stopping": false,
  "checkpoint_path": "/custom/path"
}
```

Load it using:

```python
from app.services.conversation_analysis.config_loader import MCTSConfigLoader

config = MCTSConfigLoader.load_config(config_file="my_config.json")
```

### Layered Configuration

The system supports layered configuration with the following priority:

1. **Default values** (lowest priority)
2. **Configuration file** (medium priority)
3. **Environment variables** (highest priority)

```python
config = MCTSConfigLoader.load_config(
    config_file="base_config.json",
    env_prefix="MCTS_"
)
```

## Environment Variables

All configuration parameters can be overridden using environment variables with the `MCTS_` prefix:

| Parameter | Environment Variable | Type | Example |
|-----------|---------------------|------|---------|
| max_tree_nodes | MCTS_MAX_TREE_NODES | int | `200` |
| timeout_seconds | MCTS_TIMEOUT_SECONDS | int | `3600` |
| enable_pruning | MCTS_ENABLE_PRUNING | bool | `false` |
| initial_exploration_constant | MCTS_INITIAL_EXPLORATION_CONSTANT | float | `2.0` |
| checkpoint_path | MCTS_CHECKPOINT_PATH | str | `/tmp/checkpoints` |

Boolean values accept: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`

## Configuration Files

### Default Search Paths

The system automatically searches for configuration files in:

1. `./mcts_config.json`
2. `./config/mcts.json`
3. `~/.mcts_config.json`

### Sample Configuration File

Use the provided `sample_mcts_config.json` as a template:

```bash
cp sample_mcts_config.json mcts_config.json
# Edit mcts_config.json with your settings
```

### Creating Sample Configuration

```python
from app.services.conversation_analysis.config_loader import MCTSConfigLoader

MCTSConfigLoader.create_sample_config_file("my_config.json")
```

## Validation

The system validates all configuration parameters and provides clear error messages:

```python
try:
    config = EnhancedMCTSConfig(max_tree_nodes=-1)
except ValueError as e:
    print(e)  # "MCTS Configuration validation failed: max_tree_nodes must be positive"
```

## Integration

The enhanced configuration is automatically integrated into the conversation analysis service:

```python
from app.services.conversation_analysis_service import ConversationAnalysisService

service = ConversationAnalysisService()
# Service automatically loads enhanced MCTS configuration
print(f"Using {service.mcts_config.max_tree_nodes} max tree nodes")
```

## Backward Compatibility

The system maintains backward compatibility with the existing `MCTSConfig` class. Existing code will continue to work without changes.

## Performance Considerations

The default configuration values are optimized for LLM-based MCTS operations:

- **Lower node limits** due to expensive LLM calls (10-15s per node)
- **Aggressive pruning** to avoid wasteful LLM operations
- **Early stopping** enabled by default to prevent unnecessary computation
- **Checkpointing** enabled to preserve progress during long analyses

## Troubleshooting

### Configuration Not Loading

1. Check file permissions and paths
2. Verify JSON syntax in configuration files
3. Check environment variable names and values
4. Review application logs for configuration loading messages

### Validation Errors

1. Ensure all numeric values are positive where required
2. Check that ratio values are between 0.0 and 1.0
3. Verify boolean values use accepted formats
4. Ensure file paths are valid and accessible

### Performance Issues

1. Adjust `max_tree_nodes` based on available resources
2. Tune `timeout_seconds` for your use case
3. Enable/disable pruning based on analysis requirements
4. Adjust parallel processing parameters based on system capabilities

## Examples

See the test files and sample configuration for complete usage examples.