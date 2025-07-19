"""
Adaptive Exploration Strategy for MCTS Algorithm

This module implements adaptive exploration strategies that adjust the exploration
constant dynamically during MCTS execution to balance exploration and exploitation
more effectively over time.
"""

import math
from typing import Protocol
from ..conversation_analysis.config import EnhancedMCTSConfig
from ...utils.logger import logger


class ExplorationStrategy(Protocol):
    """Protocol for exploration strategies"""
    
    def get_exploration_constant(self, iteration: int, max_iterations: int) -> float:
        """Calculate exploration constant for given iteration"""
        ...


class AdaptiveExplorationStrategy:
    """
    Adaptive exploration strategy that adjusts exploration constant over time.
    
    The strategy starts with high exploration (initial_exploration_constant) and
    gradually decreases to low exploration (final_exploration_constant) as iterations
    progress, allowing for broad exploration early and focused exploitation later.
    """
    
    def __init__(self, config: EnhancedMCTSConfig):
        """
        Initialize adaptive exploration strategy.
        
        Args:
            config: Enhanced MCTS configuration containing exploration parameters
        """
        self.config = config
        self._last_logged_constant = None
        self._log_threshold = 0.05  # Log when constant changes by more than 5%
        
        logger.info(
            f"Initialized AdaptiveExplorationStrategy: "
            f"adaptive={config.adaptive_exploration}, "
            f"initial_c={config.initial_exploration_constant}, "
            f"final_c={config.final_exploration_constant}"
        )
    
    def get_exploration_constant(self, iteration: int, max_iterations: int) -> float:
        """
        Calculate exploration constant for the given iteration.
        
        Uses linear interpolation between initial and final exploration constants
        based on iteration progress when adaptive exploration is enabled.
        
        Args:
            iteration: Current iteration number (0-based)
            max_iterations: Total number of iterations
            
        Returns:
            Exploration constant for UCB1 calculation
            
        Raises:
            ValueError: If iteration or max_iterations are invalid
        """
        if iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {iteration}")
        if max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive, got {max_iterations}")
        if iteration >= max_iterations:
            # Handle edge case where iteration equals or exceeds max_iterations
            iteration = max_iterations - 1
        
        if not self.config.adaptive_exploration:
            # Use fixed exploration constant when adaptive exploration is disabled
            constant = self.config.initial_exploration_constant
            self._log_exploration_change(constant, iteration, "fixed")
            return constant
        
        # Calculate progress ratio (0.0 at start, 1.0 at end)
        progress = iteration / max_iterations
        
        # Linear interpolation between initial and final constants
        constant = (
            self.config.initial_exploration_constant * (1 - progress) +
            self.config.final_exploration_constant * progress
        )
        
        self._log_exploration_change(constant, iteration, "adaptive")
        return constant
    
    def _log_exploration_change(self, constant: float, iteration: int, strategy_type: str):
        """
        Log exploration constant changes for monitoring.
        
        Only logs when the constant changes significantly to avoid log spam.
        
        Args:
            constant: Current exploration constant
            iteration: Current iteration number
            strategy_type: Type of strategy ("adaptive" or "fixed")
        """
        should_log = (
            self._last_logged_constant is None or
            abs(constant - self._last_logged_constant) >= self._log_threshold or
            iteration == 0  # Always log first iteration
        )
        
        if should_log:
            logger.debug(
                f"Exploration constant updated: {constant:.3f} "
                f"(iteration {iteration}, strategy: {strategy_type})"
            )
            self._last_logged_constant = constant


class LinearDecayStrategy(AdaptiveExplorationStrategy):
    """
    Linear decay exploration strategy (alias for AdaptiveExplorationStrategy).
    
    This is the default adaptive strategy that uses linear interpolation.
    """
    pass


class ExponentialDecayStrategy:
    """
    Exponential decay exploration strategy.
    
    Uses exponential decay instead of linear interpolation for more aggressive
    exploration reduction in later iterations.
    """
    
    def __init__(self, config: EnhancedMCTSConfig, decay_rate: float = 2.0):
        """
        Initialize exponential decay strategy.
        
        Args:
            config: Enhanced MCTS configuration
            decay_rate: Rate of exponential decay (higher = faster decay)
        """
        self.config = config
        self.decay_rate = decay_rate
        self._last_logged_constant = None
        self._log_threshold = 0.05
        
        logger.info(
            f"Initialized ExponentialDecayStrategy: "
            f"adaptive={config.adaptive_exploration}, "
            f"decay_rate={decay_rate}"
        )
    
    def get_exploration_constant(self, iteration: int, max_iterations: int) -> float:
        """
        Calculate exploration constant using exponential decay.
        
        Args:
            iteration: Current iteration number (0-based)
            max_iterations: Total number of iterations
            
        Returns:
            Exploration constant for UCB1 calculation
        """
        if iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {iteration}")
        if max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive, got {max_iterations}")
        if iteration >= max_iterations:
            iteration = max_iterations - 1
        
        if not self.config.adaptive_exploration:
            constant = self.config.initial_exploration_constant
            self._log_exploration_change(constant, iteration, "fixed")
            return constant
        
        # Exponential decay: c(t) = c_final + (c_initial - c_final) * exp(-decay_rate * progress)
        progress = iteration / max_iterations
        decay_factor = math.exp(-self.decay_rate * progress)
        
        constant = (
            self.config.final_exploration_constant +
            (self.config.initial_exploration_constant - self.config.final_exploration_constant) * decay_factor
        )
        
        self._log_exploration_change(constant, iteration, "exponential")
        return constant
    
    def _log_exploration_change(self, constant: float, iteration: int, strategy_type: str):
        """Log exploration constant changes for monitoring."""
        should_log = (
            self._last_logged_constant is None or
            abs(constant - self._last_logged_constant) >= self._log_threshold or
            iteration == 0
        )
        
        if should_log:
            logger.debug(
                f"Exploration constant updated: {constant:.3f} "
                f"(iteration {iteration}, strategy: {strategy_type})"
            )
            self._last_logged_constant = constant


class SquareRootDecayStrategy:
    """
    Square root decay exploration strategy.
    
    Uses square root decay for moderate exploration reduction that's between
    linear and exponential decay rates.
    """
    
    def __init__(self, config: EnhancedMCTSConfig):
        """
        Initialize square root decay strategy.
        
        Args:
            config: Enhanced MCTS configuration
        """
        self.config = config
        self._last_logged_constant = None
        self._log_threshold = 0.05
        
        logger.info(
            f"Initialized SquareRootDecayStrategy: "
            f"adaptive={config.adaptive_exploration}"
        )
    
    def get_exploration_constant(self, iteration: int, max_iterations: int) -> float:
        """
        Calculate exploration constant using square root decay.
        
        Args:
            iteration: Current iteration number (0-based)
            max_iterations: Total number of iterations
            
        Returns:
            Exploration constant for UCB1 calculation
        """
        if iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {iteration}")
        if max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive, got {max_iterations}")
        if iteration >= max_iterations:
            iteration = max_iterations - 1
        
        if not self.config.adaptive_exploration:
            constant = self.config.initial_exploration_constant
            self._log_exploration_change(constant, iteration, "fixed")
            return constant
        
        # Square root decay: more gradual than exponential, faster than linear
        progress = iteration / max_iterations
        sqrt_progress = math.sqrt(progress)
        
        constant = (
            self.config.initial_exploration_constant * (1 - sqrt_progress) +
            self.config.final_exploration_constant * sqrt_progress
        )
        
        self._log_exploration_change(constant, iteration, "sqrt")
        return constant
    
    def _log_exploration_change(self, constant: float, iteration: int, strategy_type: str):
        """Log exploration constant changes for monitoring."""
        should_log = (
            self._last_logged_constant is None or
            abs(constant - self._last_logged_constant) >= self._log_threshold or
            iteration == 0
        )
        
        if should_log:
            logger.debug(
                f"Exploration constant updated: {constant:.3f} "
                f"(iteration {iteration}, strategy: {strategy_type})"
            )
            self._last_logged_constant = constant


def create_exploration_strategy(config: EnhancedMCTSConfig, strategy_type: str = "linear") -> ExplorationStrategy:
    """
    Factory function to create exploration strategies.
    
    Args:
        config: Enhanced MCTS configuration
        strategy_type: Type of strategy ("linear", "exponential", "sqrt")
        
    Returns:
        Exploration strategy instance
        
    Raises:
        ValueError: If strategy_type is not recognized
    """
    strategy_map = {
        "linear": AdaptiveExplorationStrategy,
        "exponential": ExponentialDecayStrategy,
        "sqrt": SquareRootDecayStrategy,
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available types: {list(strategy_map.keys())}"
        )
    
    strategy_class = strategy_map[strategy_type]
    
    # ExponentialDecayStrategy has additional parameters
    if strategy_type == "exponential":
        return strategy_class(config)
    else:
        return strategy_class(config)