"""
Unit tests for Adaptive Exploration Strategy

Tests the adaptive exploration strategy implementation across different scenarios
including edge cases, configuration variations, and strategy types.
"""

import pytest
import math
from unittest.mock import patch, MagicMock

from app.services.mcts.adaptive_exploration import (
    AdaptiveExplorationStrategy,
    LinearDecayStrategy,
    ExponentialDecayStrategy,
    SquareRootDecayStrategy,
    create_exploration_strategy
)
from app.services.conversation_analysis.config import EnhancedMCTSConfig


class TestAdaptiveExplorationStrategy:
    """Test cases for AdaptiveExplorationStrategy"""
    
    def test_initialization_with_default_config(self):
        """Test initialization with default configuration"""
        config = EnhancedMCTSConfig()
        strategy = AdaptiveExplorationStrategy(config)
        
        assert strategy.config == config
        assert strategy._last_logged_constant is None
        assert strategy._log_threshold == 0.05
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.1,
            adaptive_exploration=True
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        assert strategy.config.initial_exploration_constant == 2.0
        assert strategy.config.final_exploration_constant == 0.1
        assert strategy.config.adaptive_exploration is True
    
    def test_adaptive_exploration_enabled_linear_interpolation(self):
        """Test adaptive exploration with linear interpolation"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        # Test at different iteration points
        test_cases = [
            (0, 100, 2.0),      # Start: should be initial constant
            (25, 100, 1.625),   # 25% progress: 2.0 * 0.75 + 0.5 * 0.25
            (50, 100, 1.25),    # 50% progress: 2.0 * 0.5 + 0.5 * 0.5
            (75, 100, 0.875),   # 75% progress: 2.0 * 0.25 + 0.5 * 0.75
            (99, 100, 0.515),   # Near end: close to final constant
        ]
        
        for iteration, max_iterations, expected in test_cases:
            result = strategy.get_exploration_constant(iteration, max_iterations)
            assert abs(result - expected) < 0.001, f"Failed for iteration {iteration}/{max_iterations}"
    
    def test_adaptive_exploration_disabled_fixed_constant(self):
        """Test behavior when adaptive exploration is disabled"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=1.5,
            final_exploration_constant=0.3,
            adaptive_exploration=False
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        # Should always return initial constant regardless of iteration
        test_cases = [(0, 100), (25, 100), (50, 100), (75, 100), (99, 100)]
        
        for iteration, max_iterations in test_cases:
            result = strategy.get_exploration_constant(iteration, max_iterations)
            assert result == 1.5, f"Failed for iteration {iteration}/{max_iterations}"
    
    def test_edge_case_single_iteration(self):
        """Test edge case with single iteration"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        # With max_iterations=1, iteration=0 should give initial constant
        result = strategy.get_exploration_constant(0, 1)
        assert result == 2.0
    
    def test_edge_case_iteration_equals_max_iterations(self):
        """Test edge case when iteration equals max_iterations"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        # Should handle gracefully by treating as last valid iteration
        result = strategy.get_exploration_constant(100, 100)
        # Should be close to final constant (iteration treated as 99/100)
        expected = 2.0 * 0.01 + 0.5 * 0.99  # 0.515
        assert abs(result - expected) < 0.001
    
    def test_edge_case_iteration_exceeds_max_iterations(self):
        """Test edge case when iteration exceeds max_iterations"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        # Should handle gracefully by treating as last valid iteration
        result = strategy.get_exploration_constant(150, 100)
        expected = 2.0 * 0.01 + 0.5 * 0.99  # 0.515
        assert abs(result - expected) < 0.001
    
    def test_invalid_iteration_negative(self):
        """Test error handling for negative iteration"""
        config = EnhancedMCTSConfig()
        strategy = AdaptiveExplorationStrategy(config)
        
        with pytest.raises(ValueError, match="Iteration must be non-negative"):
            strategy.get_exploration_constant(-1, 100)
    
    def test_invalid_max_iterations_zero(self):
        """Test error handling for zero max_iterations"""
        config = EnhancedMCTSConfig()
        strategy = AdaptiveExplorationStrategy(config)
        
        with pytest.raises(ValueError, match="Max iterations must be positive"):
            strategy.get_exploration_constant(0, 0)
    
    def test_invalid_max_iterations_negative(self):
        """Test error handling for negative max_iterations"""
        config = EnhancedMCTSConfig()
        strategy = AdaptiveExplorationStrategy(config)
        
        with pytest.raises(ValueError, match="Max iterations must be positive"):
            strategy.get_exploration_constant(0, -10)
    
    @patch('app.services.mcts.adaptive_exploration.logger')
    def test_logging_behavior_first_iteration(self, mock_logger):
        """Test that first iteration is always logged"""
        config = EnhancedMCTSConfig()
        strategy = AdaptiveExplorationStrategy(config)
        
        strategy.get_exploration_constant(0, 100)
        
        # Should log the first iteration
        mock_logger.debug.assert_called()
        call_args = mock_logger.debug.call_args[0][0]
        assert "iteration 0" in call_args
        assert "strategy: adaptive" in call_args
    
    @patch('app.services.mcts.adaptive_exploration.logger')
    def test_logging_behavior_significant_change(self, mock_logger):
        """Test logging when exploration constant changes significantly"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        # First call should log
        strategy.get_exploration_constant(0, 100)
        mock_logger.debug.reset_mock()
        
        # Small change should not log
        strategy.get_exploration_constant(1, 100)
        assert not mock_logger.debug.called
        
        # Large change should log
        strategy.get_exploration_constant(50, 100)
        mock_logger.debug.assert_called()
    
    def test_same_initial_and_final_constants(self):
        """Test behavior when initial and final constants are the same"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=1.0,
            final_exploration_constant=1.0,
            adaptive_exploration=True
        )
        strategy = AdaptiveExplorationStrategy(config)
        
        # Should always return the same constant
        for iteration in [0, 25, 50, 75, 99]:
            result = strategy.get_exploration_constant(iteration, 100)
            assert result == 1.0


class TestLinearDecayStrategy:
    """Test cases for LinearDecayStrategy (alias for AdaptiveExplorationStrategy)"""
    
    def test_linear_decay_is_adaptive_strategy(self):
        """Test that LinearDecayStrategy is an alias for AdaptiveExplorationStrategy"""
        config = EnhancedMCTSConfig()
        linear_strategy = LinearDecayStrategy(config)
        adaptive_strategy = AdaptiveExplorationStrategy(config)
        
        # Should behave identically
        for iteration in [0, 25, 50, 75, 99]:
            linear_result = linear_strategy.get_exploration_constant(iteration, 100)
            adaptive_result = adaptive_strategy.get_exploration_constant(iteration, 100)
            assert linear_result == adaptive_result


class TestExponentialDecayStrategy:
    """Test cases for ExponentialDecayStrategy"""
    
    def test_initialization_with_default_decay_rate(self):
        """Test initialization with default decay rate"""
        config = EnhancedMCTSConfig()
        strategy = ExponentialDecayStrategy(config)
        
        assert strategy.config == config
        assert strategy.decay_rate == 2.0
        assert strategy._last_logged_constant is None
    
    def test_initialization_with_custom_decay_rate(self):
        """Test initialization with custom decay rate"""
        config = EnhancedMCTSConfig()
        strategy = ExponentialDecayStrategy(config, decay_rate=3.0)
        
        assert strategy.decay_rate == 3.0
    
    def test_exponential_decay_calculation(self):
        """Test exponential decay calculation"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        strategy = ExponentialDecayStrategy(config, decay_rate=2.0)
        
        # Test specific points
        result_start = strategy.get_exploration_constant(0, 100)
        assert abs(result_start - 2.0) < 0.001  # Should be close to initial
        
        result_mid = strategy.get_exploration_constant(50, 100)
        # At 50% progress: exp(-2.0 * 0.5) = exp(-1) ≈ 0.368
        expected_mid = 0.5 + (2.0 - 0.5) * math.exp(-1)
        assert abs(result_mid - expected_mid) < 0.001
        
        result_end = strategy.get_exploration_constant(99, 100)
        # Should be close to final constant
        assert result_end < 1.0  # Should be much lower than initial
    
    def test_exponential_decay_disabled_adaptive(self):
        """Test exponential decay when adaptive exploration is disabled"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=1.5,
            adaptive_exploration=False
        )
        strategy = ExponentialDecayStrategy(config)
        
        # Should always return initial constant
        for iteration in [0, 25, 50, 75, 99]:
            result = strategy.get_exploration_constant(iteration, 100)
            assert result == 1.5
    
    def test_exponential_decay_faster_than_linear(self):
        """Test that exponential decay reduces exploration faster than linear"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        
        linear_strategy = AdaptiveExplorationStrategy(config)
        exp_strategy = ExponentialDecayStrategy(config, decay_rate=2.0)
        
        # At mid-point, exponential should be lower than linear
        linear_mid = linear_strategy.get_exploration_constant(50, 100)
        exp_mid = exp_strategy.get_exploration_constant(50, 100)
        
        assert exp_mid < linear_mid


class TestSquareRootDecayStrategy:
    """Test cases for SquareRootDecayStrategy"""
    
    def test_sqrt_decay_calculation(self):
        """Test square root decay calculation"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        strategy = SquareRootDecayStrategy(config)
        
        # Test specific points
        result_start = strategy.get_exploration_constant(0, 100)
        assert abs(result_start - 2.0) < 0.001
        
        result_mid = strategy.get_exploration_constant(25, 100)  # 25% progress
        # sqrt(0.25) = 0.5, so: 2.0 * 0.5 + 0.5 * 0.5 = 1.25
        expected_mid = 2.0 * 0.5 + 0.5 * 0.5
        assert abs(result_mid - expected_mid) < 0.001
    
    def test_sqrt_decay_different_from_linear(self):
        """Test that sqrt decay behaves differently from linear decay"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=2.0,
            final_exploration_constant=0.5,
            adaptive_exploration=True
        )
        
        linear_strategy = AdaptiveExplorationStrategy(config)
        sqrt_strategy = SquareRootDecayStrategy(config)
        
        # Test at different progress points
        test_points = [10, 25, 50, 75, 90]
        
        for iteration in test_points:
            linear_result = linear_strategy.get_exploration_constant(iteration, 100)
            sqrt_result = sqrt_strategy.get_exploration_constant(iteration, 100)
            
            # sqrt should decay faster than linear (except at start and end)
            if iteration > 0 and iteration < 100:
                assert sqrt_result != linear_result, f"Results should differ at iteration {iteration}"
                
        # At 25% progress, sqrt should be lower than linear (faster decay)
        linear_25 = linear_strategy.get_exploration_constant(25, 100)
        sqrt_25 = sqrt_strategy.get_exploration_constant(25, 100)
        assert sqrt_25 < linear_25


class TestCreateExplorationStrategy:
    """Test cases for the strategy factory function"""
    
    def test_create_linear_strategy(self):
        """Test creating linear strategy"""
        config = EnhancedMCTSConfig()
        strategy = create_exploration_strategy(config, "linear")
        
        assert isinstance(strategy, AdaptiveExplorationStrategy)
    
    def test_create_exponential_strategy(self):
        """Test creating exponential strategy"""
        config = EnhancedMCTSConfig()
        strategy = create_exploration_strategy(config, "exponential")
        
        assert isinstance(strategy, ExponentialDecayStrategy)
    
    def test_create_sqrt_strategy(self):
        """Test creating square root strategy"""
        config = EnhancedMCTSConfig()
        strategy = create_exploration_strategy(config, "sqrt")
        
        assert isinstance(strategy, SquareRootDecayStrategy)
    
    def test_create_unknown_strategy(self):
        """Test error handling for unknown strategy type"""
        config = EnhancedMCTSConfig()
        
        with pytest.raises(ValueError, match="Unknown strategy type: unknown"):
            create_exploration_strategy(config, "unknown")
    
    def test_create_default_strategy(self):
        """Test creating strategy with default type"""
        config = EnhancedMCTSConfig()
        strategy = create_exploration_strategy(config)  # Default should be linear
        
        assert isinstance(strategy, AdaptiveExplorationStrategy)


class TestIntegrationScenarios:
    """Integration test scenarios for different use cases"""
    
    def test_high_latency_llm_scenario(self):
        """Test adaptive exploration for high-latency LLM scenario"""
        # Configuration optimized for expensive LLM calls
        config = EnhancedMCTSConfig(
            initial_exploration_constant=1.8,  # Higher initial exploration
            final_exploration_constant=0.3,    # Lower final exploration
            adaptive_exploration=True,
            max_tree_nodes=50  # Fewer nodes due to cost
        )
        
        strategy = AdaptiveExplorationStrategy(config)
        
        # Should start high and end low for cost efficiency
        start_constant = strategy.get_exploration_constant(0, 20)
        mid_constant = strategy.get_exploration_constant(10, 20)
        end_constant = strategy.get_exploration_constant(19, 20)
        
        assert start_constant == 1.8
        assert end_constant < 0.4  # Should be close to final
        assert start_constant > mid_constant > end_constant  # Decreasing trend
    
    def test_fast_iteration_scenario(self):
        """Test adaptive exploration for fast iteration scenario"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=1.2,
            final_exploration_constant=0.8,
            adaptive_exploration=True
        )
        
        strategy = AdaptiveExplorationStrategy(config)
        
        # With smaller range, changes should be more gradual
        constants = []
        for i in range(0, 100, 10):
            constants.append(strategy.get_exploration_constant(i, 100))
        
        # Should be monotonically decreasing
        for i in range(1, len(constants)):
            assert constants[i] <= constants[i-1]
        
        # Range should be smaller - check that we go from 1.2 to approximately 0.8
        range_diff = constants[0] - constants[-1]
        assert abs(range_diff - 0.36) < 0.05  # Allow for iteration step differences (90/100 vs 100/100)
    
    def test_disabled_adaptive_exploration_consistency(self):
        """Test consistency when adaptive exploration is disabled"""
        config = EnhancedMCTSConfig(
            initial_exploration_constant=1.414,
            final_exploration_constant=0.5,
            adaptive_exploration=False
        )
        
        strategies = [
            AdaptiveExplorationStrategy(config),
            ExponentialDecayStrategy(config),
            SquareRootDecayStrategy(config)
        ]
        
        # All strategies should return the same constant when adaptive is disabled
        for iteration in [0, 25, 50, 75, 99]:
            constants = [
                strategy.get_exploration_constant(iteration, 100)
                for strategy in strategies
            ]
            
            # All should be equal to initial constant
            for constant in constants:
                assert constant == 1.414