"""
Tests for ResourceManager class.

Tests resource limit detection, enforcement, and graceful degradation
for memory usage, node counts, and execution timeouts.
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from app.services.mcts.resource_manager import ResourceManager, ResourceUsage
from app.services.conversation_analysis.config import EnhancedMCTSConfig


@pytest.fixture
def default_config():
    """Default configuration for testing"""
    return EnhancedMCTSConfig(
        max_memory_mb=100,
        max_tree_nodes=50,
        timeout_seconds=30
    )


@pytest.fixture
def resource_manager(default_config):
    """ResourceManager instance for testing"""
    with patch('app.services.mcts.resource_manager.psutil.Process') as mock_process:
        mock_process.return_value.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB
        mock_process.return_value.cpu_percent.return_value = 25.0
        return ResourceManager(default_config)


class TestResourceManagerInitialization:
    """Test ResourceManager initialization"""
    
    def test_initialization_with_config(self, default_config):
        """Test ResourceManager initializes correctly with configuration"""
        with patch('app.services.mcts.resource_manager.psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 50 * 1024 * 1024
            
            manager = ResourceManager(default_config)
            
            assert manager.config == default_config
            assert manager.current_memory_mb == 0.0
            assert manager.current_node_count == 0
            assert manager.peak_memory_mb == 0.0
            assert manager.peak_node_count == 0
            assert manager.initial_memory_mb == 50.0  # 50MB converted from bytes
    
    def test_initialization_sets_start_time(self, default_config):
        """Test that initialization sets start time"""
        with patch('app.services.mcts.resource_manager.psutil.Process'):
            start_time = time.time()
            manager = ResourceManager(default_config)
            
            # Should be very close to current time
            assert abs(manager.start_time - start_time) < 1.0


class TestMemoryMonitoring:
    """Test memory usage monitoring"""
    
    def test_get_current_memory_mb(self, resource_manager):
        """Test memory usage calculation"""
        # Mock memory info to return 75MB
        resource_manager.process.memory_info.return_value.rss = 75 * 1024 * 1024
        
        memory_mb = resource_manager._get_current_memory_mb()
        assert memory_mb == 75.0
    
    def test_memory_monitoring_with_psutil_error(self, resource_manager):
        """Test memory monitoring handles psutil errors gracefully"""
        import psutil
        resource_manager.process.memory_info.side_effect = psutil.NoSuchProcess(123)
        
        memory_mb = resource_manager._get_current_memory_mb()
        assert memory_mb == 0.0
    
    def test_update_resource_usage_tracks_memory(self, resource_manager):
        """Test that update_resource_usage tracks memory correctly"""
        # Mock memory to return 80MB
        resource_manager.process.memory_info.return_value.rss = 80 * 1024 * 1024
        
        resource_manager.update_resource_usage(25)
        
        assert resource_manager.current_memory_mb == 80.0
        assert resource_manager.current_node_count == 25
        assert resource_manager.peak_memory_mb == 80.0
        assert resource_manager.peak_node_count == 25
    
    def test_peak_memory_tracking(self, resource_manager):
        """Test that peak memory is tracked correctly"""
        # First update with 60MB
        resource_manager.process.memory_info.return_value.rss = 60 * 1024 * 1024
        resource_manager.update_resource_usage(10)
        assert resource_manager.peak_memory_mb == 60.0
        
        # Second update with 90MB (higher)
        resource_manager.process.memory_info.return_value.rss = 90 * 1024 * 1024
        resource_manager.update_resource_usage(20)
        assert resource_manager.peak_memory_mb == 90.0
        
        # Third update with 70MB (lower - peak should remain 90MB)
        resource_manager.process.memory_info.return_value.rss = 70 * 1024 * 1024
        resource_manager.update_resource_usage(15)
        assert resource_manager.peak_memory_mb == 90.0


class TestResourceLimitChecking:
    """Test resource limit checking functionality"""
    
    def test_check_memory_limit_within_bounds(self, resource_manager):
        """Test memory limit check when within bounds"""
        # Set memory to 80MB (under 100MB limit)
        resource_manager.process.memory_info.return_value.rss = 80 * 1024 * 1024
        
        within_limit, message = resource_manager.check_memory_limit()
        
        assert within_limit is True
        assert message == ""
    
    def test_check_memory_limit_exceeded(self, resource_manager):
        """Test memory limit check when exceeded"""
        # Set memory to 120MB (over 100MB limit)
        resource_manager.process.memory_info.return_value.rss = 120 * 1024 * 1024
        
        within_limit, message = resource_manager.check_memory_limit()
        
        assert within_limit is False
        assert "Memory limit exceeded: 120.0MB > 100MB" in message
    
    def test_check_node_limit_within_bounds(self, resource_manager):
        """Test node limit check when within bounds"""
        within_limit, message = resource_manager.check_node_limit(40)
        
        assert within_limit is True
        assert message == ""
        assert resource_manager.current_node_count == 40
    
    def test_check_node_limit_exceeded(self, resource_manager):
        """Test node limit check when exceeded"""
        within_limit, message = resource_manager.check_node_limit(60)
        
        assert within_limit is False
        assert "Node limit exceeded: 60 > 50" in message
        assert "each node = 10-15s LLM call" in message
    
    def test_check_timeout_within_bounds(self, resource_manager):
        """Test timeout check when within bounds"""
        # Simulate 10 seconds elapsed (under 30s limit)
        resource_manager.start_time = time.time() - 10
        
        within_limit, message = resource_manager.check_timeout()
        
        assert within_limit is True
        assert message == ""
    
    def test_check_timeout_exceeded(self, resource_manager):
        """Test timeout check when exceeded"""
        # Simulate 40 seconds elapsed (over 30s limit)
        resource_manager.start_time = time.time() - 40
        
        within_limit, message = resource_manager.check_timeout()
        
        assert within_limit is False
        assert "Timeout exceeded:" in message
        assert "> 30s" in message
    
    def test_check_resource_limits_all_within_bounds(self, resource_manager):
        """Test comprehensive resource check when all limits are respected"""
        # Set memory to 80MB, nodes to 40, time to 20s
        resource_manager.process.memory_info.return_value.rss = 80 * 1024 * 1024
        resource_manager.current_node_count = 40
        resource_manager.start_time = time.time() - 20
        
        within_limits, message = resource_manager.check_resource_limits()
        
        assert within_limits is True
        assert message == ""
    
    def test_check_resource_limits_memory_exceeded(self, resource_manager):
        """Test comprehensive resource check when memory limit exceeded"""
        # Set memory to 120MB (over limit)
        resource_manager.process.memory_info.return_value.rss = 120 * 1024 * 1024
        resource_manager.current_node_count = 40
        resource_manager.start_time = time.time() - 20
        
        within_limits, message = resource_manager.check_resource_limits()
        
        assert within_limits is False
        assert "Memory limit exceeded" in message
    
    def test_check_resource_limits_node_exceeded(self, resource_manager):
        """Test comprehensive resource check when node limit exceeded"""
        # Set nodes to 60 (over limit)
        resource_manager.process.memory_info.return_value.rss = 80 * 1024 * 1024
        resource_manager.current_node_count = 60
        resource_manager.start_time = time.time() - 20
        
        within_limits, message = resource_manager.check_resource_limits()
        
        assert within_limits is False
        assert "Node limit exceeded" in message
    
    def test_check_resource_limits_timeout_exceeded(self, resource_manager):
        """Test comprehensive resource check when timeout exceeded"""
        # Set time to 40s (over limit)
        resource_manager.process.memory_info.return_value.rss = 80 * 1024 * 1024
        resource_manager.current_node_count = 40
        resource_manager.start_time = time.time() - 40
        
        within_limits, message = resource_manager.check_resource_limits()
        
        assert within_limits is False
        assert "Timeout exceeded" in message


class TestResourceUsageRatios:
    """Test resource usage ratio calculations"""
    
    def test_get_memory_usage_ratio(self, resource_manager):
        """Test memory usage ratio calculation"""
        resource_manager.current_memory_mb = 75.0
        # Config has max_memory_mb = 100
        
        ratio = resource_manager.get_memory_usage_ratio()
        assert ratio == 0.75
    
    def test_get_memory_usage_ratio_exceeded(self, resource_manager):
        """Test memory usage ratio when limit exceeded"""
        resource_manager.current_memory_mb = 120.0
        
        ratio = resource_manager.get_memory_usage_ratio()
        assert ratio == 1.2
    
    def test_get_node_usage_ratio(self, resource_manager):
        """Test node usage ratio calculation"""
        resource_manager.current_node_count = 30
        # Config has max_tree_nodes = 50
        
        ratio = resource_manager.get_node_usage_ratio()
        assert ratio == 0.6
    
    def test_get_time_usage_ratio(self, resource_manager):
        """Test time usage ratio calculation"""
        # Simulate 15 seconds elapsed out of 30s limit
        resource_manager.start_time = time.time() - 15
        
        ratio = resource_manager.get_time_usage_ratio()
        assert abs(ratio - 0.5) < 0.1  # Allow small timing variations
    
    def test_get_remaining_time(self, resource_manager):
        """Test remaining time calculation"""
        # Simulate 20 seconds elapsed out of 30s limit
        resource_manager.start_time = time.time() - 20
        
        remaining = resource_manager.get_remaining_time()
        assert abs(remaining - 10.0) < 1.0  # Allow small timing variations
    
    def test_get_remaining_time_exceeded(self, resource_manager):
        """Test remaining time when timeout exceeded"""
        # Simulate 40 seconds elapsed out of 30s limit
        resource_manager.start_time = time.time() - 40
        
        remaining = resource_manager.get_remaining_time()
        assert remaining == 0.0


class TestGracefulShutdown:
    """Test graceful shutdown detection"""
    
    def test_should_trigger_graceful_shutdown_memory_approaching(self, resource_manager):
        """Test graceful shutdown when memory approaching limit"""
        # Set memory to 95MB (95% of 100MB limit)
        resource_manager.current_memory_mb = 95.0
        
        should_shutdown, reason = resource_manager.should_trigger_graceful_shutdown()
        
        assert should_shutdown is True
        assert "Memory usage approaching limit" in reason
        assert "95.0%" in reason
    
    def test_should_trigger_graceful_shutdown_nodes_approaching(self, resource_manager):
        """Test graceful shutdown when nodes approaching limit"""
        # Set nodes to 47 (94% of 50 limit)
        resource_manager.current_node_count = 47
        
        should_shutdown, reason = resource_manager.should_trigger_graceful_shutdown()
        
        assert should_shutdown is True
        assert "Node count approaching limit" in reason
        assert "94.0%" in reason
    
    def test_should_trigger_graceful_shutdown_time_approaching(self, resource_manager):
        """Test graceful shutdown when time approaching limit"""
        # Simulate 28 seconds elapsed (93.3% of 30s limit)
        resource_manager.start_time = time.time() - 28
        
        should_shutdown, reason = resource_manager.should_trigger_graceful_shutdown()
        
        assert should_shutdown is True
        assert "Execution time approaching limit" in reason
    
    def test_should_not_trigger_graceful_shutdown(self, resource_manager):
        """Test no graceful shutdown when resources are comfortable"""
        # Set all resources to 50% usage
        resource_manager.current_memory_mb = 50.0
        resource_manager.current_node_count = 25
        resource_manager.start_time = time.time() - 15
        
        should_shutdown, reason = resource_manager.should_trigger_graceful_shutdown()
        
        assert should_shutdown is False
        assert reason == ""


class TestResourceSummary:
    """Test resource usage summary generation"""
    
    def test_get_resource_summary(self, resource_manager):
        """Test comprehensive resource summary"""
        # Set up some usage
        resource_manager.current_memory_mb = 80.0
        resource_manager.peak_memory_mb = 90.0
        resource_manager.current_node_count = 35
        resource_manager.peak_node_count = 40
        resource_manager.start_time = time.time() - 20
        resource_manager.initial_memory_mb = 50.0
        resource_manager.process.cpu_percent.return_value = 45.0
        
        summary = resource_manager.get_resource_summary()
        
        assert summary["current_memory_mb"] == 80.0
        assert summary["peak_memory_mb"] == 90.0
        assert summary["memory_limit_mb"] == 100
        assert summary["memory_usage_ratio"] == 0.8
        assert summary["current_node_count"] == 35
        assert summary["peak_node_count"] == 40
        assert summary["node_limit"] == 50
        assert summary["node_usage_ratio"] == 0.7
        assert summary["timeout_seconds"] == 30
        assert abs(summary["elapsed_seconds"] - 20.0) < 1.0
        assert abs(summary["remaining_time_seconds"] - 10.0) < 1.0
        assert summary["initial_memory_mb"] == 50.0
        assert summary["memory_growth_mb"] == 30.0


class TestResourceUsageDataClass:
    """Test ResourceUsage dataclass"""
    
    def test_resource_usage_creation(self):
        """Test ResourceUsage object creation"""
        usage = ResourceUsage(
            memory_mb=75.5,
            node_count=42,
            elapsed_seconds=123.4,
            cpu_percent=67.8
        )
        
        assert usage.memory_mb == 75.5
        assert usage.node_count == 42
        assert usage.elapsed_seconds == 123.4
        assert usage.cpu_percent == 67.8
    
    def test_resource_usage_string_representation(self):
        """Test ResourceUsage string representation"""
        usage = ResourceUsage(
            memory_mb=75.5,
            node_count=42,
            elapsed_seconds=123.4,
            cpu_percent=67.8
        )
        
        str_repr = str(usage)
        assert "Memory: 75.5MB" in str_repr
        assert "Nodes: 42" in str_repr
        assert "Time: 123.4s" in str_repr
        assert "CPU: 67.8%" in str_repr


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_zero_limits_handling(self):
        """Test handling of zero resource limits"""
        # Create a mock config object instead of using the actual EnhancedMCTSConfig
        # since it has validation that prevents zero values
        mock_config = Mock()
        mock_config.max_memory_mb = 0
        mock_config.max_tree_nodes = 0
        mock_config.timeout_seconds = 0
        
        with patch('app.services.mcts.resource_manager.psutil.Process'):
            manager = ResourceManager(mock_config)
            
            # Should handle zero limits gracefully
            assert manager.get_memory_usage_ratio() == 0.0
            assert manager.get_node_usage_ratio() == 0.0
            assert manager.get_time_usage_ratio() == 0.0
    
    def test_psutil_access_denied_handling(self, resource_manager):
        """Test handling of psutil access denied errors"""
        import psutil
        resource_manager.process.memory_info.side_effect = psutil.AccessDenied()
        resource_manager.process.cpu_percent.side_effect = psutil.AccessDenied()
        
        # Should handle errors gracefully
        memory_mb = resource_manager._get_current_memory_mb()
        cpu_percent = resource_manager._get_cpu_percent()
        
        assert memory_mb == 0.0
        assert cpu_percent == 0.0
    
    def test_timer_reset(self, resource_manager):
        """Test timer reset functionality"""
        # Simulate some elapsed time
        original_start = resource_manager.start_time
        time.sleep(0.1)
        
        resource_manager.reset_timer()
        
        # Start time should be updated
        assert resource_manager.start_time > original_start


class TestLogging:
    """Test logging functionality"""
    
    @patch('app.services.mcts.resource_manager.logger')
    def test_log_resource_violation(self, mock_logger, resource_manager):
        """Test resource violation logging"""
        resource_manager.current_memory_mb = 120.0
        resource_manager.current_node_count = 60
        resource_manager.peak_memory_mb = 125.0
        resource_manager.peak_node_count = 65
        
        resource_manager.log_resource_violation("Test violation")
        
        # Should log warning with details
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "Resource limit violation: Test violation" in call_args
        assert "Peak memory: 125.0MB" in call_args
        assert "Peak nodes: 65" in call_args
    
    @patch('app.services.mcts.resource_manager.logger')
    def test_log_graceful_shutdown(self, mock_logger, resource_manager):
        """Test graceful shutdown logging"""
        resource_manager.current_memory_mb = 95.0
        resource_manager.current_node_count = 47
        
        resource_manager.log_graceful_shutdown("Memory approaching limit")
        
        # Should log info with details
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Graceful shutdown initiated: Memory approaching limit" in call_args


if __name__ == "__main__":
    pytest.main([__file__])