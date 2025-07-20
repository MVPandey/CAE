"""
Resource management system for MCTS optimization.

This module provides the ResourceManager class that monitors memory usage,
node counts, and execution timeouts to prevent resource exhaustion during
MCTS tree search operations.
"""

import time
from dataclasses import dataclass

import psutil

from app.services.conversation_analysis.config import EnhancedMCTSConfig
from app.utils.logger import logger


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""

    memory_mb: float
    node_count: int
    elapsed_seconds: float
    cpu_percent: float

    def __str__(self) -> str:
        return (
            f"Memory: {self.memory_mb:.1f}MB, "
            f"Nodes: {self.node_count}, "
            f"Time: {self.elapsed_seconds:.1f}s, "
            f"CPU: {self.cpu_percent:.1f}%"
        )


class ResourceManager:
    """Manages resource limits and monitoring for MCTS algorithm execution.

    Monitors memory usage, node counts, and execution time to prevent
    resource exhaustion and ensure graceful degradation when limits are exceeded.
    """

    def __init__(self, config: EnhancedMCTSConfig) -> None:
        """Initialize ResourceManager with configuration.

        Args:
            config: Enhanced MCTS configuration with resource limits
        """
        self.config = config
        self.start_time = time.time()
        self.current_memory_mb = 0.0
        self.current_node_count = 0
        self.peak_memory_mb = 0.0
        self.peak_node_count = 0

        # Get initial process for memory monitoring
        self.process = psutil.Process()
        self.initial_memory_mb = self._get_current_memory_mb()

        logger.info(
            f"ResourceManager initialized with limits: "
            f"memory={config.max_memory_mb}MB, "
            f"nodes={config.max_tree_nodes}, "
            f"timeout={config.timeout_seconds}s"
        )
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB using psutil."""
        try:
            # Get memory info for current process
            memory_info = self.process.memory_info()
            # Convert bytes to MB (RSS - Resident Set Size)
            return memory_info.rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not get CPU usage: {e}")
            return 0.0

    def update_resource_usage(self, node_count: int) -> None:
        """Update current resource usage tracking.

        Args:
            node_count: Current number of nodes in the MCTS tree
        """
        self.current_memory_mb = self._get_current_memory_mb()
        self.current_node_count = node_count

        # Track peak usage
        self.peak_memory_mb = max(self.peak_memory_mb, self.current_memory_mb)
        self.peak_node_count = max(self.peak_node_count, self.current_node_count)

    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage snapshot.

        Returns:
            ResourceUsage object with current metrics
        """
        elapsed = time.time() - self.start_time
        cpu_percent = self._get_cpu_percent()

        return ResourceUsage(
            memory_mb=self.current_memory_mb,
            node_count=self.current_node_count,
            elapsed_seconds=elapsed,
            cpu_percent=cpu_percent,
        )
    
    def check_resource_limits(self) -> tuple[bool, str]:
        """Check if current resource usage is within configured limits.

        Returns:
            Tuple of (within_limits, violation_message)
            - within_limits: True if all limits are respected
            - violation_message: Description of limit violation (empty if within limits)
        """
        # Update current usage before checking
        self.update_resource_usage(self.current_node_count)

        # Memory limit check
        if self.current_memory_mb > self.config.max_memory_mb:
            message = (
                f"Memory limit exceeded: {self.current_memory_mb:.1f}MB > "
                f"{self.config.max_memory_mb}MB"
            )
            logger.warning(message)
            return False, message

        # Node count limit check (critical due to 10-15s LLM latency per node)
        if self.current_node_count > self.config.max_tree_nodes:
            message = (
                f"Node limit exceeded: {self.current_node_count} > "
                f"{self.config.max_tree_nodes} (each node = 10-15s LLM call)"
            )
            logger.warning(message)
            return False, message

        # Timeout check
        elapsed = time.time() - self.start_time
        if elapsed > self.config.timeout_seconds:
            message = (
                f"Timeout exceeded: {elapsed:.1f}s > "
                f"{self.config.timeout_seconds}s"
            )
            logger.warning(message)
            return False, message

        return True, ""

    def check_memory_limit(self) -> tuple[bool, str]:
        """Check only memory limit (for frequent checks during tree expansion).

        Returns:
            Tuple of (within_limit, violation_message)
        """
        current_memory = self._get_current_memory_mb()
        self.current_memory_mb = current_memory
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory)

        if current_memory > self.config.max_memory_mb:
            message = (
                f"Memory limit exceeded: {current_memory:.1f}MB > "
                f"{self.config.max_memory_mb}MB"
            )
            return False, message

        return True, ""
    
    def check_node_limit(self, node_count: int) -> tuple[bool, str]:
        """Check only node count limit (for frequent checks during expansion).

        Args:
            node_count: Current number of nodes to check

        Returns:
            Tuple of (within_limit, violation_message)
        """
        self.current_node_count = node_count
        self.peak_node_count = max(self.peak_node_count, node_count)

        if node_count > self.config.max_tree_nodes:
            message = (
                f"Node limit exceeded: {node_count} > "
                f"{self.config.max_tree_nodes} (each node = 10-15s LLM call)"
            )
            return False, message

        return True, ""

    def check_timeout(self) -> tuple[bool, str]:
        """Check only timeout limit (for frequent checks during execution).

        Returns:
            Tuple of (within_limit, violation_message)
        """
        elapsed = time.time() - self.start_time

        if elapsed > self.config.timeout_seconds:
            message = (
                f"Timeout exceeded: {elapsed:.1f}s > "
                f"{self.config.timeout_seconds}s"
            )
            return False, message

        return True, ""

    def get_remaining_time(self) -> float:
        """Get remaining execution time in seconds.

        Returns:
            Remaining time in seconds (0 if timeout exceeded)
        """
        elapsed = time.time() - self.start_time
        remaining = self.config.timeout_seconds - elapsed
        return max(0.0, remaining)

    def get_memory_usage_ratio(self) -> float:
        """Get current memory usage as ratio of limit (0.0 to 1.0+).

        Returns:
            Memory usage ratio (values > 1.0 indicate limit exceeded)
        """
        if self.config.max_memory_mb <= 0:
            return 0.0
        return self.current_memory_mb / self.config.max_memory_mb
    
    def get_node_usage_ratio(self) -> float:
        """Get current node count as ratio of limit (0.0 to 1.0+).

        Returns:
            Node usage ratio (values > 1.0 indicate limit exceeded)
        """
        if self.config.max_tree_nodes <= 0:
            return 0.0
        return self.current_node_count / self.config.max_tree_nodes

    def get_time_usage_ratio(self) -> float:
        """Get elapsed time as ratio of timeout limit (0.0 to 1.0+).

        Returns:
            Time usage ratio (values > 1.0 indicate timeout exceeded)
        """
        if self.config.timeout_seconds <= 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return elapsed / self.config.timeout_seconds

    def log_resource_violation(self, violation_message: str) -> None:
        """Log resource limit violation with current usage details.

        Args:
            violation_message: Description of the violation
        """
        usage = self.get_current_usage()
        logger.warning(
            f"Resource limit violation: {violation_message}. "
            f"Current usage: {usage}. "
            f"Peak memory: {self.peak_memory_mb:.1f}MB, "
            f"Peak nodes: {self.peak_node_count}"
        )

    def log_graceful_shutdown(self, reason: str) -> None:
        """Log graceful shutdown due to resource constraints.

        Args:
            reason: Reason for shutdown
        """
        usage = self.get_current_usage()
        logger.info(
            f"Graceful shutdown initiated: {reason}. "
            f"Final usage: {usage}. "
            f"Peak memory: {self.peak_memory_mb:.1f}MB, "
            f"Peak nodes: {self.peak_node_count}"
        )
    
    def get_resource_summary(self) -> dict:
        """Get comprehensive resource usage summary.

        Returns:
            Dictionary with resource usage statistics
        """
        usage = self.get_current_usage()

        return {
            "current_memory_mb": self.current_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_limit_mb": self.config.max_memory_mb,
            "memory_usage_ratio": self.get_memory_usage_ratio(),
            "current_node_count": self.current_node_count,
            "peak_node_count": self.peak_node_count,
            "node_limit": self.config.max_tree_nodes,
            "node_usage_ratio": self.get_node_usage_ratio(),
            "elapsed_seconds": usage.elapsed_seconds,
            "timeout_seconds": self.config.timeout_seconds,
            "time_usage_ratio": self.get_time_usage_ratio(),
            "remaining_time_seconds": self.get_remaining_time(),
            "cpu_percent": usage.cpu_percent,
            "initial_memory_mb": self.initial_memory_mb,
            "memory_growth_mb": self.current_memory_mb - self.initial_memory_mb,
        }

    def should_trigger_graceful_shutdown(self) -> tuple[bool, str]:
        """Determine if graceful shutdown should be triggered based on resource usage.

        This method provides early warning before hard limits are hit,
        allowing for graceful cleanup and result preservation.

        Returns:
            Tuple of (should_shutdown, reason)
        """
        # Check if we're approaching limits (90% threshold)
        memory_ratio = self.get_memory_usage_ratio()
        node_ratio = self.get_node_usage_ratio()
        time_ratio = self.get_time_usage_ratio()

        if memory_ratio >= 0.9:
            return True, f"Memory usage approaching limit: {memory_ratio:.1%}"

        if node_ratio >= 0.9:
            return True, f"Node count approaching limit: {node_ratio:.1%}"

        if time_ratio >= 0.9:
            return True, f"Execution time approaching limit: {time_ratio:.1%}"

        return False, ""

    def reset_timer(self) -> None:
        """Reset the execution timer (useful for checkpoint restoration)."""
        self.start_time = time.time()
        logger.info("Resource manager timer reset")