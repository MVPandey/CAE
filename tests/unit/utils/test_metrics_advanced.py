"""Advanced tests for metrics system edge cases."""

from unittest.mock import MagicMock, patch

import pytest

from app.utils.metrics import MetricsCollector


class TestMetricsCollectorEdgeCases:
    """Test edge cases for metrics collector."""

    @pytest.mark.asyncio
    @patch("app.utils.metrics.request_total")
    @patch("app.utils.metrics.active_requests")
    async def test_track_request_sync_function_error(self, mock_active, mock_total):
        """Test tracking sync function that raises error."""
        collector = MetricsCollector()

        @collector.track_request("POST", "/error")
        def error_endpoint():
            raise RuntimeError("Sync error")

        with pytest.raises(RuntimeError, match="Sync error"):
            error_endpoint()

        mock_total.labels.assert_called_with(method="POST", endpoint="/error", status="error")
        mock_total.labels.return_value.inc.assert_called_once()

        assert mock_active.labels.return_value.dec.call_count == 1

    def test_get_metrics_dict_with_info_type(self):
        """Test get_metrics_dict with info type metrics."""
        collector = MetricsCollector()

        mock_metric = MagicMock()
        mock_metric.name = "app_info"
        mock_metric.type = "info"
        mock_sample = MagicMock()
        mock_sample.name = "app_info"
        mock_sample.labels = {"version": "1.0.0"}
        mock_sample.value = 1.0
        mock_metric.samples = [mock_sample]

        mock_collector = MagicMock()
        mock_collector.collect.return_value = [mock_metric]

        with patch.object(collector.registry, "collect", return_value=[mock_collector]):
            metrics_dict = collector.get_metrics_dict()

        assert len(metrics_dict) == 0

    def test_get_metrics_dict_histogram_buckets(self):
        """Test get_metrics_dict excludes histogram buckets."""
        collector = MetricsCollector()

        mock_metric = MagicMock()
        mock_metric.name = "request_duration"
        mock_metric.type = "histogram"

        samples = []
        for name, value in [
            ("request_duration_bucket", 10),
            ("request_duration_count", 100),
            ("request_duration_sum", 250.5),
            ("request_duration_created", 123),
        ]:
            sample = MagicMock()
            sample.name = name
            sample.labels = {"method": "GET"}
            sample.value = value
            samples.append(sample)

        mock_metric.samples = samples

        mock_collector = MagicMock()
        mock_collector.collect.return_value = [mock_metric]

        with patch.object(collector.registry, "collect", return_value=[mock_collector]):
            metrics_dict = collector.get_metrics_dict()

        assert len(metrics_dict) == 2
        assert 'request_duration_count{method="GET"}' in metrics_dict
        assert 'request_duration_sum{method="GET"}' in metrics_dict

    def test_get_metrics_dict_summary_type(self):
        """Test get_metrics_dict with summary type metrics."""
        collector = MetricsCollector()

        mock_metric = MagicMock()
        mock_metric.name = "response_time"
        mock_metric.type = "summary"

        samples = []
        for name, value in [
            ("response_time_count", 50),
            ("response_time_sum", 125.0),
            ("response_time", 2.5),
        ]:
            sample = MagicMock()
            sample.name = name
            sample.labels = {"endpoint": "/api/test"}
            sample.value = value
            samples.append(sample)

        mock_metric.samples = samples

        mock_metric_family = MagicMock()
        mock_metric_family.collect.return_value = [mock_metric]

        with patch.object(collector.registry, "collect", return_value=[mock_metric_family]):
            metrics_dict = collector.get_metrics_dict()

        assert len(metrics_dict) > 0

    def test_get_metrics_dict_no_labels(self):
        """Test get_metrics_dict with metrics that have no labels."""
        collector = MetricsCollector()

        mock_metric = MagicMock()
        mock_metric.name = "simple_counter"
        mock_metric.type = "counter"
        mock_sample = MagicMock()
        mock_sample.name = "simple_counter"
        mock_sample.labels = {}
        mock_sample.value = 42.0
        mock_metric.samples = [mock_sample]

        mock_collector = MagicMock()
        mock_collector.collect.return_value = [mock_metric]

        with patch.object(collector.registry, "collect", return_value=[mock_collector]):
            metrics_dict = collector.get_metrics_dict()

        assert "simple_counter" in metrics_dict
        assert metrics_dict["simple_counter"] == 42.0

    def test_get_metrics_dict_complex_labels(self):
        """Test get_metrics_dict with complex label values."""
        collector = MetricsCollector()

        mock_metric = MagicMock()
        mock_metric.name = "complex_metric"
        mock_metric.type = "gauge"
        mock_sample = MagicMock()
        mock_sample.name = "complex_metric"
        mock_sample.labels = {
            "path": "/api/v1/users/123",
            "status": "success",
            "method": "GET",
        }
        mock_sample.value = 1.0
        mock_metric.samples = [mock_sample]

        mock_collector = MagicMock()
        mock_collector.collect.return_value = [mock_metric]

        with patch.object(collector.registry, "collect", return_value=[mock_collector]):
            metrics_dict = collector.get_metrics_dict()

        expected_key = 'complex_metric{path="/api/v1/users/123",status="success",method="GET"}'
        assert expected_key in metrics_dict

    @patch("app.utils.metrics.generate_latest")
    def test_get_metrics_uses_generate_latest(self, mock_generate):
        """Test that get_metrics uses generate_latest from prometheus_client."""
        collector = MetricsCollector()
        mock_generate.return_value = b"mock metrics output"

        result = collector.get_metrics()

        assert result == b"mock metrics output"
        mock_generate.assert_called_once_with(collector.registry)
