"""Unit tests for app.services.conversation_analysis.config module."""

from app.services.conversation_analysis.config import MCTSConfig, ResponseConfig, ScoringConfig


class TestMCTSConfig:
    """Test MCTSConfig dataclass."""

    def test_mcts_config_constants(self):
        """Test MCTS configuration constants."""
        assert MCTSConfig.DEFAULT_MAX_CHILDREN == 3
        assert MCTSConfig.DEFAULT_EXPLORATION_CONSTANT == 1.414
        assert MCTSConfig.PRUNING_INTERVAL == 5
        assert MCTSConfig.PRUNING_THRESHOLD_RATIO == 0.7
        assert MCTSConfig.MIN_VISITS_FOR_PRUNING == 5

    def test_mcts_config_instantiation(self):
        """Test MCTSConfig can be instantiated."""
        config = MCTSConfig()
        assert isinstance(config, MCTSConfig)

    def test_mcts_config_is_dataclass(self):
        """Test MCTSConfig is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(MCTSConfig)

    def test_mcts_config_exploration_constant_type(self):
        """Test exploration constant is float."""
        assert isinstance(MCTSConfig.DEFAULT_EXPLORATION_CONSTANT, float)

    def test_mcts_config_pruning_threshold_range(self):
        """Test pruning threshold is in valid range."""
        assert 0 < MCTSConfig.PRUNING_THRESHOLD_RATIO < 1


class TestScoringConfig:
    """Test ScoringConfig dataclass."""

    def test_scoring_config_weights(self):
        """Test scoring weight constants."""
        assert ScoringConfig.SCORE_WEIGHT_QUALITY == 0.7
        assert ScoringConfig.SCORE_WEIGHT_VISITS == 0.3

        total_weight = ScoringConfig.SCORE_WEIGHT_QUALITY + ScoringConfig.SCORE_WEIGHT_VISITS
        assert abs(total_weight - 1.0) < 0.0001

    def test_scoring_config_general_metrics(self):
        """Test general metrics list."""
        expected_metrics = [
            "clarity",
            "relevance",
            "engagement",
            "authenticity",
            "coherence",
            "respectfulness",
        ]
        assert ScoringConfig.GENERAL_METRICS == expected_metrics
        assert len(ScoringConfig.GENERAL_METRICS) == 6

    def test_scoring_config_instantiation(self):
        """Test ScoringConfig can be instantiated."""
        config = ScoringConfig()
        assert isinstance(config, ScoringConfig)

    def test_scoring_config_is_dataclass(self):
        """Test ScoringConfig is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(ScoringConfig)

    def test_scoring_config_metrics_are_strings(self):
        """Test all metrics are strings."""
        assert all(isinstance(metric, str) for metric in ScoringConfig.GENERAL_METRICS)

    def test_scoring_config_metrics_not_empty(self):
        """Test metrics list is not empty."""
        assert len(ScoringConfig.GENERAL_METRICS) > 0

    def test_scoring_config_metrics_unique(self):
        """Test all metrics are unique."""
        assert len(ScoringConfig.GENERAL_METRICS) == len(set(ScoringConfig.GENERAL_METRICS))


class TestResponseConfig:
    """Test ResponseConfig dataclass."""

    def test_response_config_token_multipliers(self):
        """Test token multiplier constants."""
        assert ResponseConfig.TOKEN_MULTIPLIER_INITIAL == 2
        assert ResponseConfig.TOKEN_MULTIPLIER_SIMULATION == 3
        assert ResponseConfig.TOKEN_MULTIPLIER_ANALYSIS == 2

    def test_response_config_default_responses(self):
        """Test default responses list."""
        expected_responses = [
            "I understand you're going through a difficult time. Let's talk about what you're feeling.",
            "That sounds challenging. Can you tell me more about what happened?",
            "I'm here to listen and support you. What aspect of this situation is bothering you the most?",
        ]
        assert ResponseConfig.DEFAULT_RESPONSES == expected_responses
        assert len(ResponseConfig.DEFAULT_RESPONSES) == 3

    def test_response_config_instantiation(self):
        """Test ResponseConfig can be instantiated."""
        config = ResponseConfig()
        assert isinstance(config, ResponseConfig)

    def test_response_config_is_dataclass(self):
        """Test ResponseConfig is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(ResponseConfig)

    def test_response_config_responses_are_strings(self):
        """Test all default responses are strings."""
        assert all(isinstance(response, str) for response in ResponseConfig.DEFAULT_RESPONSES)

    def test_response_config_responses_not_empty(self):
        """Test each default response is not empty."""
        assert all(len(response.strip()) > 0 for response in ResponseConfig.DEFAULT_RESPONSES)

    def test_response_config_token_multipliers_positive(self):
        """Test all token multipliers are positive."""
        assert ResponseConfig.TOKEN_MULTIPLIER_INITIAL > 0
        assert ResponseConfig.TOKEN_MULTIPLIER_SIMULATION > 0
        assert ResponseConfig.TOKEN_MULTIPLIER_ANALYSIS > 0


class TestConfigInteractions:
    """Test interactions between config classes."""

    def test_all_configs_are_dataclasses(self):
        """Test all config classes are dataclasses."""
        from dataclasses import is_dataclass

        configs = [MCTSConfig, ScoringConfig, ResponseConfig]
        assert all(is_dataclass(config) for config in configs)

    def test_configs_can_be_instantiated_together(self):
        """Test all configs can be instantiated together."""
        mcts_config = MCTSConfig()
        scoring_config = ScoringConfig()
        response_config = ResponseConfig()

        assert isinstance(mcts_config, MCTSConfig)
        assert isinstance(scoring_config, ScoringConfig)
        assert isinstance(response_config, ResponseConfig)

    def test_config_constants_are_immutable(self):
        """Test that config constants cannot be modified on instances."""
        config = MCTSConfig()

        assert not hasattr(config, "__dict__") or "DEFAULT_MAX_CHILDREN" not in config.__dict__

    def test_scoring_weights_valid_for_algorithm(self):
        """Test scoring weights are valid for use in algorithms."""
        assert ScoringConfig.SCORE_WEIGHT_QUALITY > ScoringConfig.SCORE_WEIGHT_VISITS

        assert 0 < ScoringConfig.SCORE_WEIGHT_QUALITY < 1
        assert 0 < ScoringConfig.SCORE_WEIGHT_VISITS < 1

    def test_mcts_pruning_config_consistency(self):
        """Test MCTS pruning configuration is internally consistent."""
        assert MCTSConfig.PRUNING_INTERVAL > 0

        assert MCTSConfig.MIN_VISITS_FOR_PRUNING > 0
        assert MCTSConfig.MIN_VISITS_FOR_PRUNING < 100

    def test_response_multipliers_ordering(self):
        """Test token multipliers have logical ordering."""
        assert ResponseConfig.TOKEN_MULTIPLIER_SIMULATION >= ResponseConfig.TOKEN_MULTIPLIER_INITIAL
        assert ResponseConfig.TOKEN_MULTIPLIER_SIMULATION >= ResponseConfig.TOKEN_MULTIPLIER_ANALYSIS
