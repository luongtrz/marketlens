from __future__ import annotations

from stockmem.src.search.taxonomy import (
    BEARISH_FACTORS,
    BULLISH_FACTORS,
    FACTOR_TYPE_MAP,
    NEUTRAL_FACTORS,
    NUM_GROUPS,
    NUM_TYPES,
    build_group_vector,
    build_type_vector,
    get_factor_group,
    get_factor_sentiment,
    get_factor_type,
)


def test_taxonomy_has_62_types_and_13_groups() -> None:
    assert NUM_TYPES == 62
    assert NUM_GROUPS == 13


def test_factor_type_map_covers_all_buckets() -> None:
    assert len(BULLISH_FACTORS) + len(BEARISH_FACTORS) + len(NEUTRAL_FACTORS) == len(FACTOR_TYPE_MAP)
    assert len(BULLISH_FACTORS) == 40
    assert len(BEARISH_FACTORS) == 40
    assert len(NEUTRAL_FACTORS) == 20


def test_build_type_vector_dim_62_binary() -> None:
    vec = build_type_vector(["Record ETF inflows", "Fed holds interest rate steady"])
    assert len(vec) == 62
    assert all(v in (0, 1) for v in vec)
    assert sum(vec) == 2


def test_build_group_vector_dim_13_binary() -> None:
    vec = build_group_vector(["Record ETF inflows", "Fed holds interest rate steady"])
    assert len(vec) == 13
    assert all(v in (0, 1) for v in vec)
    assert sum(vec) == 2  # Market Performance + Macroeconomic


def test_oov_factor_is_dropped_silently() -> None:
    vec_type = build_type_vector(["completely unknown factor"])
    vec_group = build_group_vector(["completely unknown factor"])
    assert sum(vec_type) == 0
    assert sum(vec_group) == 0


def test_factors_in_same_group_collapse_to_one_group_bit() -> None:
    # Both factors live under "Market Performance"
    vec = build_group_vector(["Record ETF inflows", "Significant volume surge"])
    assert sum(vec) == 1


def test_factors_mapping_to_same_type_dedupe_to_one_type_bit() -> None:
    # "Institutional adoption increasing" and "BlackRock increases BTC holdings"
    # both map to "Institutional Adoption".
    vec = build_type_vector(
        ["Institutional adoption increasing", "BlackRock increases BTC holdings"]
    )
    assert sum(vec) == 1


def test_helpers_return_consistent_mapping() -> None:
    f = "Record ETF inflows"
    assert get_factor_type(f) == "ETF Flow"
    assert get_factor_group(f) == "Market Performance"
    assert get_factor_sentiment(f) == "bullish"

    assert get_factor_sentiment("Major exchange hack") == "bearish"
    assert get_factor_sentiment("Market sideways waiting for signal") == "neutral"
    assert get_factor_sentiment("unknown") is None
