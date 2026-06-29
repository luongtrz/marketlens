"""Tests for shared asset tag detection."""

from shared.asset_tags import detect_asset_tags, primary_asset_tag, text_matches_symbol


def test_eth_whole_word() -> None:
    tags = detect_asset_tags("Ethereum Foundation sells ETH", "Large stake moved on chain")
    assert tags == frozenset({"ETH"})


def test_eth_not_tether_substring() -> None:
    assert detect_asset_tags("Tether freezes USDT on Tron", "Stablecoin issuer Tether said…") == frozenset()
    assert primary_asset_tag("Tether discloses stake in Antalpha", "") == "General"


def test_btc_and_eth_both() -> None:
    tags = detect_asset_tags("Bitcoin and Ethereum rally", "")
    assert tags == frozenset({"BTC", "ETH"})


def test_eth_url_slug() -> None:
    tags = detect_asset_tags(
        "Some headline without tickers",
        "",
        "https://decrypt.co/news/ethereum/vitalik-proposal",
    )
    assert "ETH" in tags


def test_megaeth_ecosystem() -> None:
    assert "ETH" in detect_asset_tags("MegaETH hits KPI milestone", "")


def test_text_matches_ethusdt() -> None:
    assert text_matches_symbol("ETH staking flows rise", "", "ETHUSDT")
    assert not text_matches_symbol("Tether mints USDT", "", "ETHUSDT")
