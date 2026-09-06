from racing_lambda.chukyo_pairwise_comparison import chukyo_pairwise_comparison


def test_pairwise_structural_comparison_is_frozen():
    rows = {row.name: row for row in chukyo_pairwise_comparison()}
    assert rows["surface+position"].ranking == ("4", "2", "5", "7", "9", "3", "6", "8", "1")
    assert rows["market_removed+juvenile"].ranking == ("4", "6", "5", "9", "2", "7", "3", "8", "1")
    assert rows["market_positive+juvenile"].ranking == ("4", "6", "2", "5", "9", "7", "3", "8", "1")
    assert rows["surface+market_removed"].ranking == ("4", "5", "9", "2", "6", "7", "3", "8", "1")
    assert rows["position+market_removed"].ranking == ("4", "9", "2", "7", "6", "5", "3", "8", "1")
    assert rows["recent+surface"].ranking == ("4", "5", "2", "6", "9", "7", "3", "8", "1")


def test_position_plus_market_removed_is_only_pair_reaching_four_top5_hits():
    rows = chukyo_pairwise_comparison()
    winners = [row for row in rows if row.top5_hits >= 4]
    assert len(winners) == 1
    assert winners[0].name == "position+market_removed"
    assert winners[0].top5_hits == 4
    assert winners[0].top3_hits == 2
