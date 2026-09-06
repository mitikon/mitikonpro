from racing_lambda.chukyo_variant_comparison import chukyo_one_at_a_time_comparison


def test_one_at_a_time_variants_are_frozen_and_comparable():
    rows = {row.name: row for row in chukyo_one_at_a_time_comparison()}
    assert rows["baseline"].ranking == ("4", "6", "2", "5", "9", "7", "3", "8", "1")
    assert rows["recent_saturation_fix"].ranking[:5] == ("4", "6", "5", "2", "9")
    assert rows["surface_mismatch_gate"].ranking[:5] == ("4", "2", "5", "6", "9")
    assert rows["neutralize_raw_position"].ranking[:5] == ("4", "2", "6", "5", "7")
    assert rows["market_gap_removed"].ranking[:5] == ("4", "6", "5", "9", "2")
    assert rows["juvenile_completeness"].ranking[:5] == ("4", "6", "2", "5", "9")


def test_no_single_structural_fix_improves_top5_hit_count_yet():
    rows = chukyo_one_at_a_time_comparison()
    assert all(row.top5_hits == 3 for row in rows)


def test_structural_changes_have_directional_effects():
    rows = {row.name: row for row in chukyo_one_at_a_time_comparison()}
    baseline = rows["baseline"].ranks_of_interest
    # Surface gate demotes the dirt-win horse 6 from 2nd to 4th.
    assert rows["surface_mismatch_gate"].ranks_of_interest["6"] > baseline["6"]
    # Removing raw forward-position preference lets horse 7 enter the top five.
    assert rows["neutralize_raw_position"].ranks_of_interest["7"] < baseline["7"]
    # Removing market gap improves the strongly backed horse 9 from 5th to 4th.
    assert rows["market_gap_removed"].ranks_of_interest["9"] < baseline["9"]
