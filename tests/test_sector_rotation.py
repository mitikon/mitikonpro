import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda.paper_pca_sub import PAPER_COMPONENTS, PAPER_WINDOW
from leading_signal_lambda.sector_rotation import (
    LAGGED_COLUMNS,
    SECTOR_ETFS,
    LaggedSectorPcaSubModel,
    build_sector_rotation_prior_basis,
    make_lagged_sector_pairs,
)
from leading_signal_lambda.sector_rotation_validation import (
    BUY,
    EXIT_RISK,
    EXIT_SIGNAL,
    EXIT_TIME,
    HOLD,
    STAY,
    TAKE_PROFIT,
    PositionState,
    decide_position_action,
    run_sector_rotation_backtest,
)


def sample_returns(rows: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.008, size=(rows, 3))
    loadings = rng.normal(size=(3, len(SECTOR_ETFS)))
    noise = rng.normal(0.0, 0.003, size=(rows, len(SECTOR_ETFS)))
    return pd.DataFrame(
        factors @ loadings * 0.3 + noise,
        index=pd.bdate_range(start, periods=rows),
        columns=SECTOR_ETFS,
    )


def prices_from_returns(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = 100.0 * (1.0 + returns).cumprod()
    open_price = close.shift(1)
    open_price.iloc[0] = close.iloc[0]
    return open_price, close


def test_lagged_pairs_align_today_with_next_day_without_lookahead():
    returns = sample_returns(5, "2022-01-03", 1)
    pairs = make_lagged_sector_pairs(returns)
    assert list(pairs.columns) == list(LAGGED_COLUMNS)
    assert pairs.index[0] == returns.index[1]
    np.testing.assert_allclose(pairs.iloc[0, : len(SECTOR_ETFS)], returns.iloc[0])
    np.testing.assert_allclose(pairs.iloc[0, len(SECTOR_ETFS) :], returns.iloc[1])


def test_sector_rotation_prior_is_three_dimensional_and_orthonormal():
    basis = build_sector_rotation_prior_basis()
    assert basis.shape == (2 * len(SECTOR_ETFS), PAPER_COMPONENTS)
    np.testing.assert_allclose(basis.T @ basis, np.eye(PAPER_COMPONENTS), atol=1e-12)


def test_sector_model_predicts_all_eleven_next_day_etfs():
    rolling_returns = sample_returns(PAPER_WINDOW + 1, "2022-01-03", 2)
    prior_returns = sample_returns(500, "2019-01-02", 3)
    model = LaggedSectorPcaSubModel().fit(
        make_lagged_sector_pairs(rolling_returns),
        make_lagged_sector_pairs(prior_returns),
    )
    current = sample_returns(1, "2023-01-03", 4).iloc[0]
    signal = model.predict(current)
    assert model.components_.shape == (2 * len(SECTOR_ETFS), PAPER_COMPONENTS)
    assert list(signal.next_standardized_prediction.index) == list(SECTOR_ETFS)
    assert signal.transmission.shape == (len(SECTOR_ETFS), len(SECTOR_ETFS))


@pytest.mark.parametrize(
    ("position", "values", "expected"),
    [
        (None, {"top_score": 0.5, "top_lambda": 0.2}, BUY),
        (None, {"top_score": 0.1, "top_lambda": 0.2}, STAY),
        ("held", {"unrealized_return": 0.11}, TAKE_PROFIT),
        ("held", {"unrealized_return": -0.06}, EXIT_RISK),
        ("old", {}, EXIT_TIME),
        ("held", {"held_score": -0.1, "held_lambda": 0.2}, EXIT_SIGNAL),
        ("held", {"held_score": 0.5, "held_lambda": 0.2}, HOLD),
    ],
)
def test_position_state_machine(position, values, expected):
    state = None
    if position:
        days = 15 if position == "old" else 2
        state = PositionState("XLK", 100.0, pd.Timestamp("2022-01-03"), days)
    defaults = {
        "top_symbol": "XLF",
        "top_score": 0.5,
        "top_lambda": 0.2,
        "held_score": 0.5,
        "held_lambda": 0.2,
        "unrealized_return": 0.0,
    }
    defaults.update(values)
    decision = decide_position_action(state, **defaults)
    assert decision.action == expected


def test_sector_backtest_freezes_past_rankings_when_future_changes():
    returns = sample_returns(145, "2022-01-03", 5)
    prior = sample_returns(500, "2019-01-02", 6)
    opens, closes = prices_from_returns(returns)
    first = run_sector_rotation_backtest(
        returns, prior, opens, closes, calibration_observations=10
    )
    changed = returns.copy()
    changed.iloc[-5:] *= 10.0
    changed_opens, changed_closes = prices_from_returns(changed)
    second = run_sector_rotation_backtest(
        changed, prior, changed_opens, changed_closes, calibration_observations=10
    )
    cutoff = returns.index[-6]
    frozen = ["rank", "raw_score", "signal_lambda", "effective_score", "current_distortion"]
    pd.testing.assert_frame_equal(
        first.rankings.loc[:cutoff, frozen], second.rankings.loc[:cutoff, frozen]
    )


def test_sector_backtest_outputs_required_signal_types_and_metrics():
    returns = sample_returns(150, "2022-01-03", 7)
    prior = sample_returns(500, "2019-01-02", 8)
    opens, closes = prices_from_returns(returns)
    result = run_sector_rotation_backtest(
        returns,
        prior,
        opens,
        closes,
        calibration_observations=5,
        entry_threshold=0.0,
        take_profit=0.02,
        stop_loss=0.02,
        max_holding_days=3,
    )
    assert set(result.decisions["action"]) <= {
        BUY, HOLD, TAKE_PROFIT, EXIT_RISK, EXIT_TIME, EXIT_SIGNAL, STAY
    }
    assert {"annualized_return", "profitable_trade_rate", "completed_trades"} <= set(
        result.metrics
    )


def test_sector_frozen_outputs_cannot_be_overwritten(tmp_path):
    returns = sample_returns(100, "2022-01-03", 9)
    prior = sample_returns(500, "2019-01-02", 10)
    opens, closes = prices_from_returns(returns)
    result = run_sector_rotation_backtest(
        returns, prior, opens, closes, calibration_observations=5
    )
    result.save_frozen(tmp_path / "rotation")
    with pytest.raises(FileExistsError):
        result.save_frozen(tmp_path / "rotation")
