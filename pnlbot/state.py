from typing import List, Optional

from .models import BotState


def update_spot_balance_range(
    state: BotState,
    total_spot: float,
    *,
    asset_snapshot: Optional[List[dict]] = None,
    observed_at: Optional[str] = None,
) -> bool:
    if total_spot <= 0:
        return False

    snapshot = list(asset_snapshot or [])
    prev_values = (
        state.max_spot_balance,
        state.min_spot_balance,
        state.max_spot_assets,
        state.min_spot_assets,
        state.max_spot_observed_at,
        state.min_spot_observed_at,
    )

    if state.max_spot_balance == 0 or total_spot > state.max_spot_balance:
        state.max_spot_balance = total_spot
        state.max_spot_assets = snapshot
        state.max_spot_observed_at = observed_at

    if state.min_spot_balance == 0 or total_spot < state.min_spot_balance:
        state.min_spot_balance = total_spot
        state.min_spot_assets = snapshot
        state.min_spot_observed_at = observed_at

    return prev_values != (
        state.max_spot_balance,
        state.min_spot_balance,
        state.max_spot_assets,
        state.min_spot_assets,
        state.max_spot_observed_at,
        state.min_spot_observed_at,
    )
