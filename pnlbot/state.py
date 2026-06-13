from .models import BotState


def update_pnl_range(state: BotState, pnl: float) -> bool:
    prev_max = state.max_pnl
    prev_min = state.min_pnl
    state.max_pnl = max(state.max_pnl, pnl)
    state.min_pnl = min(state.min_pnl, pnl)
    return state.max_pnl != prev_max or state.min_pnl != prev_min


def update_spot_balance_range(state: BotState, total_spot: float) -> bool:
    if total_spot <= 0:
        return False

    prev_max = state.max_spot_balance
    prev_min = state.min_spot_balance

    if state.max_spot_balance == 0:
        state.max_spot_balance = total_spot
    if state.min_spot_balance == 0:
        state.min_spot_balance = total_spot

    state.max_spot_balance = max(state.max_spot_balance, total_spot)
    state.min_spot_balance = min(state.min_spot_balance, total_spot)
    return state.max_spot_balance != prev_max or state.min_spot_balance != prev_min
