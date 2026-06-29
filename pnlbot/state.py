from .models import BotState


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
