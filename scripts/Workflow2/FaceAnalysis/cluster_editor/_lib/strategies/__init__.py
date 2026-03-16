from .base import EditorStrategy
from .face import FaceModeStrategy
from .location import LocationModeStrategy
from .cleaning import CleaningModeStrategy
from .matches import MatchesModeStrategy

def get_strategy(mode: str) -> EditorStrategy:
    """Фабричный метод для получения стратегии по имени режима."""
    strategies = {
        "face": FaceModeStrategy,
        "location": LocationModeStrategy,
        "cleaning": CleaningModeStrategy,
        "matches": MatchesModeStrategy
    }
    
    strategy_cls = strategies.get(mode)
    if not strategy_cls:
        raise ValueError(f"Unknown mode: {mode}")
    
    return strategy_cls()