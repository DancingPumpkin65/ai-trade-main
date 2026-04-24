from __future__ import annotations

from datetime import datetime, timezone

from trading_agents.core.models import AlpacaOrderIntent, AlpacaOrderStatus, TradingSignal


class AlpacaPreviewService:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.symbol_mapping: dict[str, str] = {}

    def register_symbol_mapping(self, casablanca_symbol: str, alpaca_symbol: str) -> None:
        self.symbol_mapping[casablanca_symbol.upper()] = alpaca_symbol.upper()

    def prepare_preview(self, signal: TradingSignal) -> AlpacaOrderIntent:
        mapped = self.symbol_mapping.get(signal.symbol.upper())
        side = None
        if signal.action == "BUY":
            side = "buy"
        elif signal.action in {"SELL", "EXIT", "REDUCE"}:
            side = "sell"

        if not mapped or not side:
            return AlpacaOrderIntent(
                request_id=signal.request_id,
                client_order_id=signal.request_id,
                source_symbol=signal.symbol,
                alpaca_symbol=mapped,
                side=side,
                type="market" if side else None,
                time_in_force="day" if side else None,
                notional=signal.position_value_mad if side else None,
                status=AlpacaOrderStatus.UNMAPPABLE,
                reason="No explicit Alpaca mapping exists for this Casablanca symbol or action.",
                created_at=datetime.now(timezone.utc),
            )

        return AlpacaOrderIntent(
            request_id=signal.request_id,
            client_order_id=signal.request_id,
            source_symbol=signal.symbol,
            alpaca_symbol=mapped,
            side=side,
            type="market",
            time_in_force="day",
            notional=signal.position_value_mad,
            submission_eligible=False,
            status=AlpacaOrderStatus.PREPARED,
            created_at=datetime.now(timezone.utc),
        )
