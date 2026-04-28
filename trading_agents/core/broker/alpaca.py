from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from trading_agents.core.models import AlpacaOrderIntent, AlpacaOrderStatus, TradingSignal


class AlpacaPreviewService:
    def __init__(
        self,
        enabled: bool = True,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        base_url: str = "https://paper-api.alpaca.markets",
        timeout_seconds: float = 10.0,
        client_factory=None,
    ):
        self.enabled = enabled
        self.api_key_id = api_key_id
        self.api_secret_key = api_secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client_factory = client_factory or httpx.Client
        self.symbol_mapping: dict[str, str] = {}

    def register_symbol_mapping(self, casablanca_symbol: str, alpaca_symbol: str) -> None:
        self.symbol_mapping[casablanca_symbol.upper()] = alpaca_symbol.upper()

    def submission_mode(self) -> str:
        if "paper-api.alpaca.markets" in self.base_url:
            return "paper"
        if "api.alpaca.markets" in self.base_url:
            return "live"
        return "custom"

    def _build_unmappable_order(
        self,
        *,
        signal: TradingSignal,
        mapped: str | None,
        side: str | None,
        reason: str,
    ) -> AlpacaOrderIntent:
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
            reason=reason,
            created_at=datetime.now(timezone.utc),
        )

    def _fetch_asset(self, symbol: str) -> dict:
        headers = {
            "APCA-API-KEY-ID": self.api_key_id or "",
            "APCA-API-SECRET-KEY": self.api_secret_key or "",
        }
        encoded_symbol = quote(symbol, safe="")
        with self.client_factory(base_url=self.base_url, headers=headers, timeout=self.timeout_seconds) as client:
            response = client.get(f"/v2/assets/{encoded_symbol}")
            response.raise_for_status()
            return response.json()

    def _validate_mapped_asset(self, mapped_symbol: str) -> tuple[bool, str | None]:
        if not self.enabled:
            return True, "Asset validation skipped because Alpaca integration is disabled."
        if not self.api_key_id or not self.api_secret_key:
            return True, "Asset validation skipped because Alpaca credentials are not configured."

        try:
            asset = self._fetch_asset(mapped_symbol)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return False, f"Mapped Alpaca asset '{mapped_symbol}' was not found."
            return False, f"Unable to validate Alpaca asset '{mapped_symbol}' (status {exc.response.status_code})."
        except httpx.HTTPError as exc:
            return False, f"Unable to validate Alpaca asset '{mapped_symbol}': {exc}."

        status = str(asset.get("status", "")).lower()
        tradable = bool(asset.get("tradable"))
        fractionable = bool(asset.get("fractionable"))

        if status != "active":
            return False, f"Mapped Alpaca asset '{mapped_symbol}' is not active."
        if not tradable:
            exchange = asset.get("exchange")
            if exchange:
                return False, f"Mapped Alpaca asset '{mapped_symbol}' is not tradable on Alpaca (exchange: {exchange})."
            return False, f"Mapped Alpaca asset '{mapped_symbol}' is not tradable on Alpaca."
        if not fractionable:
            return False, f"Mapped Alpaca asset '{mapped_symbol}' is not fractionable for notional day orders."
        return True, None

    def prepare_preview(self, signal: TradingSignal) -> AlpacaOrderIntent:
        mapped = self.symbol_mapping.get(signal.symbol.upper())
        side = None
        if signal.action == "BUY":
            side = "buy"
        elif signal.action in {"SELL", "EXIT", "REDUCE"}:
            side = "sell"

        if not mapped or not side:
            return self._build_unmappable_order(
                signal=signal,
                mapped=mapped,
                side=side,
                reason="No explicit Alpaca mapping exists for this Casablanca symbol or action.",
            )

        is_valid, validation_note = self._validate_mapped_asset(mapped)
        if not is_valid:
            return self._build_unmappable_order(
                signal=signal,
                mapped=mapped,
                side=side,
                reason=validation_note or "Mapped Alpaca asset failed validation.",
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
            reason=validation_note,
            broker_submission_mode=self.submission_mode(),
            created_at=datetime.now(timezone.utc),
        )

    def approve_preview(self, order: AlpacaOrderIntent, *, submission_enabled: bool) -> AlpacaOrderIntent:
        if order.status != AlpacaOrderStatus.PREPARED:
            raise ValueError("Only prepared Alpaca previews can be approved.")
        reason = order.reason
        if not submission_enabled:
            reason = "Operator approved the order command, but broker submission remains disabled by configuration."
        return order.model_copy(
            update={
                "status": AlpacaOrderStatus.APPROVED,
                "submission_eligible": bool(submission_enabled),
                "reason": reason,
            }
        )

    def reject_preview(self, order: AlpacaOrderIntent) -> AlpacaOrderIntent:
        if order.status != AlpacaOrderStatus.PREPARED:
            raise ValueError("Only prepared Alpaca previews can be rejected.")
        return order.model_copy(
            update={
                "status": AlpacaOrderStatus.REJECTED,
                "submission_eligible": False,
                "reason": "Operator rejected the Alpaca order command.",
            }
        )

    def submit_order(self, order: AlpacaOrderIntent) -> AlpacaOrderIntent:
        if order.status != AlpacaOrderStatus.APPROVED:
            raise ValueError("Only approved Alpaca previews can be submitted.")
        if not self.enabled:
            raise ValueError("Alpaca integration is disabled.")
        if not self.api_key_id or not self.api_secret_key:
            raise ValueError("Alpaca credentials are not configured for broker submission.")
        if not order.alpaca_symbol or not order.side or not order.type or not order.time_in_force:
            raise ValueError("The Alpaca order preview is incomplete and cannot be submitted.")

        headers = {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
        }
        payload = {
            "symbol": order.alpaca_symbol,
            "side": order.side,
            "type": order.type,
            "time_in_force": order.time_in_force,
            "client_order_id": order.client_order_id,
        }
        if order.notional is not None:
            payload["notional"] = f"{order.notional:.2f}"
        elif order.qty is not None:
            payload["qty"] = str(order.qty)
        else:
            raise ValueError("The Alpaca order preview has neither notional nor quantity.")

        try:
            with self.client_factory(base_url=self.base_url, headers=headers, timeout=self.timeout_seconds) as client:
                response = client.post("/v2/orders", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            message = exc.response.text or f"status {exc.response.status_code}"
            raise ValueError(f"Alpaca order submission failed: {message}") from exc
        except httpx.HTTPError as exc:
            raise ValueError(f"Alpaca order submission failed: {exc}") from exc

        return order.model_copy(
            update={
                "preview_only": False,
                "submission_eligible": True,
                "broker_order_id": body.get("id"),
                "broker_order_status": body.get("status"),
                "broker_submission_mode": self.submission_mode(),
                "submitted_at": datetime.now(timezone.utc),
                "reason": f"Submitted to Alpaca {self.submission_mode()} trading.",
            }
        )
