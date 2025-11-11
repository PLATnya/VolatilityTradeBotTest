import os
import json
import time
import hmac
import base64
import hashlib
import requests
from typing import Literal, Optional
from datetime import datetime, timezone

from urllib.parse import urlencode
from dotenv import load_dotenv

class OkxClient:
    def __init__(self, api_key: str | None, api_secret: str | None, passphrase: str | None,
                 base_url: str = "https://www.okx.com", simulated: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = base_url.rstrip("/")
        self.simulated = simulated

    @classmethod
    def from_env(cls) -> "OkxClient":
        load_dotenv()
        return cls(
            api_key=os.getenv("OKX_API_KEY"),
            api_secret=os.getenv("OKX_API_SECRET"),
            passphrase=os.getenv("OKX_API_PASSPHRASE"),
            base_url=os.getenv("OKX_BASE_URL", "https://www.okx.com"),
            simulated=os.getenv("OKX_SIMULATED", "").strip() in {"1", "true", "TRUE"},
        )

    def _ts(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"

    def _sign(self, ts: str, method: str, request_path: str, body: str = "") -> str:
        prehash = f"{ts}{method.upper()}{request_path}{body}"
        digest = hmac.new(self.api_secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def _headers(self, ts: str, sign: str, auth: bool) -> dict:
        h = {"Content-Type": "application/json"}
        if auth:
            if not self.api_key or not self.api_secret or not self.passphrase:
                raise RuntimeError("OKX credentials not configured. Set OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE.")
            h.update({
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": sign,
                "OK-ACCESS-TIMESTAMP": ts,
                "OK-ACCESS-PASSPHRASE": self.passphrase,
            })
            if self.simulated:
                h["x-simulated-trading"] = "1"
        return h

    def _request(self, method: str, path: str, params: dict | None = None, data: dict | None = None, auth: bool = False) -> dict:
        query = ""
        if params:
            query = "?" + urlencode(sorted(params.items()), doseq=True)

        request_path = f"{path}{query}"
        url = f"{self.base_url}{request_path}"

        body_str = ""
        if method.upper() != "GET" and data is not None:
            body_str = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

        ts = self._ts()
        sign = self._sign(ts, method, request_path, body_str) if auth else ""

        headers = self._headers(ts, sign, auth=auth)

        timeout = (10, 30)
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            resp = requests.post(url, headers=headers, data=body_str or "{}", timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        try:
            payload = resp.json()
        except Exception:
            resp.raise_for_status()
            raise

        if payload.get("code") != "0":
            print(payload)
            code = payload.get("code")
            msg = payload.get("msg")
            raise RuntimeError(f"OKX API error code={code} msg={msg}")
        return payload

    def get_ticker(self, inst_id: str) -> dict:
        r = self._request("GET", "/api/v5/market/ticker", params={"instId": inst_id}, auth=False)
        d = (r.get("data") or [{}])[0]
        return {
            "instId": d.get("instId", inst_id),
            "last": d.get("last"),
            "bidPx": d.get("bidPx"),
            "askPx": d.get("askPx"),
            "high24h": d.get("high24h"),
            "low24h": d.get("low24h"),
            "vol24h": d.get("vol24h"),
            "ts": d.get("ts"),
        }

    def get_candles(
        self,
        inst_id: str,
        bar: str = "1m",
        limit: int = 100,
        before: str | None = None,
        after: str | None = None,
    ) -> list[dict]:
        """
        Get candlestick data (OHLCV) for an instrument.

        Args:
            inst_id: Instrument ID, e.g. "BTC-USDT-SWAP"
            bar: Candle period - "1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "8H", "12H", "1D", "1W", "1M"
            limit: Number of candles to fetch (1-300, default 100)
            before: Pagination - retrieve records earlier than the requested ts (timestamp in ms as string).
            after: Pagination - retrieve records later than the requested ts (timestamp in ms as string).

        Returns:
            List of candle dictionaries with keys: ts, o, h, l, c, vol, volCcy
        """
        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(min(max(1, limit), 300))  # Clamp between 1 and 300
        }

        def convert_time_to_ms(time: str) -> str:
            # Convert "ddmmyyyy_ssmmhh" to ms timestamp
            # Example: '01052023_300412' -> 1 May 2023, 12:04:30
            time_ts = 0
            try:
                date_part, time_part = time.split('_')
                day = int(date_part[:2])
                month = int(date_part[2:4])
                year = int(date_part[4:8])
                sec = int(time_part[:2])
                min_ = int(time_part[2:4])
                hour = int(time_part[4:6])
                from datetime import datetime, timezone
                dt = datetime(year, month, day, hour, min_, sec, tzinfo=timezone.utc)
                time_ts = str(int(dt.timestamp() * 1000))
            except Exception as e:
                raise ValueError(f"Invalid 'before' format: {before} ({e})")

            return time_ts

        if before is not None:
            params["before"] = convert_time_to_ms(before)

        if after is not None:
            params["after"] = convert_time_to_ms(after)

        r = self._request("GET", "/api/v5/market/candles", params=params, auth=False)
        candles = r.get("data", [])
        # OKX returns candles in reverse chronological order (newest first)
        # Return in chronological order (oldest first) for plotting
        candles.reverse()
        return candles

    def get_balance(self, ccy: str | None = None) -> dict:
        params = {}
        if ccy:
            params["ccy"] = ccy
        r = self._request("GET", "/api/v5/account/balance", params=params, auth=True)
        d = (r.get("data") or [{}])[0]
        return {
            "uTime": d.get("uTime"),
            "totalEq": d.get("totalEq"),
            "details": d.get("details", []),
        }

    def get_positions(self, inst_type: str | None = None, inst_id: str | None = None, pos_id: str | None = None) -> list[dict]:
        params = {}
        if inst_type:
            params["instType"] = inst_type
        if inst_id:
            params["instId"] = inst_id
        if pos_id:
            params["posId"] = pos_id
        r = self._request("GET", "/api/v5/account/positions", params=params, auth=True)
        return r.get("data", [])



    def place_order(
        self,
        instrument_id: str,
        trade_mode: str,
        order_side: str,
        order_type: str,
        size: str,
        position_side: str | None = None,
        price: str | None = None,
        price_usd: str | None = None,
        price_volume: str | None = None,
        reduce_only: bool | None = None,
        currency: str | None = None,
        client_order_id: str | None = None,
        order_tag: str | None = None,
        target_currency: str | None = None,
        ban_amend: bool | None = None,
        price_amend_type: str | None = None,
        trade_quote_currency: str | None = None,
        stop_mode: str | None = None,
        attached_algo_orders: list[dict] | None = None,
    ) -> dict:
        payload: dict[str, object] = {
            "instId": instrument_id,
            "tdMode": trade_mode,
            "side": order_side,
            "ordType": order_type,
            "sz": str(size),
        }
        if position_side:
            payload["posSide"] = position_side
        if price:
            payload["px"] = str(price)
        if price_usd:
            payload["pxUsd"] = str(price_usd)
        if price_volume:
            payload["pxVol"] = str(price_volume)
        if reduce_only is not None:
            payload["reduceOnly"] = "true" if reduce_only else "false"
        if currency:
            payload["ccy"] = currency
        if client_order_id:
            payload["clOrdId"] = client_order_id
        if order_tag:
            payload["tag"] = order_tag
        if target_currency:
            payload["tgtCcy"] = target_currency
        if ban_amend is not None:
            payload["banAmend"] = "true" if ban_amend else "false"
        if price_amend_type:
            payload["pxAmendType"] = price_amend_type
        if trade_quote_currency:
            payload["tradeQuoteCcy"] = trade_quote_currency
        if stop_mode:
            payload["stpMode"] = stop_mode
        if attached_algo_orders:
            payload["attachAlgoOrds"] = attached_algo_orders
        response = self._request("POST", "/api/v5/trade/order", data=payload, auth=True)
        data = (response.get("data") or [{}])[0]
        return {
            "ordId": data.get("ordId"),
            "clOrdId": data.get("clOrdId"),
            "sCode": data.get("sCode"),
            "sMsg": data.get("sMsg"),
            "request": payload,
        }

    def place_stop_loss_take_profit(
        self,
        inst_id: str,
        stop_loss_price: Optional[str] = None,
        take_profit_price: Optional[str] = None,
        stop_loss_percent: Optional[float] = None,
        take_profit_percent: Optional[float] = None,
        pos_id: Optional[str] = None,
    ) -> dict:
        """
        Place stop loss and take profit orders for an open position on OKX.

        Args:
            inst_id: Instrument ID (e.g., "BTC-USDT-SWAP").
            stop_loss_price: Stop loss trigger price. If None, use stop_loss_percent.
            take_profit_price: Take profit trigger price. If None, use take_profit_percent.
            stop_loss_percent: Stop loss as percent from entry. Used if stop_loss_price is None.
            take_profit_percent: Take profit as percent from entry. Used if take_profit_price is None.
            pos_id: Optional position ID to target. Defaults to first matching position.

        Returns:
            Dict with position, stop_loss_order, and take_profit_order.
        """
       
        trade_mode = "cross"
        # Get the position
        positions = self.get_positions(inst_id=inst_id, pos_id=pos_id)
        if not positions:
            raise ValueError(f"No open position found for instrument {inst_id}")
        
        # Use the first position if pos_id not specified, or find matching one
        if pos_id:
            position = next((p for p in positions if p.get("posId") == pos_id), None)
            if not position:
                raise ValueError(f"Position {pos_id} not found")
        else:
            # Find a position with actual size (not zero)
            position = next((p for p in positions if float(p.get("pos", "0")) != 0), None)
            if not position:
                raise ValueError(f"No active position found for instrument {inst_id}")
        

        pos_size = position.get("pos", "0")
        if float(pos_size) == 0:
            raise ValueError(f"Position size is zero for instrument {inst_id}")
        
        pos_side = position.get("posSide", "net")

        avg_px = position.get("avgPx", "0")
        if float(avg_px) == 0:
            raise ValueError(f"Position average price is zero for instrument {inst_id}")
        
        # Determine if it's a long or short position
        is_long = float(pos_size) > 0
        
        # Calculate trigger prices if percentages are provided
        entry_price = float(avg_px)
        
        if stop_loss_price is None:
            if stop_loss_percent is None:
                raise ValueError("Either stop_loss_price or stop_loss_percent must be provided")
            if is_long:
                stop_loss_price = str(entry_price * (1 - stop_loss_percent / 100))
            else:
                stop_loss_price = str(entry_price * (1 + stop_loss_percent / 100))
        
        if take_profit_price is None:
            if take_profit_percent is None:
                raise ValueError("Either take_profit_price or take_profit_percent must be provided")
            if is_long:
                take_profit_price = str(entry_price * (1 + take_profit_percent / 100))
            else:
                take_profit_price = str(entry_price * (1 - take_profit_percent / 100))
        
        # Determine order side (opposite of position side to close)
       
        
        # Use absolute value of position size
        size = str(abs(float(pos_size)))
        
        order_side = "sell" if is_long else "buy"
        pos_side = pos_side if pos_side in ["long", "short"] else None


        # Place stop loss order
        stop_loss_result = self._place_algo_order(
            algo_type="sl",
            inst_id=inst_id,
            trade_mode=trade_mode,
            order_side=order_side,
            size=size,
            trigger_price=stop_loss_price,
            pos_side=pos_side,
            execute_price="-1",
        )
        

        # Place take profit order
        take_profit_result = self._place_algo_order(
            algo_type="tp",
            inst_id=inst_id,
            trade_mode=trade_mode,
            order_side=order_side,
            size=size,
            trigger_price=take_profit_price,
            pos_side=pos_side,
            execute_price="-1",
        )
        
        return {
            "position": {
                "instId": position.get("instId"),
                "posId": position.get("posId"),
                "pos": pos_size,
                "posSide": pos_side,
                "avgPx": avg_px,
                "entry_price": entry_price,
            },
            "stop_loss_order": stop_loss_result,
            "take_profit_order": take_profit_result,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
        }


    def _place_algo_order(
        self,
        algo_type: Literal["tp", "sl"],
        inst_id: str,
        trade_mode: str,
        order_side: str,
        size: str,
        trigger_price: str,
        pos_side: Optional[str] = None,
        execute_price: str = "-1",  # -1 means market price
    ) -> dict:
        """
        Place a stop loss order on OKX.
        
        Args:
            inst_id: Instrument ID.
            trade_mode: Trade mode.
            order_side: Order side.
            size: Order size.
            trigger_price: Trigger price.
            pos_side: Position side.
            execute_price: Execution price.

        Returns:
            Dict with order id, client order id, status code, status message, and request payload.
        """
        payload = {
            "instId": inst_id,
            "tdMode": trade_mode,
            "side": order_side,
            "ordType": "conditional",
            "sz": size
        }

        if algo_type == "tp":
            payload["tpTriggerPx"] = trigger_price
            payload["tpOrdPx"] = execute_price
        elif algo_type == "sl":
            payload["slTriggerPx"] = trigger_price
            payload["slOrdPx"] = execute_price
        
        if pos_side:
            payload["posSide"] = pos_side
        
        response = self._request("POST", "/api/v5/trade/order-algo", data=payload, auth=True)
        data = (response.get("data") or [{}])[0]
        
        return {
            "algoId": data.get("algoId"),
            "clOrdId": data.get("clOrdId"),
            "sCode": data.get("sCode"),
            "sMsg": data.get("sMsg"),
            "request": payload,
        }

    def set_margin_leverage(
        self,
        leverage: str,
        instrument_id: str | None = None,
        margin_mode: str = "cross",
    ) -> dict:
        payload: dict[str, object] = {
            "lever": str(leverage),
            "mgnMode": margin_mode,
        }
        if instrument_id:
            payload["instId"] = instrument_id
        response = self._request("POST", "/api/v5/account/set-leverage", data=payload, auth=True)
        return (response.get("data") or [{}])[0]

    def close_position(self, inst_id: str, mgn_mode: str, pos_side: str | None = None, ccy: str | None = None, auto_cxl: bool | None = None) -> list[dict]:
        payload = {
            "instId": inst_id,
            "mgnMode": mgn_mode,
        }
        if pos_side:
            payload["posSide"] = pos_side
        if ccy:
            payload["ccy"] = ccy
        if auto_cxl is not None:
            payload["autoCxl"] = "true" if auto_cxl else "false"
        r = self._request("POST", "/api/v5/trade/close-position", data=payload, auth=True)
        return r.get("data", [])