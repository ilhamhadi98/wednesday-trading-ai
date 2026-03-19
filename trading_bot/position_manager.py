import MetaTrader5 as mt5
from .config import Settings
import pandas as pd

def _round_price(price: float, digits: int) -> float:
    return round(float(price), digits)

def manage_open_positions(client, df_signals: pd.DataFrame, settings: Settings):
    """
    Manajemen posisi M15 scalping:
    1. Trailing Stop Loss
    2. Break-Even
    3. AI Dynamic Exit (jika model berubah pikiran ekstrem)
    """
    positions = client.positions_all()
    if not positions:
        return

    # Ambil latest signal probability per symbol
    ai_view = {}
    for _, row in df_signals.iterrows():
        if row["status"] == "OK":
            ai_view[row["symbol"]] = {"signal": row["signal"], "prob": row["probability"]}

    for pos in positions:
        if pos.magic != settings.magic_number:
            continue
        
        sym_info = client.symbol_info(pos.symbol)
        spec = client.symbol_spec(pos.symbol)
        tick = client.latest_tick(pos.symbol)
        if not sym_info or not tick:
            continue
            
        point = sym_info.point
        pip_size = point * 10 if sym_info.digits in [3,5] else point
        
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        entry = pos.price_open
        current_sl = pos.sl
        current_tp = pos.tp
        current_price = tick.bid if is_buy else tick.ask
        
        profit_pips = (current_price - entry) / pip_size if is_buy else (entry - current_price) / pip_size
        
        needs_update = False
        new_sl = current_sl

        # -- 3. AI DYNAMIC EXIT (Reversal Cepat) --
        # Jika buy, tapi AI sekarang sangat yakin SELL (prob < 0.40) -> CLOSE!
        view = ai_view.get(pos.symbol)
        if view:
            close_request = None
            if is_buy and view["signal"] == "SELL" and view["prob"] < 0.40:
                print(f"[EXIT] {pos.symbol} DYNAMIC EXIT (BUY ditutup krn sinyal berubah ke SELL, prob={view['prob']:.4f})", flush=True)
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL, "position": pos.ticket, "price": current_price,
                    "deviation": settings.deviation, "magic": settings.magic_number,
                    "comment": "ai-dynamic-exit", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
                }
            elif not is_buy and view["signal"] == "BUY" and view["prob"] > 0.60:
                print(f"[EXIT] {pos.symbol} DYNAMIC EXIT (SELL ditutup krn sinyal berubah ke BUY, prob={view['prob']:.4f})", flush=True)
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_BUY, "position": pos.ticket, "price": current_price,
                    "deviation": settings.deviation, "magic": settings.magic_number,
                    "comment": "ai-dynamic-exit", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
                }
            
            if close_request:
                # Coba 3 macam filling mode agar pasti tereksekusi
                for mode in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
                    close_request["type_filling"] = mode
                    res = mt5.order_send(close_request)
                    if res and res.retcode != 10030:
                        break
                continue

        # -- 2. BREAK-EVEN (Jika profit > 5 pips, ubah SL ke Entry) --
        be_pips = 5.0
        if profit_pips > be_pips:
            if is_buy and current_sl < entry:
                new_sl = entry + (1 * pip_size)  # Plus 1 pip untuk komisi/spread
                needs_update = True
            elif not is_buy and (current_sl > entry or current_sl == 0):
                new_sl = entry - (1 * pip_size)
                needs_update = True
                
        # -- 1. TRAILING STOP --
        trail_pips = float(settings.trailing_stop_pips) if settings.use_trailing_stop else 5.0
        if profit_pips > trail_pips:
            if is_buy:
                t_sl = current_price - (trail_pips * pip_size)
                if t_sl > new_sl:
                    new_sl = t_sl
                    needs_update = True
            else:
                t_sl = current_price + (trail_pips * pip_size)
                if t_sl < new_sl or new_sl == 0:
                    new_sl = t_sl
                    needs_update = True

        if needs_update and new_sl != current_sl:
            # Proteksi batas broker
            stop_level = max(sym_info.trade_stops_level * point, 1 * pip_size)
            if is_buy and (current_price - new_sl) < stop_level:
                new_sl = current_price - stop_level
            elif not is_buy and (new_sl - current_price) < stop_level:
                new_sl = current_price + stop_level

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "sl": _round_price(new_sl, sym_info.digits),
                "tp": current_tp,
                "magic": settings.magic_number
            }
            res = mt5.order_send(request)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[MANAGE] {pos.symbol} SL diupdate ke {_round_price(new_sl, sym_info.digits)} (Profit berjalan: {profit_pips:.1f} pips)", flush=True)
