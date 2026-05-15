#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Oraculum 제5장 v5.

핵심:
1) Gate trade_history_gate.txt 를 16줄 거래 블록으로 파싱한다.
2) TXT 시간은 Australia/Brisbane 기준으로 해석해 UTC ms로 변환한다.
3) 각 포지션 보유기간과 겹치는 BTCUSDT 5분봉을 매칭한다.
4) 각 5분봉의 close/open 가격 기준 평가손익률을 계산한다.
5) 거래별 최악 평가손익률(MAE = 보유기간 중 5분 단위 평가손익률의 최소값) 분포와,
   전체 5분 단위 평가손익률 샘플 분포를 둘 다 생성한다.
6) 실제 청산이 없어도 MAE 분포의 -100% 이하 왼쪽 꼬리를 KDE로 적분해 간접 청산 위험을 산출한다.

출력:
- bullshit/ch5_distribution_data.js
- bullshit/ch5_trade_mae_clean.csv
- bullshit/ch5_5m_roi_samples.csv
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde, norm
except Exception:  # pragma: no cover
    gaussian_kde = None
    norm = None

BNE = ZoneInfo("Australia/Brisbane")
UTC = ZoneInfo("UTC")
BIN_MS = 5 * 60 * 1000

ROOT = Path(__file__).resolve().parent.parent
BULL = ROOT / "bullshit"
DATA = ROOT / "data"
TRADE_CANDIDATES = [DATA / "trade_history_gate.txt", BULL / "trade_history_gate.txt", ROOT / "trade_history_gate.txt"]
CANDLE_CANDIDATES = [DATA / "btcusdt_5m.csv", BULL / "btcusdt_5m.csv", ROOT / "btcusdt_5m.csv"]
OUT_JS = BULL / "ch5_distribution_data.js"
OUT_JSON = BULL / "ch5_distribution_data.json"
OUT_TRADES = BULL / "ch5_trade_mae_clean.csv"
OUT_SAMPLES = BULL / "ch5_5m_roi_samples.csv"


def first_existing(paths: Iterable[Path], label: str) -> Path:
    for p in paths:
        if p.is_file():
            return p
    joined = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(f"missing {label}; tried:\n{joined}")


def parse_num(text: str) -> float:
    s = str(text).replace(",", "").strip()
    s = re.sub(r"[^0-9+\-.]", "", s)
    if s in {"", "+", "-", "."}:
        return float("nan")
    return float(s)


def parse_bne_ms(date_text: str, time_text: str) -> int:
    dt = datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=BNE).astimezone(UTC).timestamp() * 1000)


def ms_to_utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def ms_to_bne_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).astimezone(BNE).strftime("%Y-%m-%d %H:%M:%S Brisbane")


def load_gate_blocks(path: Path) -> pd.DataFrame:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    try:
        start = lines.index("BTCUSDT")
    except ValueError as exc:
        raise ValueError("trade_history_gate.txt에서 BTCUSDT 거래 블록 시작점을 찾지 못했습니다.") from exc

    rows: List[Dict[str, Any]] = []
    block_size = 16
    for offset, i in enumerate(range(start, len(lines), block_size), start=1):
        block = lines[i : i + block_size]
        if len(block) < block_size:
            raise ValueError(f"마지막 거래 블록 길이가 부족합니다: offset={offset}, len={len(block)}")
        if block[0] != "BTCUSDT":
            raise ValueError(f"거래 블록 시작이 BTCUSDT가 아닙니다: offset={offset}, head={block[:5]}")

        leverage = parse_num(block[2])
        side = block[3].strip().lower()
        if side not in {"long", "short"}:
            raise ValueError(f"side는 Long/Short 여야 합니다: offset={offset}, side={block[3]}")
        entry_ms = parse_bne_ms(block[5], block[6])
        exit_ms = parse_bne_ms(block[13], block[14])
        if exit_ms <= entry_ms:
            raise ValueError(f"exit_time <= entry_time: offset={offset}")

        rows.append(
            {
                "trade_id": offset,
                "symbol": block[0],
                "margin_mode": block[1],
                "leverage": leverage,
                "side": side,
                "status": block[4],
                "entry_time_bne": f"{block[5]} {block[6]}",
                "exit_time_bne": f"{block[13]} {block[14]}",
                "entry_time_utc_ms": entry_ms,
                "exit_time_utc_ms": exit_ms,
                "entry_time_utc": ms_to_utc_iso(entry_ms),
                "exit_time_utc": ms_to_utc_iso(exit_ms),
                "entry_price": parse_num(block[7]),
                "avg_exit_price": parse_num(block[8]),
                "cumulative_closed_usdt": parse_num(block[9]),
                "peak_position_usdt": parse_num(block[10]),
                "closed_pnl_usdt": parse_num(block[11]),
                "closed_roi_pct": parse_num(block[12]),
                "action": block[15],
            }
        )
    return pd.DataFrame(rows)


def load_5m_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"5분봉 CSV 필수 컬럼 누락: {sorted(missing)}")
    for col in ["open_time_ms", "close_time_ms", "open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time_ms", "close_time_ms", "open", "high", "low", "close"])
    df = df.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
    return df


def overlap_candles(candles: pd.DataFrame, entry_ms: int, exit_ms: int) -> pd.DataFrame:
    # 캔들 [open_time, close_time] 이 포지션 기간 (entry, exit) 과 겹치면 포함.
    return candles[(candles["open_time_ms"] < exit_ms) & (candles["close_time_ms"] > entry_ms)].copy()


def leveraged_roi_pct(side: str, entry_price: float, price: float, leverage: float) -> float:
    if side == "long":
        return (price - entry_price) / entry_price * leverage * 100.0
    if side == "short":
        return (entry_price - price) / entry_price * leverage * 100.0
    raise ValueError(f"unknown side: {side}")


def make_edges(values: np.ndarray, lo: float = -100.0, hi: float = 100.0, step: float = 10.0) -> np.ndarray:
    # 청산선(-100%)이 항상 보이도록 기본은 -100~100. 값이 범위를 넘으면 10% 단위로 확장.
    finite = values[np.isfinite(values)]
    if finite.size:
        lo = min(lo, math.floor(float(finite.min()) / step) * step)
        hi = max(hi, math.ceil(float(finite.max()) / step) * step)
    return np.arange(lo, hi + step, step, dtype=float)


def hist_block(values: Iterable[float], edges: Optional[np.ndarray] = None, kde_overlay: bool = True) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if edges is None:
        edges = make_edges(arr)
    labels = [f"{int(edges[i])}%~{int(edges[i + 1])}%" for i in range(len(edges) - 1)]
    if arr.size:
        # 히스토그램 시각화에서는 바깥 값을 양끝 bin에 포함시켜 총합이 보존되게 한다.
        clipped = np.clip(arr, edges[0] + 1e-9, edges[-1] - 1e-9)
        counts = np.histogram(clipped, bins=edges)[0].astype(int).tolist()
    else:
        counts = [0] * (len(edges) - 1)

    overlay = [0.0] * len(counts)
    show = False
    if kde_overlay and arr.size >= 2 and gaussian_kde is not None:
        try:
            kde = gaussian_kde(arr, bw_method="scott")
            overlay = []
            for i in range(len(edges) - 1):
                p = float(kde.integrate_box_1d(float(edges[i]), float(edges[i + 1])))
                overlay.append(round(arr.size * p, 4))
            show = True
        except Exception:
            pass
    return {
        "counts": counts,
        "binLabels": labels,
        "normalCounts": overlay,  # 기존 index 호환용 이름. 실제로는 KDE expected counts.
        "showNormalOverlay": show,
    }


def stats_block(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def left_tail_prob(values: Iterable[float], threshold: float = -100.0) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    out: Dict[str, Any] = {
        "thresholdPct": threshold,
        "observedCountAtOrBelowThreshold": int(np.sum(arr <= threshold)) if arr.size else 0,
        "observedFrequency": float(np.mean(arr <= threshold)) if arr.size else None,
        "kdeProbability": None,
        "normalProbability": None,
    }
    if arr.size >= 2:
        if gaussian_kde is not None:
            try:
                kde = gaussian_kde(arr, bw_method="scott")
                out["kdeProbability"] = float(max(0.0, min(1.0, kde.integrate_box_1d(float("-inf"), threshold))))
            except Exception:
                out["kdeProbability"] = None
        if norm is not None:
            sd = float(np.std(arr, ddof=1))
            if sd > 0:
                out["normalProbability"] = float(norm.cdf((threshold - float(np.mean(arr))) / sd))
    return out


def build() -> Dict[str, Any]:
    BULL.mkdir(parents=True, exist_ok=True)
    trade_path = first_existing(TRADE_CANDIDATES, "trade_history_gate.txt")
    candle_path = first_existing(CANDLE_CANDIDATES, "btcusdt_5m.csv")
    trades = load_gate_blocks(trade_path)
    candles = load_5m_candles(candle_path)

    sample_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

    for _, tr in trades.iterrows():
        sub = overlap_candles(candles, int(tr.entry_time_utc_ms), int(tr.exit_time_utc_ms))
        close_vals: List[float] = []
        open_vals: List[float] = []
        for _, c in sub.iterrows():
            close_roi = leveraged_roi_pct(tr.side, float(tr.entry_price), float(c.close), float(tr.leverage))
            open_roi = leveraged_roi_pct(tr.side, float(tr.entry_price), float(c.open), float(tr.leverage))
            close_vals.append(close_roi)
            open_vals.append(open_roi)
            sample_rows.append(
                {
                    "trade_id": int(tr.trade_id),
                    "symbol": tr.symbol,
                    "side": tr.side,
                    "leverage": float(tr.leverage),
                    "candle_open_time_utc": ms_to_utc_iso(int(c.open_time_ms)),
                    "candle_close_time_utc": ms_to_utc_iso(int(c.close_time_ms)),
                    "candle_close_time_bne": ms_to_bne_iso(int(c.close_time_ms)),
                    "open": float(c.open),
                    "close": float(c.close),
                    "roi_open_pct": float(open_roi),
                    "roi_close_pct": float(close_roi),
                }
            )

        if close_vals:
            close_arr = np.asarray(close_vals, dtype=float)
            open_arr = np.asarray(open_vals, dtype=float)
            worst_idx = int(np.argmin(close_arr))
            best_idx = int(np.argmax(close_arr))
            worst_sample = sub.iloc[worst_idx]
            best_sample = sub.iloc[best_idx]
            coverage_status = "ok"
            mae_close = float(close_arr[worst_idx])
            max_close = float(close_arr[best_idx])
            mae_open = float(np.min(open_arr))
            n_samples = int(len(close_arr))
            worst_time_utc = ms_to_utc_iso(int(worst_sample.close_time_ms))
            worst_time_bne = ms_to_bne_iso(int(worst_sample.close_time_ms))
            best_time_utc = ms_to_utc_iso(int(best_sample.close_time_ms))
            worst_price_close = float(worst_sample.close)
        else:
            coverage_status = "no_overlap"
            mae_close = max_close = mae_open = float("nan")
            n_samples = 0
            worst_time_utc = worst_time_bne = best_time_utc = ""
            worst_price_close = float("nan")

        trade_rows.append(
            {
                **{k: tr[k] for k in trades.columns},
                "matched_5m_candles": n_samples,
                "coverage_status": coverage_status,
                "mae_close_pct": mae_close,
                "mae_open_pct": mae_open,
                "max_favorable_close_pct": max_close,
                "worst_close_price": worst_price_close,
                "worst_time_utc": worst_time_utc,
                "worst_time_bne": worst_time_bne,
                "best_time_utc": best_time_utc,
                "crossed_liquidation_line_close": bool(np.isfinite(mae_close) and mae_close <= -100.0),
            }
        )

    trade_clean = pd.DataFrame(trade_rows)
    samples = pd.DataFrame(sample_rows)
    trade_clean.to_csv(OUT_TRADES, index=False)
    samples.to_csv(OUT_SAMPLES, index=False)

    closed = trade_clean["closed_roi_pct"].to_numpy(dtype=float)
    sample_close = samples["roi_close_pct"].to_numpy(dtype=float) if not samples.empty else np.asarray([], dtype=float)
    sample_open = samples["roi_open_pct"].to_numpy(dtype=float) if not samples.empty else np.asarray([], dtype=float)
    mae_close = trade_clean["mae_close_pct"].to_numpy(dtype=float)
    mae_open = trade_clean["mae_open_pct"].to_numpy(dtype=float)

    common_edges = np.arange(-100, 110, 10, dtype=float)
    sample_edges = make_edges(sample_close, -100, 100, 10)
    mae_edges = make_edges(mae_close, -100, 100, 10)

    wins = int(np.sum(closed > 0))
    losses = int(np.sum(closed < 0))
    flats = int(np.sum(closed == 0))
    mae_tail = left_tail_prob(mae_close, -100.0)
    sample_tail = left_tail_prob(sample_close, -100.0)

    def round_trades_for_js(df: pd.DataFrame) -> List[Dict[str, Any]]:
        keep = [
            "trade_id", "symbol", "side", "leverage", "entry_time_bne", "exit_time_bne", "entry_price",
            "avg_exit_price", "closed_pnl_usdt", "closed_roi_pct", "matched_5m_candles", "mae_close_pct",
            "mae_open_pct", "max_favorable_close_pct", "worst_close_price", "worst_time_bne", "coverage_status",
        ]
        out = []
        for rec in df[keep].to_dict(orient="records"):
            clean = {}
            for k, v in rec.items():
                if isinstance(v, float):
                    clean[k] = None if not math.isfinite(v) else round(v, 8)
                else:
                    clean[k] = v
            out.append(clean)
        return out

    payload: Dict[str, Any] = {
        "schemaVersion": "ch5-v5-liquidation-shadow-results-below-charts",
        "nTrades": int(len(trade_clean)),
        "symbol": "BTCUSDT",
        "tradeTimeZone": "Australia/Brisbane",
        "candleTimeZone": "UTC",
        "candleTimeframe": "5m",
        "samplePriceField": "close",
        "liquidationLinePct": -100.0,
        "sourceFiles": {
            "trades": str(trade_path.relative_to(ROOT)) if trade_path.is_relative_to(ROOT) else str(trade_path),
            "candles5m": str(candle_path.relative_to(ROOT)) if candle_path.is_relative_to(ROOT) else str(candle_path),
        },
        "closedRoiPct": {
            "histogram": hist_block(closed, common_edges),
            "wins": wins,
            "losses": losses,
            "flats": flats,
            "stats": stats_block(closed),
            "values": [round(float(x), 8) for x in closed if np.isfinite(x)],
            "noteKo": "거래가 최종적으로 얼마에 끝났는지 보여주는 결과표입니다. 보유 중 위험은 위 두 분포에서 확인합니다.",
        },
        "roi5mHoldingSamplesClosePct": {
            "histogram": hist_block(sample_close, sample_edges),
            "stats": stats_block(sample_close),
            "sampleCount": int(sample_close.size),
            "samplePriceField": "close",
            "tailAtMinus100": sample_tail,
            "noteKo": "포지션이 열려 있는 동안 5분마다 평가손익률을 찍어 모두 모은 분포입니다. 전략이 평소 어떤 위험 구간에 머물렀는지 보여주는 핵심 지표입니다.",
        },
        "roi5mHoldingSamplesOpenPct": {
            "histogram": hist_block(sample_open, sample_edges),
            "stats": stats_block(sample_open),
            "sampleCount": int(sample_open.size),
            "samplePriceField": "open",
            "tailAtMinus100": left_tail_prob(sample_open, -100.0),
            "noteKo": "각 5분봉 open 가격 기준 평가손익률 샘플 분포입니다. 기본 표시값은 close 기준입니다.",
        },
        "mae5mApproxWorstRoiPct": {
            "bankruptcyPct": mae_tail.get("kdeProbability"),  # 기존 index 호환용
            "estimatedLiquidationProbabilityKde": mae_tail.get("kdeProbability"),
            "estimatedLiquidationProbabilityNormal": mae_tail.get("normalProbability"),
            "observedLiquidationCount": mae_tail.get("observedCountAtOrBelowThreshold"),
            "histogram": hist_block(mae_close, mae_edges),
            "stats": stats_block(mae_close),
            "samplePriceField": "close",
            "values": [round(float(x), 8) for x in mae_close if np.isfinite(x)],
            "excludedNoOverlap": int(np.sum(trade_clean["coverage_status"] != "ok")),
            "noteKo": "거래 하나마다 보유 중 가장 깊게 밀린 손익률 하나를 기록한 분포입니다. 이 분포의 −100% 이하 꼬리 면적으로 관측되지 않은 청산 가능성을 간접 추정합니다.",
        },
        "mae5mOpenWorstRoiPct": {
            "histogram": hist_block(mae_open, mae_edges),
            "stats": stats_block(mae_open),
            "samplePriceField": "open",
            "tailAtMinus100": left_tail_prob(mae_open, -100.0),
            "noteKo": "거래별 5분봉 open 기준 최악 평가손익률 분포입니다. 기본 표시는 close 기준입니다.",
        },
        "summary": {
            "nTrades": int(len(trade_clean)),
            "n5mSamplesClose": int(sample_close.size),
            "wins": wins,
            "losses": losses,
            "flats": flats,
            "worstObservedTradeMaeClosePct": float(np.nanmin(mae_close)) if mae_close.size else None,
            "worstObserved5mSampleClosePct": float(np.nanmin(sample_close)) if sample_close.size else None,
            "estimatedTradeLiquidationProbabilityKde": mae_tail.get("kdeProbability"),
            "estimatedSampleTailProbabilityKde": sample_tail.get("kdeProbability"),
        },
        "trades": round_trades_for_js(trade_clean),
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_JS.write_text("window.CH5_DISTRIBUTION_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    return payload


def main() -> None:
    payload = build()
    s = payload["summary"]
    print(f"wrote {OUT_JS}")
    print(f"wrote {OUT_TRADES}")
    print(f"wrote {OUT_SAMPLES}")
    print(f"trades={s['nTrades']}, 5m close samples={s['n5mSamplesClose']}")
    print(f"worst trade MAE close={s['worstObservedTradeMaeClosePct']:.4f}%")
    p = s.get("estimatedTradeLiquidationProbabilityKde")
    if p is not None:
        print(f"KDE P(trade MAE <= -100%) ≈ {p * 100:.6f}%")


if __name__ == "__main__":
    main()
