"""
Smart-E Robo — 後端 API Server
部署於 Render，處理投資人自選 ETF 的即時 MVO 計算

Endpoints:
  GET  /health       確認 server 存活
  POST /api/mvo      接收標的清單，回傳 MVO 三目標結果 + 回測
"""

import hashlib
import math
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # 允許 GitHub Pages 跨域請求


# ── 工具函式 ──────────────────────────────────────────────────

def download_single(ticker: str, start: str, end: str):
    """逐檔下載，含字母後綴嘗試 .TW / .TWO"""
    suffixes = [".TW", ".TWO"] if any(c.isalpha() for c in ticker) else [".TW"]
    for sfx in suffixes:
        try:
            r = yf.download(f"{ticker}{sfx}", start=start, end=end,
                            auto_adjust=True, progress=False)
            if r is None or r.empty or len(r) < 30:
                continue
            if isinstance(r.columns, pd.MultiIndex):
                close = r["Close"].iloc[:, 0]
            else:
                close = r["Close"] if "Close" in r.columns else r.iloc[:, 0]
            close.name = ticker
            return close
        except Exception:
            continue
    return None


def mvo_optimize(price_dict: dict, rf: float = 0.015, n_sim: int = 5000):
    """
    Monte Carlo MVO，固定 seed 確保重現性
    回傳三個目標：最大夏普 / 最大報酬 / 最小波動
    """
    if len(price_dict) < 2:
        return None

    prices = pd.DataFrame(price_dict).ffill().bfill().dropna()
    if len(prices) < 60:
        return None

    rets  = prices.pct_change().dropna()
    valid = rets.columns.tolist()
    rets  = rets[valid]
    mu    = rets.mean() * 252
    cov   = rets.cov()  * 252
    n     = len(valid)

    # 固定 seed（hashlib md5，跨環境一致）
    ticker_str = ','.join(sorted(valid))
    seed       = int(hashlib.md5(ticker_str.encode()).hexdigest(), 16) % (2 ** 31)
    rng        = np.random.default_rng(seed)

    best = {
        'sharpe': {'val': -np.inf, 'w': None},
        'return': {'val': -np.inf, 'w': None},
        'minvol': {'val':  np.inf, 'w': None},
    }
    frontier = []

    for _ in range(n_sim):
        w   = rng.dirichlet(np.ones(n))
        ret = float(w @ mu)
        vol = float(np.sqrt(np.maximum(w @ cov.values @ w, 0)))
        shp = (ret - rf) / vol if vol > 0 else 0
        frontier.append({
            'ret':    round(ret, 4),
            'vol':    round(vol, 4),
            'sharpe': round(shp, 4),
        })
        if shp > best['sharpe']['val']: best['sharpe'] = {'val': shp,  'w': w.copy()}
        if ret > best['return']['val']: best['return'] = {'val': ret,  'w': w.copy()}
        if vol < best['minvol']['val']: best['minvol'] = {'val': vol,  'w': w.copy()}

    def make_result(key: str) -> dict:
        w      = best[key]['w']
        wts    = {t: round(float(w[i]), 4) for i, t in enumerate(valid)}
        r      = float(w @ mu)
        v      = float(np.sqrt(np.maximum(w @ cov.values @ w, 0)))
        sharpe = (r - rf) / v if v > 0 else 0
        return {
            'tickers':         valid,
            'weights':         wts,
            'expected_return': round(r, 4),
            'volatility':      round(v, 4),
            'sharpe':          round(sharpe, 4),
        }

    return {
        'maxSharpe':     make_result('sharpe'),
        'maxReturn':     make_result('return'),
        'minVol':        make_result('minvol'),
        'frontier':      frontier[:2000],
        'valid_tickers': valid,
    }


def run_backtest(price_dict: dict, weights: dict, rf: float = 0.015) -> dict:
    """用給定的權重對 price_dict 做回測"""
    tickers = [t for t in weights if t in price_dict]
    if len(tickers) < 2:
        return {}

    prices = pd.DataFrame({t: price_dict[t] for t in tickers}).ffill().bfill()
    rets   = prices.pct_change().dropna()
    w_arr  = np.array([weights[t] for t in tickers])
    w_norm = w_arr / w_arr.sum()

    port = (rets.values * w_norm).sum(axis=1)
    cum  = (1 + port).cumprod()

    peak   = np.maximum.accumulate(cum)
    mdd    = float(((cum - peak) / peak).min())
    mean   = port.mean()
    std    = port.std()
    sharpe = float(mean / std * np.sqrt(252)) if std > 0 else 0

    return {
        'cumReturn': round(float(cum[-1] - 1), 4),
        'mdd':       round(mdd, 4),
        'sharpe':    round(sharpe, 4),
        'cumArr':    [round(float(v), 4) for v in cum[::max(1, len(cum)//300)]],
    }


def clean_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj


# ── Endpoints ─────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/mvo', methods=['POST'])
def api_mvo():
    """
    接收標的清單，下載價格，跑 MVO，回傳三目標結果與回測。

    Request body (JSON):
      {
        "tickers": ["0050", "00878", ...],   必填，2-15 檔
        "start":   "2023-01-01",             選填，預設近3年
        "end":     "2026-12-31",             選填
        "rf":      0.015                     選填，無風險利率
      }

    Response (JSON):
      {
        "maxSharpe": { weights, expected_return, volatility, sharpe, backtest },
        "maxReturn":  { ... },
        "minVol":     { ... },
        "frontier":   [ {ret, vol, sharpe}, ... ],
        "valid_tickers": [...],
        "failed_tickers": [...]
      }
    """
    data = request.get_json(silent=True)
    if not data or 'tickers' not in data:
        return jsonify({'error': 'tickers required'}), 400

    tickers = [str(t).strip().upper() for t in data['tickers']]
    start   = data.get('start', '2023-01-01')
    end     = data.get('end',   '2026-12-31')
    rf      = float(data.get('rf', 0.015))

    if len(tickers) < 2:
        return jsonify({'error': 'at least 2 tickers required'}), 400
    if len(tickers) > 15:
        return jsonify({'error': 'max 15 tickers'}), 400

    # 下載價格
    price_dict = {}
    failed     = []
    for t in tickers:
        px = download_single(t, start, end)
        if px is not None and len(px) >= 60:
            price_dict[t] = px
        else:
            failed.append(t)

    if len(price_dict) < 2:
        return jsonify({
            'error':          f'insufficient price data',
            'failed_tickers': failed,
        }), 400

    # MVO
    mvo = mvo_optimize(price_dict, rf=rf)
    if not mvo:
        return jsonify({'error': 'MVO optimization failed'}), 500

    # 各目標的回測
    for key in ['maxSharpe', 'maxReturn', 'minVol']:
        bt = run_backtest(price_dict, mvo[key]['weights'], rf=rf)
        mvo[key]['backtest'] = bt

    mvo['failed_tickers'] = failed

    return jsonify(clean_nan(mvo))



@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    import anthropic

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'invalid request'}), 400

    holdings = data.get('holdings', '')
    bt_s     = data.get('backtest', {})
    bt_e     = data.get('equal_weight', {})

    def fmt(v):
        try: return f"{float(v)*100:.1f}%"
        except: return str(v) if v else 'N/A'

    prompt = f"""你是一位專業的台灣ETF投資顧問，請針對以下投資組合提供繁體中文的分析建議：

【投資組合】
{holdings}

【回測績效（近3年）】
- 累積報酬：{fmt(bt_s.get('cumulative_return'))}
- 最大回檔：{fmt(bt_s.get('max_drawdown'))}
- 夏普比率：{bt_s.get('sharpe', 'N/A')}
- 對比100檔等權 累積報酬：{fmt(bt_e.get('cumulative_return'))}，夏普：{bt_e.get('sharpe', 'N/A')}

請從以下角度提供建議（每點2-3句，條列式）：
1. 組合特色與風險集中度分析
2. 費用率評估
3. 相對於等權市場的優劣勢
4. 潛在風險提示
5. 具體改善建議"""

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({'analysis': message.content[0].text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
