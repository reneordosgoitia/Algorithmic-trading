#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, time
from binance.client import Client
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

# ───────── INIT ───────── #

c    = Client("YOUR_KEY", "YOUR_SECRET")
I    = Client.KLINE_INTERVAL_15MINUTE
tz   = timezone(timedelta(hours=-5))

start = int((datetime(2024, 8, 15, tzinfo=tz) - timedelta(minutes=400*15))
            .astimezone(timezone.utc).timestamp() * 1e3)
end   = int(datetime(2024, 12, 7, 23, 59, 59, tzinfo=tz)
            .astimezone(timezone.utc).timestamp() * 1e3)

COLS = ['open_time','open','high','low','close','volume','close_time',
        'qav','nt','tbv','tqv','ig']

# ───────── FETCH ───────── #

def fetch(sym):
    raw, cur = [], start
    while cur < end:
        # use keyword args!
        bars = c.futures_klines(
            symbol=sym,
            interval=I,
            startTime=cur,
            endTime=end,
            limit=500
        )
        if not bars: 
            break
        raw += bars
        cur  = int(bars[-1][0]) + 1

    df = pd.DataFrame(raw, columns=COLS)[['open_time','close','volume']]
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df[['close','volume']] = df[['close','volume']].apply(pd.to_numeric)
    return df.set_index('open_time')

# ───────── MAIN ───────── #

t0 = time.perf_counter()
with ThreadPoolExecutor() as ex:
    # get list of results, then unpack
    btc_data, eth_data = list(ex.map(fetch, ["BTCUSDT", "ETHUSDT"]))


# In[2]:


import numpy as np
btc_data['log_return'] = np.log(btc_data['close'] / btc_data['close'].shift(1))
eth_data['log_return'] = np.log(eth_data['close'] / eth_data['close'].shift(1))

# Drop missing values
btc_data.dropna(inplace=True)
eth_data.dropna(inplace=True)


# In[3]:


import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from binance.client import Client
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans

# ───────── CONFIGURATION ───────── #
# 1. Set up your API credentials and client
API_KEY    = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"
client     = Client(API_KEY, API_SECRET)

# 2. Define the 15-minute interval and your local timezone (UTC-5)
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
TZ       = timezone(timedelta(hours=-5))

# 3. Define the date range (in UTC-5), then convert to UTC-milliseconds
start_local = datetime(2024,  8, 15, tzinfo=TZ) - timedelta(minutes=400*15)
end_local   = datetime(2024, 12,  7, 23, 59, 59, tzinfo=TZ)
START_MS = int(start_local.astimezone(timezone.utc).timestamp() * 1e3)
END_MS   = int(end_local  .astimezone(timezone.utc).timestamp() * 1e3)

# 4. Column names for the raw kline data
COLS = [
    'open_time','open','high','low','close','volume',
    'close_time','quote_asset_volume','num_trades',
    'taker_buy_base_vol','taker_buy_quote_vol','ignore'
]

# ───────── FETCH FUNCTION ───────── #
def fetch(symbol: str) -> pd.DataFrame:
    """
    Fetches 15m futures klines for `symbol` from Binance in batches of 500,
    returns a DataFrame indexed by datetime with 'close' and 'volume'.
    """
    raw_data = []
    cursor   = START_MS

    # 5. Loop until we reach the end timestamp
    while cursor < END_MS:
        batch = client.futures_klines(
            symbol=symbol,
            interval=INTERVAL,
            startTime=cursor,
            endTime=END_MS,
            limit=500
        )
        if not batch:
            break

        raw_data.extend(batch)
        # Move cursor to just after the last bar fetched
        cursor = int(batch[-1][0]) + 1

    # 6. Convert the raw list into a single DataFrame
    df = pd.DataFrame(raw_data, columns=COLS)
    df = df[['open_time', 'close', 'volume']].copy()
    df['open_time']  = pd.to_datetime(df['open_time'], unit='ms')
    df['close']      = pd.to_numeric(df['close'], errors='coerce')
    df['volume']     = pd.to_numeric(df['volume'], errors='coerce')
    df.set_index('open_time', inplace=True)

    return df

# ───────── MAIN SCRIPT ───────── #
if __name__ == "__main__":
    # 7. Fetch BTC and ETH data **in parallel** to save time
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        btc_data, eth_data = list(executor.map(fetch, ["BTCUSDT", "ETHUSDT"]))
    elapsed = time.perf_counter() - t0
    print(f"Fetched {btc_data.shape[0]} BTC + {eth_data.shape[0]} ETH bars in {elapsed:.2f}s")

    # 8. Calculate log-returns and drop the first NaN
    for df in (btc_data, eth_data):
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)

    # 9. Create rolling windows of size h1, step = h1-h2
    h1, h2 = 35, 28
    step   = h1 - h2

    def make_features(series: pd.Series) -> np.ndarray:
        """
        Sliding window over `series` to compute mean & std for each window.
        """
        means, stds = [], []
        for start in range(0, len(series) - h1 + 1, step):
            window = series.iloc[start : start + h1]
            means.append(window.mean())
            stds.append(window.std())
        # Stack as [std, mean]
        return np.column_stack((stds, means))

    btc_features = make_features(btc_data['log_return'])
    eth_features = make_features(eth_data['log_return'])

    # 10. Run **3-cluster** KMeans on both feature sets
    kmeans = KMeans(n_clusters=3, random_state=42)
    btc_labels = kmeans.fit_predict(btc_features)
    eth_labels = kmeans.fit_predict(eth_features)

    # 11. Plot results side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

    for ax, feats, labels, title in [
        (ax1, btc_features, btc_labels, "BTC Mean–Variance Clusters"),
        (ax2, eth_features, eth_labels, "ETH Mean–Variance Clusters"),
    ]:
        for cluster_id in range(3):
            mask = (labels == cluster_id)
            ax.scatter(feats[mask, 0], feats[mask, 1], alpha=0.6, label=f"Cluster {cluster_id}")
        ax.set_xlabel("Std Dev (Volatility)")
        ax.set_ylabel("Mean Return")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# In[ ]:




