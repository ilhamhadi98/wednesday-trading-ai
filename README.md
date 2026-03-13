# Auto Trading Deep Learning (MT5, H1 + H4, Multi-Symbol)

Project ini berisi bot trading otomatis berbasis deep learning (LSTM) yang:

- Ambil data historis dari MetaTrader 5 pada timeframe `H1` dan `H4`
- Melakukan feature engineering multi-timeframe
- Training model untuk prediksi arah candle berikutnya
- Backtesting strategi (termasuk mode `walk-forward` untuk self-learning)
- Scan peluang lintas banyak simbol (bukan hanya 1 simbol)
- Batch backtest multi-simbol untuk ranking market terbaik
- Menjalankan live trading pada akun demo MT5 dengan SL/TP + risk management

## 1) Persiapan

1. Install Python 3.10+.
2. Install MetaTrader 5 terminal dan login akun demo.
3. Aktifkan `Algo Trading` di MT5.
4. Install dependency:

```bash
pip install -r requirements.txt
```

5. Copy `.env.example` jadi `.env` lalu isi kredensial demo MT5.

## 2) Training Model

```bash
python train_model.py
```

Output model:

- `outputs/model.keras`
- `outputs/scaler.pkl`
- `outputs/model_meta.json`
- `outputs/train_metrics.json`

## 3) Backtest

### Mode rekomendasi (self-learning walk-forward)

```bash
python run_backtest.py --mode walk_forward
```

### Mode train-test biasa

```bash
python run_backtest.py --mode train_test
```

Output backtest:

- `outputs/backtest_report.json`
- `outputs/backtest_trades.csv`
- `outputs/backtest_summary.txt`

### Backtest banyak simbol sekaligus

```bash
python run_multi_backtest.py --max-symbols 12 --epochs 3 --top 10
```

Atau tentukan simbol sendiri:

```bash
python run_multi_backtest.py --symbols EURUSD,GBPUSD,USDJPY,XAUUSD --epochs 3
```

Output:

- `outputs/multi_backtest_results.csv`
- `outputs/multi_backtest_results.json`
- `outputs/multi_backtest_summary.json`

## 4) Scan Peluang Semua Simbol

Untuk mencari peluang pasar secara luas:

```bash
python scan_opportunities.py --top 15
```

Atau simbol tertentu:

```bash
python scan_opportunities.py --symbols EURUSD,GBPUSD,XAUUSD --top 10
```

Output:

- `outputs/market_opportunities.csv`
- `outputs/market_opportunities.json`
- `outputs/market_scan_summary.json`

Default scanner sekarang `pair-only` (fokus pair tradable), bukan semua simbol acak.

## 5) Backtest Portfolio Pair (3 Bulan Terakhir)

Sesuai skenario:
- baca semua pair tradable,
- cari sinyal BUY/SELL,
- jalankan simulasi dengan batas maksimal 3 pair aktif bersamaan.

Perintah:

```bash
python backtest_pairs_3m.py --months 3
```

Output:

- `outputs/portfolio_backtest_3m_report.json`
- `outputs/portfolio_backtest_3m_trades.csv`
- `outputs/portfolio_backtest_3m_equity.csv`
- `outputs/portfolio_backtest_3m_symbol_stats.csv`
- `outputs/portfolio_backtest_3m_skipped_pairs.csv`

## 6) Laporan HTML (Grafik + Riwayat Trade)

Untuk membuat laporan visual HTML dari hasil backtest:

```bash
python generate_html_report.py --title "Portfolio Backtest Report (3 Months)"
```

Output:

- `outputs/portfolio_backtest_3m_report.html`

## 7) Live Trading Demo

Pastikan model sudah ada (`train_model.py` sudah dijalankan), lalu:

```bash
python live_demo.py
```

Atau mode multi-simbol (eksekusi top peluang):

```bash
python live_multi_demo.py --top 3
```

Perintah di atas default `dry-run` (hanya analisa, tidak kirim order).  
Untuk benar-benar kirim order ke akun demo:

```bash
python live_multi_demo.py --top 3 --execute
```

Behavior live:

- Prediksi tiap bar `H1` baru dengan input fitur `H1 + H4`
- Sinyal:
  - `prob >= PRED_BUY_THRESHOLD` => BUY
  - `prob <= PRED_SELL_THRESHOLD` => SELL
  - selain itu => HOLD
- Retraining otomatis tiap `RETRAIN_EVERY_BARS` bar (self-learning)
- Hanya untuk akun demo (disarankan)

## 8) Konfigurasi Multi-Symbol

Parameter di `.env`:

- `SYMBOL_DISCOVERY_MODE=VISIBLE`:
  - `VISIBLE` = scan semua simbol yang terlihat di Market Watch
  - `ALL` = scan semua simbol tradable di server
  - `LIST` = scan daftar pada `SYMBOLS`
- `SYMBOLS=EURUSD,GBPUSD,...` = daftar manual jika mode `LIST`
- `SCAN_MAX_SYMBOLS=50` = limit jumlah simbol per scan
- `MIN_BARS_REQUIRED=800` = minimum data valid per simbol
- `MAX_SIGNAL_BAR_AGE_HOURS=168` = maksimal umur candle terakhir agar sinyal dianggap fresh
- `MAX_ACTIVE_PAIRS=3` = batas pair aktif bersamaan (live & backtest portfolio)
- `PAIR_FILTER_MODE=ALL|MAJORS|MAJORS_MINORS|CUSTOM` = mode filter pair
- `PAIR_INCLUDE=EURUSD,GBPUSD` = pair wajib masuk
- `PAIR_EXCLUDE=XAUUSD` = pair yang dibuang
- `PAIR_MAX_VOLUME_MIN=0.2` = buang pair dengan lot minimum terlalu besar
- `RISK_PROFILE=KECIL|SEDANG|BESAR` = agresivitas trading
  - `KECIL`: threshold lebih ketat, spread filter lebih ketat, pair heavy (lot minimum besar) difilter
  - `SEDANG`: default
  - `BESAR`: lebih agresif, threshold lebih longgar
- `BACKTEST_FAST_TF=H1`, `BACKTEST_SLOW_TF=H4` = timeframe untuk train/backtest
- `LIVE_FAST_TF=M5`, `LIVE_SLOW_TF=M15` = timeframe untuk live scan/eksekusi

## 9) Catatan Penting

- Secara teknis bisa backtest di `H1/H4` lalu live di `M5/M15`, tetapi validasinya tidak apple-to-apple.  
  Untuk hasil paling aktual, backtest sebaiknya pakai timeframe yang sama dengan live.

- Ini adalah sistem riset/edukasi, bukan jaminan profit.
- Hasil backtest bisa berbeda dengan forward/live karena spread, slippage, latency.
- Mulai dari lot kecil pada akun demo dan evaluasi minimal beberapa minggu sebelum keputusan lanjutan.
