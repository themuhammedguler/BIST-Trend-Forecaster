# src/data_ingestion.py
import argparse
import os
from datetime import date

import pandas as pd

import config
import network


def fetch_data(start_date=None, end_date=None, output_path=None):
    start = start_date or config.START_DATE
    end = end_date or config.END_DATE
    out_path = output_path or config.DATA_PATH

    print(f"Veri çekme işlemi başladı ({start} - {end})... Bu işlem biraz sürebilir.")
    all_data = []

    for ticker in config.TICKERS:
        try:
            # Veriyi çek (Yahoo Finance sembol değişikliği varsa eşle)
            yahoo_ticker = getattr(config, "TICKER_YAHOO_MAP", {}).get(ticker, ticker)
            df = network.download_with_retry(
                yahoo_ticker, start=start, end=end, progress=False
            )

            # Multi-index düzeltmesi (yfinance yeni versiyonları için)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df["Ticker"] = ticker.replace(".IS", "")  # .IS uzantısını temizle
            df.reset_index(inplace=True)
            all_data.append(df)
            print(f"✅ {ticker} çekildi. ({len(df)} satır)")
        except Exception as e:
            print(f"❌ {ticker} hatası: {e}")

    # Tüm hisseleri alt alta birleştir
    final_df = pd.concat(all_data, ignore_index=True)

    # Sütun isimlerini düzenle
    final_df.columns = [c.lower() for c in final_df.columns]
    final_df.rename(columns={"date": "Date"}, inplace=True)

    # Kaydet
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    final_df.to_csv(out_path, index=False)
    print(f"\n🎉 Veri seti oluşturuldu: {out_path}")
    print(f"Toplam Satır Sayısı: {len(final_df)}")
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 30 veri seti indirme aracı")
    parser.add_argument(
        "--start-date", default=None, help="Başlangıç tarihi (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Bitiş tarihi (YYYY-MM-DD) veya 'today' / 'latest'",
    )
    parser.add_argument("--output-path", default=None, help="Çıktı CSV dosya yolu")
    args = parser.parse_args()

    end_d = args.end_date
    if end_d in ("today", "latest"):
        end_d = date.today().isoformat()

    fetch_data(start_date=args.start_date, end_date=end_d, output_path=args.output_path)
