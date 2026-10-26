# src/data_ingestion.py
import argparse
import os
import sys
from datetime import date

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import config  # type: ignore
    import network  # type: ignore
except ImportError:
    from src import config, network


def _download_ticker_frames(start, end, include_macro=True):
    all_data = []
    tickers = list(config.TICKERS)
    if include_macro:
        macro_tickers = getattr(config, "MACRO_TICKERS", [])
        fetch_list = tickers + [t for t in macro_tickers if t not in tickers]
    else:
        fetch_list = tickers

    for ticker in fetch_list:
        try:
            # Veriyi çek (Yahoo Finance sembol değişikliği varsa eşle)
            yahoo_ticker = getattr(config, "TICKER_YAHOO_MAP", {}).get(ticker, ticker)
            df = network.download_with_retry(yahoo_ticker, start=start, end=end, progress=False)

            # Multi-index düzeltmesi (yfinance yeni versiyonları için)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # .IS veya =X uzantısını temizle
            clean_ticker = ticker.replace(".IS", "").replace("=X", "")
            df["Ticker"] = clean_ticker
            df.reset_index(inplace=True)
            all_data.append(df)
            print(f"✅ {ticker} ({clean_ticker}) çekildi. ({len(df)} satır)")
        except Exception as e:
            print(f"❌ {ticker} hatası: {e}")

    return all_data


def fetch_incremental(start_date=None, end_date=None, output_path=None, include_macro=True):
    """Mevcut veri setindeki en son tarihi tespit ederek sadece eksik günleri indirir
    ve mevcut CSV'ye ekler. Dosya yoksa veya bozuksa otomatik olarak tam indirmeye döner."""
    out_path = output_path or config.DATA_PATH

    if not os.path.exists(out_path):
        print(f"Mevcut veri dosyası bulunamadı ({out_path}), tam indirmeye dönülüyor...")
        return fetch_data(
            start_date=start_date,
            end_date=end_date,
            output_path=out_path,
            include_macro=include_macro,
            incremental=False,
        )

    try:
        existing_df = pd.read_csv(out_path)
    except Exception as e:
        print(f"Mevcut dosya okunamadı ({e}), tam indirmeye dönülüyor...")
        return fetch_data(
            start_date=start_date,
            end_date=end_date,
            output_path=out_path,
            include_macro=include_macro,
            incremental=False,
        )

    if existing_df.empty or "Date" not in existing_df.columns:
        print("Mevcut veri setinde 'Date' sütunu bulunamadı, tam indirmeye dönülüyor...")
        return fetch_data(
            start_date=start_date,
            end_date=end_date,
            output_path=out_path,
            include_macro=include_macro,
            incremental=False,
        )

    date_series = pd.to_datetime(existing_df["Date"])
    latest_date = date_series.max()
    if pd.isna(latest_date):
        print("Geçerli bir son tarih tespit edilemedi, tam indirmeye dönülüyor...")
        return fetch_data(
            start_date=start_date,
            end_date=end_date,
            output_path=out_path,
            include_macro=include_macro,
            incremental=False,
        )

    inc_start = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    inc_end = end_date or config.END_DATE
    if inc_end is None or inc_end in ("today", "latest"):
        inc_end = date.today().isoformat()

    if inc_start > inc_end:
        print(
            f"Veri seti zaten güncel ({latest_date.strftime('%Y-%m-%d')}). Yeni indirme yapılmadı."
        )
        return existing_df

    print(f"Mevcut veri setinin son tarihi: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Eksik aralık için artımlı indirme başlatılıyor: {inc_start} -> {inc_end}")

    new_data = _download_ticker_frames(inc_start, inc_end, include_macro=include_macro)
    if not new_data:
        print("Yeni veri indirilemedi, mevcut veri seti korundu.")
        return existing_df

    new_df = pd.concat(new_data, ignore_index=True)
    new_df.columns = [c.lower() for c in new_df.columns]
    new_df.rename(columns={"date": "Date"}, inplace=True)
    new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.strftime("%Y-%m-%d")

    # Mevcut veriyi ve yeni veriyi birleştir
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    # Tarih ve ticker bazında mükerrer kayıtları temizle (yeni gelen geçerli)
    combined_df.drop_duplicates(subset=["Date", "ticker"], keep="last", inplace=True)
    combined_df.sort_values(by=["Date", "ticker"], inplace=True)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    combined_df.to_csv(out_path, index=False)
    added_rows = len(combined_df) - len(existing_df)
    print(f"\n🎉 Veri seti güncellendi: {out_path}")
    print(f"Eklenen yeni satır sayısı: {added_rows}")
    print(f"Toplam Satır Sayısı: {len(combined_df)}")
    return combined_df


def fetch_data(
    start_date=None,
    end_date=None,
    output_path=None,
    include_macro=True,
    incremental=False,
):
    if incremental:
        return fetch_incremental(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            include_macro=include_macro,
        )

    start = start_date or config.START_DATE
    end = end_date or config.END_DATE
    if end is None or end in ("today", "latest"):
        end = date.today().isoformat()
    out_path = output_path or config.DATA_PATH

    print(f"Veri çekme işlemi başladı ({start} - {end})... Bu işlem biraz sürebilir.")
    all_data = _download_ticker_frames(start, end, include_macro=include_macro)

    if not all_data:
        print("Hiçbir hisse verisi çekilemedi.")
        return pd.DataFrame()

    # Tüm hisseleri alt alta birleştir
    final_df = pd.concat(all_data, ignore_index=True)

    # Sütun isimlerini düzenle
    final_df.columns = [c.lower() for c in final_df.columns]
    final_df.rename(columns={"date": "Date"}, inplace=True)
    final_df["Date"] = pd.to_datetime(final_df["Date"]).dt.strftime("%Y-%m-%d")
    final_df.sort_values(by=["Date", "ticker"], inplace=True)

    # Kaydet
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    final_df.to_csv(out_path, index=False)
    print(f"\n🎉 Veri seti oluşturuldu: {out_path}")
    print(f"Toplam Satır Sayısı: {len(final_df)}")
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 30 veri seti indirme aracı")
    parser.add_argument("--start-date", default=None, help="Başlangıç tarihi (YYYY-MM-DD)")
    parser.add_argument(
        "--end-date",
        default=None,
        help="Bitiş tarihi (YYYY-MM-DD) veya 'today' / 'latest'",
    )
    parser.add_argument("--output-path", default=None, help="Çıktı CSV dosya yolu")
    parser.add_argument(
        "--no-macro",
        action="store_true",
        help="Endeks ve kur verilerini indirme",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Mevcut veri setini sadece eksik günlerle güncelle",
    )
    args = parser.parse_args()

    end_d = args.end_date
    if end_d in ("today", "latest"):
        end_d = date.today().isoformat()

    fetch_data(
        start_date=args.start_date,
        end_date=end_d,
        output_path=args.output_path,
        include_macro=not args.no_macro,
        incremental=args.incremental,
    )
