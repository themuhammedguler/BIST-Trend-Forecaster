# src/data_ingestion.py
import yfinance as yf
import pandas as pd
import config
import os

def fetch_data():
    print("Veri çekme işlemi başladı... Bu işlem biraz sürebilir.")
    all_data = []
    
    for ticker in config.TICKERS:
        try:
            # Veriyi çek
            df = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
            
            # Multi-index düzeltmesi (yfinance yeni versiyonları için)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df['Ticker'] = ticker.replace(".IS", "") # .IS uzantısını temizle
            df.reset_index(inplace=True)
            all_data.append(df)
            print(f"✅ {ticker} çekildi. ({len(df)} satır)")
        except Exception as e:
            print(f"❌ {ticker} hatası: {e}")
            
    # Tüm hisseleri alt alta birleştir
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sütun isimlerini düzenle
    final_df.columns = [c.lower() for c in final_df.columns]
    final_df.rename(columns={'date': 'Date'}, inplace=True)
    
    # Kaydet
    if not os.path.exists(os.path.dirname(config.DATA_PATH)):
        os.makedirs(os.path.dirname(config.DATA_PATH))
        
    final_df.to_csv(config.DATA_PATH, index=False)
    print(f"\n🎉 Veri seti oluşturuldu: {config.DATA_PATH}")
    print(f"Toplam Satır Sayısı: {len(final_df)}")

if __name__ == "__main__":
    fetch_data()