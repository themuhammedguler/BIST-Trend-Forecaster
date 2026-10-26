# src/drift.py
"""BIST 30 Veri ve Öznitelik Sapması (Data Drift / PSI) İzleme Modülü (#45).

Finansal piyasaların dinamik yapısı nedeniyle modelin eğitildiği dağılım ile canlı
piyasadaki öznitelik dağılımları zamanla kayabilir. Bu modül, Population Stability Index (PSI)
kullanarak sayısal göstergelerdeki sapmayı hesaplar ve sistem sağlığını izler.

PSI Eşikleri:
  - PSI < 0.10: Sağlıklı (Sapma yok, model güvenilir)
  - 0.10 <= PSI <= 0.25: Hafif Sapma (İzlenmeli)
  - PSI > 0.25: Kritik Sapma (Yeniden eğitim önerilir)
"""

import argparse
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

try:
    from src import config, features
except ImportError:
    import config  # type: ignore
    import features  # type: ignore


CYCLICAL_FEATURES = {"month", "day_of_week"}


def calculate_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
    pseudocount: float = 0.5,
) -> float:
    """İki veri dağılımı (referans/eğitim ve güncel/çıkarım) arasındaki
    Population Stability Index (PSI) değerini hesaplar.
    """
    if isinstance(reference, pd.Series):
        ref_arr = reference.to_numpy()
    else:
        ref_arr = np.asarray(reference)

    if isinstance(current, pd.Series):
        curr_arr = current.to_numpy()
    else:
        curr_arr = np.asarray(current)

    # NaN ve sonsuz değerleri filtrele
    ref_clean = ref_arr[np.isfinite(ref_arr)]
    curr_clean = curr_arr[np.isfinite(curr_arr)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Sabit değer / sıfır varyans kontrolü
    ref_min, ref_max = float(np.min(ref_clean)), float(np.max(ref_clean))
    if ref_min == ref_max:
        curr_min, curr_max = float(np.min(curr_clean)), float(np.max(curr_clean))
        if curr_min == curr_max == ref_min:
            return 0.0
        return 1.0

    # Referans dağılım üzerinden yüzdelik kovaları hesapla
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        bin_edges = np.array([ref_min - 1e-5, ref_max + 1e-5])

    # Dışarıdaki değerleri kapsamak için sonsuz sınırları ekle
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Histogram frekansları
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    k = len(ref_counts)
    # Laplace/Lidstone düzeltmesi ile sıfır frekans patlamasını önle
    alpha = max(epsilon, pseudocount)
    ref_pct = (ref_counts + alpha) / (len(ref_clean) + k * alpha)
    curr_pct = (curr_counts + alpha) / (len(curr_clean) + k * alpha)

    # PSI toplam formülü
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi))


def classify_drift(psi: float, feature_name: Optional[str] = None) -> tuple[str, str, str]:
    """PSI değerine ve öznitelik tipine göre durum kategorisi, rozet ve öneri metni döndürür."""
    if feature_name in CYCLICAL_FEATURES:
        return (
            "Takvim Döngüsü",
            "📅 Takvim",
            "Doğal takvim döngüsü; modelin yeniden eğitilmesini gerektirmez.",
        )
    if psi < 0.10:
        return (
            "Sağlıklı",
            "🟢 Sağlıklı",
            "Sapma yok, model güvenle kullanılabilir.",
        )
    elif psi <= 0.25:
        return (
            "Hafif Sapma",
            "🟡 Dikkat",
            "Hafif dağılım kayması mevcut, performansı izleyin.",
        )
    else:
        return (
            "Kritik Sapma",
            "🔴 Sapma Algılandı",
            "Kritik veri sapması tespit edildi! Modelin yeniden eğitilmesi önerilir.",
        )


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[list[str]] = None,
    num_buckets: int = 10,
) -> pd.DataFrame:
    """İki DataFrame arasında öznitelik bazlı PSI ve sapma tablosu üretir."""
    if features is None:
        target_features = [
            f for f in config.FEATURES if f in reference_df.columns and f in current_df.columns
        ]
        if not target_features:
            common = set(reference_df.columns).intersection(set(current_df.columns))
            target_features = [
                c for c in sorted(common) if pd.api.types.is_numeric_dtype(reference_df[c])
            ]
    else:
        target_features = [
            f for f in features if f in reference_df.columns and f in current_df.columns
        ]

    records = []
    for feat in target_features:
        psi = calculate_psi(reference_df[feat], current_df[feat], num_buckets=num_buckets)
        status, badge, action = classify_drift(psi, feature_name=feat)
        records.append(
            {
                "Öznitelik": feat,
                "PSI": round(psi, 4),
                "Durum": status,
                "Rozet": badge,
                "Öneri": action,
            }
        )

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out = df_out.sort_values(by="PSI", ascending=False).reset_index(drop=True)
    return df_out


def summarize_drift(drift_df: pd.DataFrame) -> dict:
    """Öznitelik sapma tablosundan genel sistem sağlık özeti üretir."""
    if drift_df is None or drift_df.empty:
        return {
            "overall_status": "Sağlıklı",
            "badge": "🟢 Sağlıklı",
            "max_psi": 0.0,
            "mean_psi": 0.0,
            "critical_features": [],
            "moderate_features": [],
            "healthy_features": [],
            "cyclical_features": [],
            "total_features": 0,
        }

    max_psi = float(drift_df["PSI"].max())
    mean_psi = float(drift_df["PSI"].mean())
    critical = drift_df[drift_df["Durum"] == "Kritik Sapma"]["Öznitelik"].tolist()
    moderate = drift_df[drift_df["Durum"] == "Hafif Sapma"]["Öznitelik"].tolist()
    healthy = drift_df[drift_df["Durum"].isin(["Sağlıklı", "Takvim Döngüsü"])]["Öznitelik"].tolist()
    cyclical = drift_df[drift_df["Durum"] == "Takvim Döngüsü"]["Öznitelik"].tolist()

    if critical:
        overall_status = "Kritik Sapma"
        badge = "🔴 Sapma Algılandı"
    elif moderate:
        overall_status = "Hafif Sapma"
        badge = "🟡 Dikkat"
    else:
        overall_status = "Sağlıklı"
        badge = "🟢 Sağlıklı"

    return {
        "overall_status": overall_status,
        "badge": badge,
        "max_psi": round(max_psi, 4),
        "mean_psi": round(mean_psi, 4),
        "critical_features": critical,
        "moderate_features": moderate,
        "healthy_features": healthy,
        "cyclical_features": cyclical,
        "total_features": len(drift_df),
    }


def compute_series_drift_by_split(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    split_ratio: float = 0.8,
    recent_rows: Optional[int] = None,
    num_buckets: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Zaman serisi DataFrame'ini referans ve güncel pencere olarak ikiye ayırıp
    sapma tablosu ve özetini döndürür."""
    if df is None or len(df) < 10:
        empty_df = pd.DataFrame(columns=["Öznitelik", "PSI", "Durum", "Rozet", "Öneri"])
        return empty_df, summarize_drift(empty_df)

    if recent_rows is not None and 0 < recent_rows < len(df):
        if "Date" in df.columns and df["Date"].nunique() > recent_rows:
            unique_dates = df["Date"].drop_duplicates().sort_values()
            cutoff_dates = set(unique_dates.iloc[-recent_rows:])
            ref_df = df[~df["Date"].isin(cutoff_dates)]
            curr_df = df[df["Date"].isin(cutoff_dates)]
        else:
            ref_df = df.iloc[:-recent_rows]
            curr_df = df.iloc[-recent_rows:]
    else:
        split_idx = max(1, int(len(df) * split_ratio))
        ref_df = df.iloc[:split_idx]
        curr_df = df.iloc[split_idx:]

    drift_df = compute_feature_drift(ref_df, curr_df, features=features, num_buckets=num_buckets)
    summary = summarize_drift(drift_df)
    return drift_df, summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="BIST 30 Veri ve Öznitelik Sapması (Data Drift / PSI) İzleme Aracı"
    )
    parser.add_argument(
        "--data-path",
        default=config.DATA_PATH,
        help="Veri seti yolu (varsayılan: config.DATA_PATH)",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=60,
        help="Güncel çıkarım penceresi gün sayısı (varsayılan: 60)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Kritik sapma tespit edildiğinde çıkış kodunu 1 yap",
    )
    args = parser.parse_args()

    print("🔍 BIST 30 Veri ve Öznitelik Sapması (Data Drift) Analizi Başlatılıyor...")
    try:
        raw_df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"❌ Veri dosyası okunamadı: {e}")
        sys.exit(1)

    print(f"📊 Ham veri okundu ({len(raw_df)} satır). Öznitelikler hesaplanıyor...")
    df_feat = features.add_features(raw_df, drop_incomplete_target=False)

    drift_df, summary = compute_series_drift_by_split(df_feat, recent_rows=args.recent_days)

    print("\n" + "=" * 65)
    print(f"  ÖZNİTELİK SAPMASI RAPORU | Genel Durum: {summary['badge']}")
    print(f"  Maksimum PSI: {summary['max_psi']:.4f} | Ortalama PSI: {summary['mean_psi']:.4f}")
    print("=" * 65)

    for _, row in drift_df.iterrows():
        print(
            f"  {row['Rozet']:<18} | {row['Öznitelik']:<18} | "
            f"PSI: {row['PSI']:<6.4f} | {row['Öneri']}"
        )
    print("=" * 65)

    if args.strict and summary["critical_features"]:
        print(f"\n❌ Kritik sapma tespit edildi ({len(summary['critical_features'])} öznitelik).")
        sys.exit(1)
    else:
        print("\n✅ Veri sapması analizi tamamlandı.")


if __name__ == "__main__":
    main()
