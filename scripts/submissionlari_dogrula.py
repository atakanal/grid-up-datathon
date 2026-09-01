"""Dondurulmuş yarışma submissionlarını manifest ve örnek sıra ile doğrular."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path


KOK_DIZIN = Path(__file__).resolve().parents[1]
MANIFEST = KOK_DIZIN / "sonuclar" / "liderlik_sonuclari.json"


def sha256(dosya: Path) -> str:
    ozet = hashlib.sha256()
    with dosya.open("rb") as akis:
        for parca in iter(lambda: akis.read(1024 * 1024), b""):
            ozet.update(parca)
    return ozet.hexdigest().upper()


def ornek_idler(zip_dosyasi: Path | None) -> list[str] | None:
    if zip_dosyasi is None:
        return None
    with zipfile.ZipFile(zip_dosyasi) as arsiv:
        with arsiv.open("sample_submission.csv") as ham_dosya:
            satirlar = csv.DictReader(satir.decode("utf-8-sig") for satir in ham_dosya)
            if satirlar.fieldnames is None or "id" not in satirlar.fieldnames:
                raise AssertionError("sample_submission.csv içinde id sütunu bulunamadı")
            return [satir["id"] for satir in satirlar]


def dosyayi_dogrula(kayit: dict[str, object], beklenen_idler: list[str] | None) -> dict[str, object]:
    dosya = KOK_DIZIN / str(kayit["dosya"])
    if not dosya.is_file():
        raise FileNotFoundError(dosya)
    ozet = sha256(dosya)
    if ozet != kayit["sha256"]:
        raise AssertionError(f"SHA-256 uyuşmazlığı: {dosya}")

    idler: list[str] = []
    en_kucuk = math.inf
    en_buyuk = -math.inf
    with dosya.open("r", encoding="utf-8-sig", newline="") as akis:
        okuyucu = csv.DictReader(akis)
        if okuyucu.fieldnames != ["id", "tuketim"]:
            raise AssertionError(f"Beklenmeyen sütunlar: {dosya}: {okuyucu.fieldnames}")
        for satir in okuyucu:
            idler.append(satir["id"])
            deger = float(satir["tuketim"])
            if not math.isfinite(deger) or deger < 0.0:
                raise AssertionError(f"Geçersiz tahmin: {dosya}")
            en_kucuk = min(en_kucuk, deger)
            en_buyuk = max(en_buyuk, deger)

    benzersiz_id = len(set(idler))
    if len(idler) != int(kayit["satir_sayisi"]) or benzersiz_id != int(kayit["benzersiz_id"]):
        raise AssertionError(f"Satır veya benzersiz ID sayısı uyuşmuyor: {dosya}")
    if beklenen_idler is not None and idler != beklenen_idler:
        raise AssertionError(f"Örnek submission sırası uyuşmuyor: {dosya}")
    return {
        "surum": kayit["surum"],
        "genel_skor": kayit["genel_skor"],
        "satir_sayisi": len(idler),
        "benzersiz_id": benzersiz_id,
        "en_kucuk_tahmin": en_kucuk,
        "en_buyuk_tahmin": en_buyuk,
        "ornek_sira_kontrol_edildi": beklenen_idler is not None,
        "sha256": ozet,
    }


def ana() -> None:
    ayrac = argparse.ArgumentParser()
    ayrac.add_argument("--yarisma-zip", type=Path)
    secenekler = ayrac.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    beklenen_idler = ornek_idler(secenekler.yarisma_zip)
    sonuclar = [
        dosyayi_dogrula(kayit, beklenen_idler)
        for kayit in manifest["submissionlar"]
    ]
    print(json.dumps(sonuclar, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    ana()
