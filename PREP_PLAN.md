# 12–29 Temmuz Hazırlık Planı

## 1. Hafta — Altyapı ve tekrar üretilebilirlik

- [ ] Python 3.11 ortamını kur
- [ ] Gereksinimleri yükle ve ortam kontrolünü çalıştır
- [ ] Git deposunu başlat
- [ ] Deney günlüğü formatını sabitle
- [ ] Eğitim komutunu sentetik veriyle çalıştır
- [ ] Submission şema kontrolünü test et

## 2. Hafta — Validasyon ve model kası

- [ ] KFold, StratifiedKFold, GroupKFold ve TimeSeriesSplit farklarını uygulamalı test et
- [ ] CatBoost ve LightGBM baseline'larını çalıştır
- [ ] OOF tahmin üretimini doğrula
- [ ] Seed tekrarıyla skor kararlılığını ölç
- [ ] Basit blend kodunu dene

## 3. Hafta — Yarışma provası

- [ ] Açık bir Kaggle tabular veri setinde 3 günlük mini prova yap
- [ ] İlk 6 saatte baseline çıkarma pratiği yap
- [ ] EDA notlarını kısa ve karar odaklı yaz
- [ ] En iyi iki modelin hata korelasyonunu incele
- [ ] Final notebook'u temiz ortamda çalıştır

## Hazır olma kriteri

29 Temmuz sonunda şu komutlar hatasız çalışmalı:

```bash
python scripts/check_environment.py
pytest -q
python -m src.train --help
python -m src.submit --help
```
