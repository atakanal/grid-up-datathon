# Grid Up Competition Rehearsal

Bu laboratuvar, gerçek veri yayımlanmadan önce bütün yarışma zincirini prova eder:

1. Sentetik `train.csv`, `test.csv` ve `sample_submission.csv` üretir.
2. Hedef değişkeni okumadan güvenli özellikler oluşturur.
3. Aynı timestamp'i iki tarafa bölmeyen zaman bloklu validasyon uygular.
4. Her fold için modeli yalnızca geçmiş veride eğitir.
5. OOF tahminleri ve fold RMSE değerlerini kaydeder.
6. Nihai modeli bütün train verisinde eğitir.
7. Test tahminini sample submission ile aynı kimlik sırasında yazar.
8. Submission şemasını, NaN/sonsuz değerleri ve kimlik sırasını doğrular.

## Çalıştırma

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\scripts\run_competition_rehearsal.py
```

## Üretilen dosyalar

- `data/raw/rehearsal_train.csv`
- `data/raw/rehearsal_test.csv`
- `data/raw/rehearsal_sample_submission.csv`
- `data/processed/rehearsal_oof.csv`
- `experiments/rehearsal_results.csv`
- `experiments/rehearsal_feature_importance.csv`
- `experiments/rehearsal_oof_scatter.png`
- `experiments/rehearsal_fold_scores.png`
- `experiments/rehearsal_feature_importance.png`
- `models/rehearsal_randomforest.joblib`
- `submissions/rehearsal_baseline.csv`

## İncelenecek noktalar

- Fold skorları zaman ilerledikçe bozuluyor mu?
- Fold standart sapması yüksek mi?
- OOF grafiğinde yüksek veya düşük tüketimler sistematik biçimde hatalı mı?
- En önemli özellikler operasyonel olarak anlamlı mı?
- Submission kimlikleri sample submission ile birebir ve aynı sırada mı?

Sentetik skorun kendisi önemli değildir. Önemli olan, veri geldiğinde aynı akışın dosya adları ve yarışma metriği değiştirilerek güvenle çalıştırılabilmesidir.
