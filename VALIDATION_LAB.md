# Validation Lab — Grid Up Hazırlık Çalışması

Bu laboratuvarın amacı bir modelden önce **doğru validasyon problemini** çözmektir. Aynı sentetik elektrik tüketim verisi üzerinde üç risk incelenir:

1. Geleceği tahmin ederken rastgele KFold kullanmak
2. Yeni sayaç/trafo genellemesinde aynı varlığı train ve validation'a bölmek
3. Rolling özellik üretirken güncel hedefi yanlışlıkla özelliğe katmak

## Çalıştırma

PowerShell'de proje kök klasöründeyken:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_validation_lab.py
```

Oluşacak dosyalar:

- `data/processed/validation_lab.csv`
- `experiments/validation_lab_results.csv`
- `experiments/validation_lab_rmse.png`

## Sonuçları nasıl okumalıyız?

RMSE için düşük değer daha iyidir. Ancak en düşük görünen CV skoru her zaman en güvenilir skor değildir.

### Gelecek tahmini

Random KFold geçmiş ve daha ileri tarihleri karıştırır. Model validation satırının çevresindeki dönemleri eğitimde görebilir. TimeSeriesSplit ise her fold'da yalnızca geçmişten geleceğe gider. Bu nedenle TimeSeriesSplit skoru daha kötü görünse bile gerçek kullanım koşuluna daha yakındır.

### Görülmemiş varlıklar

Aynı `meter_id` hem eğitim hem validation tarafına girerse model sayaç kimliğini ezberleyebilir. Deployment sırasında yeni sayaç veya trafo gelecekse GroupKFold kullanılmalıdır.

### Feature leakage

`safe_roll_3`, rolling işleminden önce `shift(1)` uygular. `leaky_roll_3` ise güncel hedefi ortalamaya katar. Leaky özellik şaşırtıcı derecede iyi skor üretir, fakat test zamanında hesaplanamaz.

## Tamamlanma kriteri

Aşağıdaki üç cümleyi kendi kelimelerinle açıklayabiliyorsan laboratuvar tamamdır:

- Split seçimi, test verisinin nasıl üretildiğine bağlıdır.
- Bir özellik yalnızca tahmin anında mevcut bilgilerden üretilmelidir.
- Public leaderboard, yerel validasyonun yerine geçmez.

## Git kaydı

```powershell
git add scripts/run_validation_lab.py VALIDATION_LAB.md
git commit -m "Add validation strategy lab"
```
