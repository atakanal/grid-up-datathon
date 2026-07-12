# Grid Up Datathon Starter

Bu depo, yarışma verisi yayınlanmadan önce kurulabilecek tekrar üretilebilir bir makine öğrenmesi altyapısıdır.

## Hızlı başlangıç

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_environment.py
python scripts/init_experiment_log.py
```

Veriler geldiğinde dosyaları `data/raw/` altına koyun ve aşağıdaki komutla ilk eğitimi başlatın:

```bash
python -m src.train \
  --train-path data/raw/train.csv \
  --test-path data/raw/test.csv \
  --target TARGET_COLUMN \
  --id-column ID_COLUMN \
  --task regression \
  --model randomforest \
  --cv kfold
```

Zaman veya grup bağımlılığı varsa `--cv timeseries` ya da `--cv group` kullanın. Grup validasyonunda ayrıca `--group-column` verilmelidir.

## Ana prensipler

- Public leaderboard yerine yerel CV karar mekanizmasıdır.
- Her deney `experiments/experiments.csv` dosyasına kaydedilir.
- Özellik üretimi fold dışında hedef bilgisi kullanmamalıdır.
- Nihai notebook temiz ortamda baştan sona çalışmalıdır.

## Klasörler

- `data/raw/`: Ham yarışma dosyaları; Git'e eklenmez.
- `data/processed/`: Ara veri setleri; Git'e eklenmez.
- `src/`: Eğitim, validasyon, özellik ve submission kodu.
- `notebooks/`: EDA ve final notebook şablonları.
- `experiments/`: Deney günlüğü.
- `models/`: Eğitilmiş modeller; Git'e eklenmez.
- `submissions/`: Kaggle gönderimleri.
- `tests/`: Basit güvenlik kontrolleri.

## Veri gelince ilk kontrol

1. Train/test şekilleri ve sütun tipleri
2. Hedef dağılımı ve resmî metrik
3. ID tekrarları, grup yapısı ve zaman sıralaması
4. Train-test kategorik farkları
5. Eksik değer desenleri
6. Leakage adayları
7. Sample submission şeması


`randomforest`, yalnızca scikit-learn ile çalışan hızlı sağlık kontrolü modelidir. Yarışmada ana adaylar LightGBM, CatBoost ve XGBoost olacaktır.
