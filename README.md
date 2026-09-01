# Grid Up Datathon

[![Submissionları doğrula](https://github.com/atakanal/grid-up-datathon/actions/workflows/submissionlari-dogrula.yml/badge.svg)](https://github.com/atakanal/grid-up-datathon/actions/workflows/submissionlari-dogrula.yml)

Bu repo, [Grid Up Datathon](https://www.kaggle.com/competitions/grid-up-datathon)
için geliştirdiğimiz günlük trafo tüketimi tahmin çalışmasının yarışmaya özel
ve sadeleştirilmiş arşividir.

Amaç, günlük tüketimi RMSLE metriğine göre tahmin etmektir. Nihai yaklaşım;
geçmişi bulunan sıcak trafoları, geçmişi bulunmayan soğuk trafolardan ayrı ele
alır ve tahminleri `log1p` uzayında birleştirir.

## En iyi sonuçlarımız

| Genel skor | Submission | Satır | SHA-256 |
|---:|---|---:|---|
| **0,99430** | [`gridup_v108_uniform_endpoint_candidate.csv`](submissions/gridup_v108_uniform_endpoint_candidate.csv) | 714.688 | `E4EF38...1151` |
| **0,99676** | [`gridup_v96b_fresh_warm_challenger.csv`](submissions/gridup_v96b_fresh_warm_challenger.csv) | 714.688 | `FBCA47...CFFC` |

Her iki dosyada da:

- 714.688 benzersiz ID bulunur;
- resmî örnek submission sırası birebir korunur;
- eksik, sonsuz veya negatif tahmin bulunmaz;
- dosya bütünlüğü SHA-256 ile dondurulmuştur.

Tam hash değerleri ve makine tarafından okunabilir sonuç kaydı
[`sonuclar/liderlik_sonuclari.json`](sonuclar/liderlik_sonuclari.json)
dosyasındadır.

## Yaklaşımın özeti

- **Sıcak/soğuk ayrımı:** Testteki 7.036 trafonun 5.012'si eğitim verisinde
  geçmişe sahip, 2.024'ü ise tamamen yenidir.
- **Sıcak tahmin:** Yakın dönem seviye ve eğilim, haftanın günü profili,
  geçen yıl mevsimselliği, geçmiş uzunluğu ve güncellik kullanıldı.
- **Soğuk tahmin:** Güç, konum, doğal devreye giriş grupları, yaşam döngüsü ve
  hiyerarşik daraltma sinyalleri kullanıldı.
- **Dış sinyaller:** Takvim, hava, su, elektrik sistemi ve faaliyet göstergeleri
  ayrı kontrollerle araştırıldı.
- **Nihai seçim:** Adaylar satır, trafo, gün, ay, tahmin ufku ve model ailesi
  bazında stres testlerinden geçirildi.

Ayrıntılar için [`docs/YAKLASIM.md`](docs/YAKLASIM.md), dosya doğrulama adımları
için [`docs/TEKRAR_URETIM.md`](docs/TEKRAR_URETIM.md) okunabilir.

## Repo yapısı

```text
.
├── docs/          # Yarışmaya özgü yöntem ve doğrulama belgeleri
├── scripts/       # Final submission bütünlük doğrulayıcısı
├── sonuclar/      # Skor ve SHA-256 manifesti
└── submissions/   # En iyi iki gerçek submission dosyası
```

## Submission dosyalarını doğrulama

Repo kök dizininde:

```bash
python scripts/submissionlari_dogrula.py
```

Resmî yarışma ZIP'i mevcutsa ID sırası da doğrulanabilir:

```powershell
python .\scripts\submissionlari_dogrula.py `
  --yarisma-zip C:\veri\grid-up-datathon.zip
```

## Veri ve uygunluk notu

Yarışma verileri ile üçüncü taraf ham veri dosyaları bu repoda dağıtılmaz.
V108'in model soy ağacında 31 Mart 2026 sonrasına ait gerçekleşmiş dış
gözlemlerden etkilenen bileşenler vardır. Bu nedenle V108, nihai notebook veya
jüri değerlendirmesinde kullanılmadan önce organizatörün dış veri yorumuyla
uygunluk açısından ayrıca incelenmelidir.
