# Tekrar üretim ve dosya bütünlüğü

## Repoda bulunanlar

Repo, genel liderlik tablosundaki en iyi iki submission dosyasını ve bunların
skor, satır sayısı ve SHA-256 değerlerini içeren dondurulmuş manifesti barındırır.
Yarışmanın ham verileri, üçüncü taraf veri dosyaları, kimlik bilgileri, model
ağırlıkları ve ara tahminler bilerek repoya eklenmemiştir.

## Submission doğrulama

Repo kök dizininde şu komut çalıştırılır:

```bash
python scripts/submissionlari_dogrula.py
```

Doğrulayıcı şu kontrolleri gerçekleştirir:

1. sütunların tam olarak `id,tuketim` olması;
2. 714.688 satır ve 714.688 benzersiz ID bulunması;
3. bütün tahminlerin sonlu ve negatif olmaması;
4. SHA-256 değerinin dondurulmuş manifestle birebir uyuşması.

Resmî yarışma ZIP'i mevcutsa örnek submission sırası da kontrol edilir:

```powershell
python .\scripts\submissionlari_dogrula.py `
  --yarisma-zip C:\veri\grid-up-datathon.zip
```

## Yeniden eğitim kapsamı

V108 çok sayıda ara tahmin ve ayrı dış veri kaynağına dayanan bir uç nokta
birleşimidir. Bu kaynakların tamamı dağıtılmadığı için repo, V108'i sıfırdan tek
komutla eğittiğini iddia etmez. Bunun yerine yarışmada gerçekten gönderilen
dosyaların byte düzeyinde değişmeden saklandığını ve doğrulanabildiğini garanti
eder.

Uçtan uca yeniden eğitim; yarışma verisini, kullanılan dış kaynakları ve deney
soy ağacındaki ara tahminleri ayrıca gerektirir.
