# Modelleme yaklaşımı

## Problemin tanımı

Hedef, trafoların günlük elektrik tüketimini RMSLE metriğine göre tahmin
etmektir. RMSLE, `log1p(gerçek)` ile `log1p(tahmin)` arasındaki RMSE'ye eşit
olduğu için eğitim, düzeltme ve birleştirme işlemleri logaritmik uzayda yapıldı.

Test panelinde birbirinden farklı iki ana grup vardır:

| Grup | Trafo | Test satırı | Pay |
|---|---:|---:|---:|
| Sıcak — eğitim verisinde geçmişi var | 5.012 | 556.319 | %77,84 |
| Soğuk — eğitim verisinde hiç görülmemiş | 2.024 | 158.369 | %22,16 |
| **Toplam** | **7.036** | **714.688** | **%100** |

Bu ayrım hem modellemeyi hem de doğrulama düzenini belirledi.

## Sıcak trafolar

Sıcak trafolarda en güçlü bilgi trafonun kendi geçmişidir. Araştırılan başlıca
bileşenler şunlardır:

- son 14, 28 ve 90 günlük dayanıklı seviye özetleri;
- trafo × haftanın günü ve ay × haftanın günü profilleri;
- geçen yılın eşlenmiş profili ve daraltılmış yıllık değişim;
- geçmiş uzunluğu, güncellik, oynaklık ve sıfır tüketim rejimi;
- tatil, Ramazan, okul takvimi ve hava etkileşimleri;
- artık hata ağaç modelleri ve önceden eğitilmiş zaman serisi modelleri.

Son yönlendirmede güncel ve güçlü geçmişe sahip trafolar ile eski veya eksik
geçmişe sahip trafolar ayrı ele alındı.

## Soğuk trafolar

Soğuk trafoların hedef geçmişi bulunmadığından yalnızca dışarıdan bilinen veya
başka trafolardan aktarılabilen sinyaller kullanıldı:

- kurulu güç ve konum hiyerarşisi;
- güce göre normalize edilmiş tüketim dağılımları;
- doğal devreye giriş grupları ve devreye giriş yaşı;
- aktif/pasif olasılığı ve yaşam döngüsü temsilleri;
- yapay olarak geçmişi gizlenmiş sıcak trafolar;
- konum → güç → genel seviye yönünde hiyerarşik daraltma.

Doğal yeni trafo grupları ile geçmişi tamamen gizlenen trafo ayrımları birlikte
kullanıldı. Yapay soğuk trafoların gerçek devreye giriş davranışını kusursuz
temsil etmediği ayrıca takip edildi.

## Doğrulama

Zaman blokları kronolojik olarak ayrıldı ve doğrulama hedeflerinin gecikme veya
kayan pencere özelliklerine sızması engellendi. Nihai aday yönleri ayrıca şu
deterministik dilimlerde sınandı:

- rastgele satır alt kümeleri;
- trafo kimliği alt kümeleri;
- gün alt kümeleri;
- takvim ayları;
- tahmin ufku blokları;
- bütün bir model ailesinin çıkarıldığı kontroller.

Bu kontroller tek bir doğrulama dilimine bağımlılığı azaltır; görünmeyen test
etiketlerini ortaya çıkarmaz ve özel liderlik tablosu için garanti vermez.

## V96b — 0,99676

V96b, iki önceki aday arasındaki yönü trafo geçmişinin kalitesine göre uygular.
Güncel sıcak satırlarda daha büyük, eski sıcak ve soğuk satırlarda daha küçük
bir katsayı kullanır. Katsayılar, korunan grubun bilinmeyen ideal noktasına
karşı dondurulmuş minimaks taramasıyla seçildi.

## V108 — 0,99430

V108, V100 tabanı ile daha önce skoru gözlenmiş 17 farklı aday yönünü `log1p`
uzayında birleştirir. Kısıtlı ikinci dereceden optimizasyon, aynı hareketin
satır, trafo, gün, ay ve tahmin ufku stres dilimlerinin tamamında olumlu kalması
şartıyla çözüldü.

V108 için tahmin edilen genel skor merkezi `0,994281`, gerçekleşen skor ise
`0,99430` oldu. Mutlak tahmin farkı yaklaşık `0,000019` düzeyindedir.

Model ailesi çıkarma kontrollerinin 14'ünden 12'si V108 yönünü destekledi.
14/14 kontrolden geçen daha temkinli çözümün beklenen merkezi `0,994313`
olduğu için son submission olarak seçilmedi.

## Dış veri notu

Araştırma boyunca hava, su tüketimi, ulusal yük, sektör hareketi, kesintiler,
turizm, liman, tarım ve benzeri kaynaklar denendi. V108'in soy ağacı,
Nisan–Temmuz 2026 dönemindeki gerçekleşmiş bazı dış gözlemlerden etkilenen aday
modeller içerir. Bu durum, organizatörün nihai dış veri yorumuna göre ayrıca
değerlendirilmelidir.
