# Proje Önerisi

## Seçilen Görev Numarası
Seçenek 3 — Açıklanabilir Makine Öğrenmesi Karar Destek Ürünü

## Ürünün Adı
RiskRadar - Kayıp Riski Karar Destek Sistemi

## Çözülecek Problem
Telekom şirketleri, hangi müşterilerin yakın zamanda hizmeti bırakma
(churn) riski taşıdığını önceden bilemediği için müşteri elde tutma
kampanyalarını geç başlatmakta veya rastgele müşterilere
yönlendirmektedir. RiskRadar, mevcut müşteri verilerini kullanarak her
müşteri için bir kayıp riski skoru üretir ve bu riskin hangi faktörlerden
(sözleşme tipi, fatura tutarı, hizmet süresi vb.) kaynaklandığını açıklar.

## Hedef Kullanıcı
Telekom şirketinde müşteri ilişkileri / müşteri elde tutma (retention) ekibi.
Bu ekip, hangi müşterilere öncelikli olarak ulaşılması gerektiğine ve
hangi konuda (fatura, sözleşme, hizmet kalitesi vb.) aksiyon alınması
gerektiğine karar vermek için ürünü kullanır.

## Kullanılacak Veri veya Bilgi Kaynakları
- Kaggle "Telco Customer Churn" veri seti
  (WA_Fn-UseC_-Telco-Customer-Churn.csv, 7043 müşteri, 21 değişken)
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Kullanılması Planlanan Teknolojiler
- Python (pandas, numpy) – veri temizleme ve hazırlama
- scikit-learn – model eğitimi (en az 2 model karşılaştırması:
  Lojistik Regresyon ve Random Forest)
- SHAP – model kararlarının açıklanması
- matplotlib, seaborn – görselleştirme
- Streamlit – kullanıcının müşteri bilgisi girip risk skoru ve
  açıklama alabileceği arayüz
- GitHub – proje yönetimi ve sürüm kontrolü

## Beklenen Ürün Çıktısı
Kullanıcının (müşteri ilişkileri ekibinin) bir müşterinin bilgilerini
(sözleşme tipi, hizmet süresi, aylık ücret, internet/telefon hizmetleri
vb.) girebildiği; girilen bilgilere göre o müşterinin kayıp (churn)
olasılığını ve risk seviyesini (düşük/orta/yüksek) gösteren; ayrıca bu
tahmine en çok etki eden faktörleri SHAP grafiğiyle açıklayan, çalışan
bir Streamlit uygulaması.

## Ürünün Diğer Çalışmalardan Ayrılan Yönü
Bu projede amaç sadece bir tahmin modeli üretmek değil, müşteri elde
tutma ekibinin günlük iş akışına uygun, açıklanabilir bir karar destek
aracı sunmaktır. Model, her müşteri için ayrı ayrı "neden risk taşıyor"
sorusuna SHAP ile cevap verir; böylece ekip genel istatistik yerine
müşteriye özel, aksiyon alınabilir bilgiye erişir.