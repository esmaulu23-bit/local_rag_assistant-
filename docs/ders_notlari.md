# Yerel Yapay Zeka ve RAG Mimarisi Ders Notları

## 1. Embeddings ve Vektör Arama
Vektör temsilcileri (embeddings), metinlerin anlamsal (semantic) anlamlarını sayılardan oluşan çok boyutlu dizilere dönüştürür.
`all-MiniLM-L6-v2` modeli metinleri 384 boyutlu vektörlere dönüştürür.
Cosine Similarity (Kosinüs Benzerliği), iki metin vektörü arasındaki açısal yakınlığı ölçerek anlamsal olarak en benzer metinleri bulmayı sağlar.

## 2. Hallucination (Uydurma) Engelleme
Büyük Dil Modelleri (LLM), eğitildikleri genel bilgilerle yanıt üretirken emin olmadıkları konularda yanlış veya uydurma (hallucination) bilgiler üretebilirler.
RAG mimarisi, modele sistem talimatı (System Prompt) ile şu kuralı verir:
- "Sadece sana verilen bağlamdaki bilgileri kullan."
- "Aranan bilgi bağlamda yoksa 'Bu bilgi yüklenen dokümanlarda yer almamaktadır.' yanıtını ver."

## 3. SQLite ile Vektör Saklama
Küçük ve orta ölçekli projelerde karmaşık vektör veritabanları yerine sunucusuz (serverless) ve tek dosyadan oluşan SQLite kullanmak oldukça verimlidir.
Metin parçaları ve bu parçalara ait JSON formatındaki vektörler aynı tabloda saklanabilir.
