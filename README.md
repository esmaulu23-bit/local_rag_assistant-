# Proje Tanıtımı
https://github.com/user-attachments/assets/28d0a388-ea73-4192-8f45-d9b6b19f9ed3
# 🤖 Offline Türkçe RAG Bilgi Asistanı

Bu proje, **Microsoft Foundry Local**, **SentenceTransformers**, **SQLite** ve **Streamlit** kullanılarak geliştirilmiş, tamamen yerel (offline) çalışan bir **Retrieval-Augmented Generation (RAG)** soru-cevap asistanıdır.

---

## 📌 Özellikler

- **%100 Çevrimdışı (Offline)**: İnternet bağlantısı veya bulut API anahtarı (OpenAI vb.) gerektirmez.
- **Dinamik Doküman İndeksleme**: `docs/` klasöründeki tüm `.txt` ve `.md` dosyalarını otomatik tarar, parçalara ayırır (chunking) ve SQLite veritabanına yazar.
- **Hafif & Hızlı Vektör Arama**: `all-MiniLM-L6-v2` embedding modeli ve Cosine Similarity yöntemi ile sub-millisecond hızında anlamsal eşleşme yapar.
- **Uydurma (Hallucination) Koruması**: Güçlü sistem prompt'u sayesinde dokümanda olmayan sorular için *"Bu bilgi yüklenen dokümanlarda yer almamaktadır."* cevabı verir.
- **Kullanılan Kaynak Gösterimi (Citation)**: Verilen cevabın hangi doküman ve parçalardan üretildiğini benzerlik skorları (% oranları) ile gösterir.
- **Kullanıcı Dostu Arayüz**: Streamlit sohbet arayüzü, sohbet geçmişi temizleme ve dinamik Top-K ayarı.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Ön Gereksinimler
- **Python 3.10+**
- **Microsoft Foundry Local**: Windows üzerinde yüklemek için:
  ```powershell
  winget install Microsoft.FoundryLocal
  ```

### 2. Bağımlılıkların Yüklenmesi
Sanal ortamı aktifleştirdikten sonra gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

### 3. Dokümanların İndekslemesi (Ingestion)
`docs/` klasörüne kendi `.txt` veya `.md` notlarınızı ekledikten sonra aşağıdaki komutla veritabanını oluşturun:
```bash
python ingestion.py
```

### 4. Test (Opsiyonel)
Arayüzü açmadan vektör aramasını doğrulamak için:
```bash
python test_rag.py
```

### 5. Uygulamanın Başlatılması
Streamlit arayüzünü başlatmak için:
```bash
streamlit run app.py
```
Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

---

## 📁 Proje Yapısı

```
local_rag_assistant/
├── docs/                      # İndekslenecek metin ve markdown dokümanları
│   ├── bilgi_notlari.txt
│   └── ders_notlari.md
├── app.py                     # Streamlit kullanıcı arayüzü ve RAG motoru
├── ingestion.py               # Doküman okuma, chunking ve SQLite vektör kayıt betiği
├── test_rag.py                # Vektör arama doğrulama ve test betiği
├── knowledge_base.db          # SQLite vektör veritabanı (otomatik oluşur)
├── requirements.txt           # Proje kütüphane bağımlılıkları
└── README.md                  # Proje dokümantasyonu
```
