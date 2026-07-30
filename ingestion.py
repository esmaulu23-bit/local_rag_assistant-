import os
import sqlite3
import json
from sentence_transformers import SentenceTransformer

DB_PATH = 'knowledge_base.db'
DOCS_DIR = 'docs'

def init_db():
    """SQLite veritabanını ve parçalar (chunks) tablosunu oluşturur/günceller."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tablonun var olup olmadığını ve doc_name sütununun olup olmadığını kontrol et
    cursor.execute("PRAGMA table_info(chunks)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'doc_name' not in columns:
        # Eğer tablo eski şemadaysa yeniden temiz oluştur
        cursor.execute('DROP TABLE IF EXISTS chunks')
        cursor.execute('''
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT,
                content TEXT,
                embedding TEXT
            )
        ''')
    conn.commit()
    conn.close()

def chunk_text(text, chunk_size=250, overlap=40):
    """Metni çakışmalı (overlapping) parçalara böler."""
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        if text.strip():
            chunks.append(text.strip())
        return chunks

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def process_documents():
    """docs/ klasöründeki tüm .txt ve .md dosyalarını okur, vektörleştirir ve SQLite'a yazar."""
    init_db()

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"'{DOCS_DIR}' klasoru olusturuldu. Lutfen icine .txt veya .md uzantili dokumanlar ekleyin!")
        return

    supported_extensions = ('.txt', '.md')
    files_to_process = [
        f for f in os.listdir(DOCS_DIR)
        if f.lower().endswith(supported_extensions)
    ]

    if not files_to_process:
        print(f"'{DOCS_DIR}' klasorunde islenecek .txt veya .md dosyasi bulunamadi!")
        return

    print("Yerel Embedding modeli yukleniyor ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Mevcut kayıtları temizle (temiz indeksleme)
    cursor.execute('DELETE FROM chunks')

    total_chunks = 0
    print(f"\nToplam {len(files_to_process)} dokuman isleniyor...")

    for file_name in files_to_process:
        file_path = os.path.join(DOCS_DIR, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = chunk_text(text)
            for chunk in chunks:
                embedding_vector = model.encode(chunk).tolist()
                cursor.execute(
                    'INSERT INTO chunks (doc_name, content, embedding) VALUES (?, ?, ?)',
                    (file_name, chunk, json.dumps(embedding_vector))
                )
                total_chunks += 1
            print(f" - [{file_name}]: {len(chunks)} parca indekslendi.")
        except Exception as e:
            print(f" [HATA] [{file_name}] islenirken hata olustu: {e}")

    conn.commit()
    conn.close()
    print(f"\nVeri yukleme basariyla tamamlandi! Toplam {total_chunks} parca SQLite veritabani tablosuna kaydedildi.\n")

if __name__ == '__main__':
    process_documents()