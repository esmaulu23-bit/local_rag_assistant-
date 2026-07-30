import os
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np

DB_PATH = 'knowledge_base.db'

def test_vector_search():
    print("=" * 50)
    print("[TEST] Yerel RAG Vektor Arama Testi Baslatiliyor...")
    print("=" * 50)

    if not os.path.exists(DB_PATH):
        print("[HATA] 'knowledge_base.db' veritabani bulunamadi. Onolarak ingestion.py calistirilmalidir!")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*), COUNT(DISTINCT doc_name) FROM chunks')
    total_chunks, total_docs = cursor.fetchone()
    print(f"[OK] SQLite Veritabani Baglantisi Basarili:")
    print(f"   - Indekslenmis Dokuman Sayisi: {total_docs}")
    print(f"   - Toplam Metin Parcasi (Chunk): {total_chunks}")

    if total_chunks == 0:
        print("[HATA] Veritabaninda hic metin parcasi yok!")
        conn.close()
        return False

    print("\n[INFO] Embedding Modeli Yukleniyor...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    test_queries = [
        "Microsoft Foundry Local nedir?",
        "RAG mimarisi ne ise yarar?",
        "Hallucination nasil engellenir?",
        "Ankara hava durumu nasil?"
    ]

    cursor.execute('SELECT doc_name, content, embedding FROM chunks')
    rows = cursor.fetchall()
    conn.close()

    for query in test_queries:
        print(f"\n[SORU] Soru: '{query}'")
        q_emb = model.encode(query)
        
        scores = []
        for doc_name, content, emb_str in rows:
            emb = np.array(json.loads(emb_str))
            sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
            scores.append((sim, doc_name, content))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_match = scores[0]
        
        print(f"   -> En Yakin Eslesme (Skor: %{top_match[0]*100:.1f}):")
        print(f"      Dokuman: {top_match[1]}")
        print(f"      Icerik Ozeti: {top_match[2][:120]}...")

    print("\n" + "=" * 50)
    print("[SUCCESS] Tum Testler Basariyla Tamamlandi!")
    print("=" * 50)
    return True

if __name__ == '__main__':
    test_vector_search()
