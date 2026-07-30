import streamlit as st
import sqlite3
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from foundry_local_sdk import FoundryLocalManager, Configuration

DB_PATH = 'knowledge_base.db'

st.set_page_config(
    page_title="Yerel RAG Asistanı",
    page_icon="🤖",
    layout="wide"
)

# Arayüz Başlığı ve Açıklaması
st.title("🤖 Offline Türkçe RAG Bilgi Asistanı")
st.caption("Microsoft Foundry Local, SQLite ve SentenceTransformers ile çalışan çevrimdışı Q&A sistemi.")

# Kenar Çubuğu (Sidebar) Yapılandırması
with st.sidebar:
    st.header("⚙️ Ayarlar & Durum")
    
    # Veritabanı İstatistikleri
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT COUNT(*), COUNT(DISTINCT doc_name) FROM chunks')
            total_chunks, total_docs = cursor.fetchone()
            st.success(f"📊 **İndekslenmiş Veri**: {total_docs} Doküman, {total_chunks} Parça")
        except Exception:
            st.warning("⚠️ Veritabanı henüz indekslenmemiş. Lütfen `ingestion.py` dosyasını çalıştırın.")
        finally:
            conn.close()
    else:
        st.error("❌ Veritabanı bulunamadı. Lütfen önce `ingestion.py` betiğini çalıştırın.")

    selected_model_alias = st.selectbox(
        "🤖 Yerel LLM Modeli",
        options=["qwen2.5-0.5b", "qwen2.5-1.5b", "phi-4-mini", "qwen2.5-7b"],
        index=0
    )

    top_k = st.slider("Aramada Getirilecek Parça Sayısı (Top-K)", min_value=1, max_value=5, value=3)
    
    if st.button("🗑️ Sohbet Geçmişini Temizle"):
        st.session_state.messages = []
        st.rerun()

# Modellerin ve İstemcinin Yüklenmesi
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_foundry_manager():
    try:
        config = Configuration("local_rag_assistant")
        manager = FoundryLocalManager(config)
        manager.start_web_service()
        return manager
    except Exception as e:
        return None

def ensure_model_loaded(manager, model_alias):
    """Seçilen modelin Foundry Local üzerinde indirildiğinden ve hafızaya yüklendiğinden emin olur."""
    if manager is None:
        return False, None
    try:
        model = manager.catalog.get_model(model_alias)
        if not model.is_cached:
            st.toast(f"⏳ '{model_alias}' modeli indiriliyor (ilk çalıştırma)...")
            model.download()
        if not model.is_loaded:
            st.toast(f"⚙️ '{model_alias}' modeli belleğe yükleniyor...")
            model.load()
        
        service_url = manager.urls[0] if manager.urls else "http://127.0.0.1:51734"
        base_url = f"{service_url.rstrip('/')}/v1"
        client = OpenAI(base_url=base_url, api_key="foundry")
        return True, client
    except Exception as e:
        return False, str(e)

embedder = load_embedder()
manager = get_foundry_manager()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def get_relevant_context(query, top_k=3):
    """Kullanıcının sorusuna en benzer doküman parçalarını getirir."""
    if not os.path.exists(DB_PATH):
        return "", []

    q_emb = embedder.encode(query).tolist()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT doc_name, content, embedding FROM chunks')
        rows = cursor.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    if not rows:
        return "", []

    results = []
    for doc_name, content, emb_str in rows:
        emb = json.loads(emb_str)
        sim = cosine_similarity(q_emb, emb)
        results.append({
            "doc_name": doc_name,
            "content": content,
            "score": sim
        })

    # Skora göre sırala
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k]

    # Model için bağlam metni hazırla
    context_blocks = []
    for item in top_results:
        context_blocks.append(f"[Kaynak: {item['doc_name']}]\n{item['content']}")
    
    formatted_context = "\n\n---\n\n".join(context_blocks)
    return formatted_context, top_results

# Chat Oturum Durumu Initialisation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş Mesajları Ekrana Basma
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Kullanılan Kaynaklar ve Benzerlik Skorları"):
                for src in msg["sources"]:
                    st.markdown(f"**📄 {src['doc_name']}** *(Benzerlik: %{src['score']*100:.1f})*")
                    st.caption(f"> {src['content']}")

# Kullanıcı Soru Girdisi
if prompt := st.chat_input("Ders notlarınız veya dokümanlarınızla ilgili bir soru sorun..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Yanıt Üretme Süreci
    with st.chat_message("assistant"):
        with st.spinner("🔍 Dokümanlar taranıyor ve yerel AI yanıt üretiyor..."):
            context, sources = get_relevant_context(prompt, top_k=top_k)
            
            # Minimum alaka skoru eşiği (örneğin %20 altında kalırsa bağlam yetersizdir)
            top_score = sources[0]["score"] if sources else 0.0
            
            if top_score < 0.20:
                answer = "Bu bilgi yüklenen dokümanlarda yer almamaktadır."
            else:
                success, client_or_err = ensure_model_loaded(manager, selected_model_alias)
                
                if success:
                    client = client_or_err
                    system_prompt = f"""Sen akıcı, saygılı ve düzgün Türkçe konuşan uzman bir asistansın.
Sana verilen İçerik bilgisini dikkatlice incele ve kullanıcının sorusunu YALNIZCA bu içerikteki bilgilere dayanarak Türkçe yanıtla.

Kurallar:
1. Yanıtında doğruluktan ayrılma. İçerikte doğrudan geçmeyen detayları uydurma.
2. Eğer aranan bilgi içerikte kesin olarak geçmiyorsa tam olarak şu yanıtı ver: 'Bu bilgi yüklenen dokümanlarda yer almamaktadır.'
3. Mümkünse yanıtının sonunda bilgiyi hangi kaynak dokümandan aldığını belirt.

İçerik:
{context}"""
                    try:
                        response = client.chat.completions.create(
                            model=selected_model_alias,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        answer = response.choices[0].message.content
                    except Exception as e:
                        # LLM üretimi esnasında hata olursa bulunan en alakalı doküman parçası sunulur
                        answer = f"**[RAG Doğrudan Bilgi Bağlamı]**\n\n{sources[0]['content']}"
                else:
                    # Model henüz yüklenmediyse doğrudan RAG dokümanından yanıt basılır
                    answer = f"**[Yerel Doküman Yanıtı]**\n\n{sources[0]['content']}"

            st.write(answer)
            
            if sources:
                with st.expander("📚 Kullanılan Kaynaklar ve Benzerlik Skorları"):
                    for src in sources:
                        st.markdown(f"**📄 {src['doc_name']}** *(Benzerlik: %{src['score']*100:.1f})*")
                        st.caption(f"> {src['content']}")

    # Asistan mesajını geçmişe sakla
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })