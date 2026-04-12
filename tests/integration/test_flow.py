"""
Test del flujo completo de PRISMA Assistant
"""
import sys
import os
import time

print("=" * 70)
print("🧪 TEST DEL FLUJO COMPLETO - PRISMA ASSISTANT")
print("=" * 70)

# ==========================
# 1. Test de Configuración
# ==========================
print("\n📋 1. Cargando configuración...")
try:
    import config
    print(f"   ✅ OpenRouter Model: {config.OPENROUTER_MODEL}")
    print(f"   ✅ Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"   ✅ Milvus URI: {config.MILVUS_URI}")
    print(f"   ✅ Similarity Threshold: {config.SIMILARITY_RELEVANT}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# ==========================
# 2. Test de Query Expansion
# ==========================
print("\n🤖 2. Probando expansión de consulta...")
try:
    from utils.query_expander import expand_query
    
    test_question = "¿Cuál es la efectividad de la IA en diagnóstico médico?"
    terms = expand_query(test_question, max_terms=5)
    
    print(f"   📝 Pregunta: {test_question}")
    print(f"   ✅ Términos expandidos ({len(terms)}): {terms[:5]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ==========================
# 3. Test de Búsqueda
# ==========================
print("\n🔍 3. Probando búsqueda de artículos...")
try:
    from modules import search_engine
    
    # Buscar solo 10 artículos para test rápido
    articles, t_search = search_engine.search_articles(terms[:3], max_results=20)
    
    print(f"   ✅ Artículos encontrados: {len(articles)}")
    print(f"   ⏱️ Tiempo: {t_search:.2f}s")
    
    if articles:
        print(f"   📄 Ejemplo: {articles[0]['title'][:60]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    articles = []

# ==========================
# 4. Test de Filtros PRISMA
# ==========================
print("\n⚙️ 4. Probando filtros PRISMA...")
try:
    from modules import filters
    
    filtered = filters.apply_filters(
        articles,
        start_year=2020,
        end_year=2025,
        language='en'
    )
    
    print(f"   ✅ Inicial: {len(articles)}")
    print(f"   ✅ Filtrados: {len(filtered)}")
    print(f"   📊 Excluidos: {len(articles) - len(filtered)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    filtered = articles

# ==========================
# 5. Test de Deduplicación
# ==========================
print("\n🔁 5. Probando deduplicación...")
try:
    from modules import deduplication
    
    dedup, removed = deduplication.remove_duplicates(filtered)
    
    print(f"   ✅ Antes: {len(filtered)}")
    print(f"   ✅ Después: {len(dedup)}")
    print(f"   📊 Duplicados eliminados: {removed}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    dedup = filtered

# ==========================
# 6. Test de Screening
# ==========================
print("\n🧠 6. Probando screening por relevancia...")
try:
    from modules import screening
    
    relevant = screening.screen_articles(dedup[:10], test_question)
    
    print(f"   ✅ Analizados: {len(dedup[:10])}")
    print(f"   ✅ Relevantes: {len(relevant)}")
    
    if relevant:
        print(f"   🎯 Mejor match: {relevant[0]['title'][:50]}...")
        print(f"      Similitud: {relevant[0].get('similarity', 0):.3f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    relevant = []

# ==========================
# 7. Test de Milvus
# ==========================
print("\n💾 7. Probando Milvus Lite...")
try:
    from modules import database
    
    if relevant:
        database.save_to_milvus(relevant)
        print(f"   ✅ Vectores guardados en: {config.MILVUS_URI}")
        
        # Verificar archivo creado
        if os.path.exists(config.MILVUS_URI):
            size = os.path.getsize(config.MILVUS_URI) / 1024
            print(f"   📁 Tamaño del archivo: {size:.2f} KB")
    else:
        print(f"   ⚠️ Sin artículos relevantes para guardar")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ==========================
# 8. Test de RAG
# ==========================
print("\n🎯 8. Probando recuperación RAG...")
try:
    from modules import rag_pipeline
    
    rag_results = rag_pipeline.retrieve_relevant(test_question, top_k=5)
    
    print(f"   ✅ Artículos recuperados: {len(rag_results)}")
    
    if rag_results:
        print(f"   📄 Top result: {rag_results[0]['title'][:50]}...")
        print(f"      Score: {rag_results[0].get('score', 0):.3f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    rag_results = []

# ==========================
# 9. Test de Síntesis
# ==========================
print("\n📝 9. Probando síntesis con IA...")
try:
    from modules import synthesis
    
    if rag_results:
        synth = synthesis.generate_synthesis(rag_results[:3], test_question)
        
        print(f"   ✅ Síntesis generada ({len(synth)} caracteres)")
        print(f"   📄 Preview: {synth[:150]}...")
    else:
        print(f"   ⚠️ Sin artículos para sintetizar")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ==========================
# 10. Test de MongoDB
# ==========================
print("\n🗄️ 10. Probando MongoDB (opcional)...")
try:
    if config.MONGODB_URI:
        database.save_to_mongo(relevant if relevant else [])
        print(f"   ✅ MongoDB disponible")
    else:
        print(f"   ⚠️ MongoDB deshabilitado (no crítico)")
except Exception as e:
    print(f"   ⚠️ MongoDB no disponible: {e}")

# ==========================
# RESUMEN FINAL
# ==========================
print("\n" + "=" * 70)
print("✅ TEST COMPLETO FINALIZADO")
print("=" * 70)
print("\nSi todos los pasos pasaron, ejecuta:")
print("   uvicorn main:app --reload")
print("\nLuego abre: http://127.0.0.1:8000")
print("=" * 70)