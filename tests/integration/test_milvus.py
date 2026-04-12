"""
Test simple y directo de Milvus Lite
"""
import os
import sys

print("=" * 60)
print("🧪 TEST DE MILVUS LITE")
print("=" * 60)

# 1. Test de PyMilvus
print("\n1️⃣ Verificando PyMilvus...")
try:
    from pymilvus import MilvusClient
    print("   ✅ PyMilvus importado correctamente")
except ImportError as e:
    print(f"   ❌ Error: {e}")
    print("   Ejecuta: pip install pymilvus==2.4.4")
    sys.exit(1)

# 2. Crear cliente con URI correcto
print("\n2️⃣ Creando cliente Milvus Lite...")
try:
    # Probar diferentes formas de URI
    uris_to_try = [
        "./milvus_test.db",
        "milvus_test.db",
        os.path.abspath("milvus_test.db")
    ]
    
    client = None
    working_uri = None
    
    for uri in uris_to_try:
        try:
            print(f"   Probando URI: {uri}")
            client = MilvusClient(uri=uri)
            working_uri = uri
            print(f"   ✅ Conectado con URI: {uri}")
            break
        except Exception as e:
            print(f"   ❌ Falló con {uri}: {e}")
            continue
    
    if client is None:
        raise Exception("No se pudo conectar con ningún URI")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 3. Crear colección de prueba
print("\n3️⃣ Creando colección...")
try:
    collection_name = "test_collection"
    
    # Eliminar si existe
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"   🗑️ Colección anterior eliminada")
    
    # Crear nueva
    client.create_collection(
        collection_name=collection_name,
        dimension=768,
        auto_id=True,
        enable_dynamic_field=True
    )
    print(f"   ✅ Colección '{collection_name}' creada")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 4. Insertar datos de prueba
print("\n4️⃣ Insertando datos de prueba...")
try:
    import numpy as np
    
    test_data = [
        {
            "title": "Test Article 1",
            "abstract": "This is a test abstract",
            "embedding": np.random.rand(768).tolist(),
            "doi": "10.1234/test1"
        },
        {
            "title": "Test Article 2", 
            "abstract": "Another test abstract",
            "embedding": np.random.rand(768).tolist(),
            "doi": "10.1234/test2"
        }
    ]
    
    result = client.insert(collection_name=collection_name, data=test_data)
    print(f"   ✅ Insertados {len(test_data)} registros")
    print(f"   IDs: {result.get('ids', 'N/A')}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 5. Buscar datos
print("\n5️⃣ Probando búsqueda...")
try:
    query_vector = np.random.rand(768).tolist()
    
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=2,
        output_fields=["title", "abstract", "doi"]
    )
    
    print(f"   ✅ Búsqueda exitosa")
    for i, hits in enumerate(results):
        print(f"   📄 Resultado {i+1}:")
        for hit in hits:
            entity = hit.get("entity", {})
            print(f"      - {entity.get('title', 'N/A')}")
            print(f"        Score: {hit.get('distance', 0):.3f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 6. Verificar archivo creado
print("\n6️⃣ Verificando archivo...")
try:
    if os.path.exists(working_uri):
        size = os.path.getsize(working_uri) / 1024
        print(f"   ✅ Archivo creado: {working_uri}")
        print(f"   📁 Tamaño: {size:.2f} KB")
    else:
        print(f"   ⚠️ Archivo no encontrado: {working_uri}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 7. Limpiar
print("\n7️⃣ Limpiando...")
try:
    client.drop_collection(collection_name)
    print(f"   ✅ Colección eliminada")
    
    if os.path.exists("milvus_test.db"):
        os.remove("milvus_test.db")
        print(f"   ✅ Archivo de prueba eliminado")
except Exception as e:
    print(f"   ⚠️ No se pudo limpiar: {e}")

print("\n" + "=" * 60)
print("✅ TEST DE MILVUS COMPLETADO EXITOSAMENTE")
print("=" * 60)
print("\nAhora puedes ejecutar:")
print("  python test_flow.py")
print("  uvicorn main:app --reload")