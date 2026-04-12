"""
Test de conexión a MongoDB Atlas
"""
from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

print("=" * 60)
print("🔧 Test de MongoDB Atlas")
print("=" * 60)

uri = os.getenv("MONGODB_URI")
print(f"\n📋 URI: {uri[:50]}...")

try:
    print("\n🔗 Intentando conectar...")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    
    print("✅ Cliente creado")
    
    # Test de conexión
    info = client.server_info()
    print(f"✅ Conectado a MongoDB v{info.get('version')}")
    
    # Listar databases
    dbs = client.list_database_names()
    print(f"\n📚 Databases disponibles: {dbs}")
    
    # Intentar acceder a la base de datos
    db = client.get_database()
    print(f"✅ Database seleccionada: {db.name}")
    
    # Listar colecciones
    collections = db.list_collection_names()
    print(f"📁 Colecciones: {collections if collections else 'Ninguna (se creará automáticamente)'}")
    
    print("\n" + "=" * 60)
    print("✅ MongoDB funciona correctamente")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n🔧 Posibles soluciones:")
    print("   1. Verifica usuario y contraseña en MongoDB Atlas")
    print("   2. Ve a Network Access y agrega tu IP")
    print("   3. Verifica que el usuario tenga permisos de lectura/escritura")
    print("   4. Asegúrate de que no haya caracteres especiales sin escapar")
    print("\n💡 El sistema puede funcionar sin MongoDB usando solo Milvus")