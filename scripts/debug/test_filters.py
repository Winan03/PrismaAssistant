import requests
import json

url = "http://127.0.0.1:8000/update_filter_count"
data = {
    "session_id": "1235292",
    "start_year": "2021",
    "end_year": "2026",
    "open_access": "true",
    "academic_quality": "false"
}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        res_json = response.json()
        print("\n--- RESULTADOS DE LA PRUEBA (SESIÓN 1235292) ---")
        print(f"Total Encontrados (raw_total): {res_json.get('raw_total')}")
        print(f"Únicos (unique_records): {res_json.get('unique_records')}")
        print(f"Finales (final_count): {res_json.get('final_count')}")
        print(f"Excluidos por Años: {res_json.get('metadata_discarded')}")
        print(f"Artículos en la lista filtrada: {res_json.get('filtered_count')}")
        
        raw_total = res_json.get('raw_total', 0)
        if raw_total == 0 or raw_total is None:
            print("\n❌ FALLO DETECTADO: El servidor sigue devolviendo ceros.")
        else:
            print("\n✅ PRUEBA EXITOSA: El servidor devuelve datos reales y consistentes.")
            print("Ya puedes usar los filtros con total seguridad.")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error de conexión: {e}")
