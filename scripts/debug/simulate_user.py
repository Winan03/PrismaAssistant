import requests
import time

url = "http://127.0.0.1:8000/update_filter_count"
session_id = "1235292"

def test_step(description, params):
    print(f"\n[STEP] {description}")
    params['session_id'] = session_id
    try:
        response = requests.post(url, data=params)
        if response.status_code == 200:
            res = response.json()
            raw = res.get('raw_total', 0)
            excl = res.get('metadata_discarded', 0)
            print(f"   -> Encontrados: {raw} | Excluidos: {excl}")
            if raw == 0:
                print("   [!] ERROR: Se perdieron los datos")
                return False
            return True
        else:
            print(f"   [!] ERROR HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"   [!] ERROR de conexion: {e}")
        return False

# ESCENARIOS DE BORRADO Y COMBINACIONES
scenarios = [
    ("Borrar Año Inicio (Vacio)", {"start_year": "", "end_year": "2026"}),
    ("Borrar Año Fin (Vacio)", {"start_year": "2021", "end_year": ""}),
    ("Borrar Ambos Años", {"start_year": "", "end_year": ""}),
    ("Solo Año Inicio (2025)", {"start_year": "2025", "end_year": ""}),
    ("Solo Año Fin (2020)", {"start_year": "", "end_year": "2020"})
]

print("=== SIMULANDO BORRADO DE CAMPOS (VALORES VACIOS) ===")
for desc, p in scenarios:
    if not test_step(desc, p):
        break
    time.sleep(0.5)
