from modules.ai_model import LocalModel

model = LocalModel.get_instance()

# TAREA: Generar un título y objetivos para una tesis
prompt_input = """
Tema: Uso de Inteligencia Artificial para optimizar revisiones sistemáticas (PRISMA).
Objetivo: Crear un título tentativo y 2 objetivos específicos.
"""

response = model.generate(
    instruction="Eres un asesor de tesis metodológico experto en normativa UPAO.",
    input_text=prompt_input,
    max_tokens=500  # Subimos a 500 para que no se corte
)

print("-" * 50)
print("🧠 RESPUESTA DEL MODELO:")
print("-" * 50)
print(response)
print("-" * 50)