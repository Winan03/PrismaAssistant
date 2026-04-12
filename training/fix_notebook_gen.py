with open('generate_notebook.py', 'rb') as f:
    raw = f.read()

lines = raw.split(b'\n')
for i, line in enumerate(lines):
    if b'test_input' in line and b'Extract' in line and b'ensayo' in line.lower():
        lines[i] = b'test_input = "Extrae [Diseno del ensayo]: Se realizo un ensayo controlado aleatorizado comparando dos intervenciones en 450 participantes durante 24 meses."\r'
        print(f'L{i+1}: reemplazado OK')

with open('generate_notebook.py', 'wb') as f:
    f.write(b'\n'.join(lines))
print('Guardado OK')
