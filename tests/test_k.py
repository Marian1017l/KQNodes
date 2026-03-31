from src.funcs.force import generar_k_particiones

print("Probando k-particiones...\n")

for p in generar_k_particiones(3, 3):
    print(p)