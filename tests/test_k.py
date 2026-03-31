import numpy as np

from src.controllers.manager import Manager
from src.funcs.force import generar_k_particiones
from src.funcs.iit import emd_efecto
from src.models.core.system import System


# ── Cargar TPM de 3 nodos ────────────────────────────────────────────────────
gestor = Manager(estado_inicial="100")
tpm = gestor.cargar_red()

estado_inicial = np.array([1, 0, 0], dtype=np.int8)
sistema = System(tpm, estado_inicial)

dist_completa = sistema.distribucion_marginal()
print(f"Distribución completa:      {dist_completa}\n")


# ── Probar k_partir con asignaciones separadas ───────────────────────────────
# futuro:  nodo 0 → grupo 0, nodo 1 → grupo 2, nodo 2 → grupo 1
# presente: nodo 0 → grupo 1, nodo 1 → grupo 0, nodo 2 → grupo 2
asig_futuro  = (0, 2, 1)
asig_presente = (1, 0, 2)
particion_k = sistema.k_partir(asig_futuro, asig_presente)
dist_k = particion_k.distribucion_marginal()
phi_k = emd_efecto(dist_k, dist_completa)

print(f"Asignación futuro={asig_futuro}, presente={asig_presente}:")
print(f"  Distribución particionada: {dist_k}")
print(f"  φ = {phi_k:.6f}\n")


# ── Verificar consistencia: k=2 debe coincidir con bipartir ──────────────────
# bipartir(alcance=[0,2], mecanismo=[1]) significa:
#   futuro:   nodos 0,2 en grupo 0  → nodo 1 en grupo 1   → asig_futuro  = (0,1,0)
#   presente: nodo 1 en grupo 0     → nodos 0,2 en grupo 1 → asig_presente = (1,0,1)
alcance   = np.array([0, 2], dtype=np.int8)
mecanismo = np.array([1],    dtype=np.int8)

dist_bipartir = sistema.bipartir(alcance, mecanismo).distribucion_marginal()
dist_k2       = sistema.k_partir((0, 1, 0), (1, 0, 1)).distribucion_marginal()

coinciden = np.allclose(dist_bipartir, dist_k2)
print(f"bipartir([0,2],[1])              → {dist_bipartir}")
print(f"k_partir((0,1,0), (1,0,1))      → {dist_k2}")
print(f"¿Coinciden? {'✓' if coinciden else '✗'}\n")


# ── Recorrer todas las k-particiones con k=3 y mostrar top 3 por φ ───────────
print("Top 3 k-particiones (k=3) por menor φ:")
resultados = []
for asig_f, asig_p in generar_k_particiones(num_elementos=3, k=3):
    dist_p = sistema.k_partir(asig_f, asig_p).distribucion_marginal()
    phi    = emd_efecto(dist_p, dist_completa)
    resultados.append((phi, asig_f, asig_p, dist_p))

resultados.sort(key=lambda x: x[0])
for phi, asig_f, asig_p, dist in resultados[:3]:
    print(f"  futuro={asig_f} presente={asig_p}  dist={np.round(dist, 4)}  φ={phi:.6f}")
