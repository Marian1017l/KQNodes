from typing import List

import numpy as np
import time

from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_STRAREGY_TAG
from typing import Callable
from src.funcs.iit import seleccionar_emd
from src.funcs.format import fmt_biparticion_fuerza_bruta
from src.constants.base import ACTUAL, EFFECT
from src.constants.models import GEOMETRIC_LABEL


class GeometricSIA(SIA):
    """
    Estrategia geométrica para encontrar la bipartición de mínima información (MIP).
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self.n: int = 0
        self.m: int = 0
        self.distancia_metrica: Callable = seleccionar_emd()
        self.tensors: List[np.ndarray] = []
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        self._representacion_inicial()

        tabla      = self._construir_tabla_costos()
        candidatos = self._identificar_candidatos(tabla)
        phi, dist, particion = self._evaluar_candidatos(candidatos)

        return Solution(
            estrategia              = GEOMETRIC_LABEL,
            perdida                 = phi,
            distribucion_subsistema = self.sia_dists_marginales,
            distribucion_particion  = dist,
            particion               = particion,
            tiempo_total            = time.time() - self.sia_tiempo_inicio,
        )
    
    def _representacion_inicial(self) -> None:
        """
        Construye la representación tensorial del subsistema.

        Para cada n-cubo del subsistema aplana sus datos en un vector 1-D.
        Estos vectores representan geométricamente cada nodo en el espacio
        de probabilidad y sirven de base para calcular distancias entre nodos.
        """
        self.n = self.sia_subsistema.indices_ncubos.size
        self.m = self.sia_subsistema.dims_ncubos.size
        self.tensors = [
            self._tensor_de(i) for i in range(self.n)
        ]
    
    def _tensor_de(self, var_index:int) -> np.ndarray:
        ncube = self.sia_subsistema.ncubos[var_index]
        return ncube.data.flatten()

    def _hallar_candidatos(self):
        indices = self.sia_subsistema.indices_ncubos
        n = self.n
        for mask in range(1, (1 << n) // 2 + 1):
            subalcance = tuple(
                int(indices[i]) for i in range(n) if (mask >> i) & 1
            )
            submecanismo = tuple(
                int(indices[j]) for j in range(n) if not ((mask >> j) & 1)
            )
            yield subalcance, submecanismo
    
    def _calcular_costo_transicion(self, i: int, j: int, tensor: np.ndarray) -> float:
        """
        Calcula t_x(i,j) según la fórmula recursiva de GeoMIP (Algorithm 1).

        t_x(i,j) = y · ( |X[i] - X[j]| + Σ_{k ∈ N(i,j)} t_x(k, j) )
        y = 2^(-d_H(i,j))
        N(i,j) = vecinos v con d_H(i,v)=1 y d_H(v,j) < d_H(i,j)
        """
        if i == j:
            return 0.0

        d = bin(i ^ j).count('1')        # distancia Hamming
        gamma = 2.0 ** (-d)

        costo_directo = abs(float(tensor[i]) - float(tensor[j]))

        costo_vecinos = 0.0
        for bit in range(self.m):
            v = i ^ (1 << bit)           # flip de un bit
            if bin(v ^ j).count('1') < d:  # v está más cerca de j → es vecino válido
                costo_vecinos += self._calcular_costo_transicion(v, j, tensor)

        return gamma * (costo_directo + costo_vecinos)
    
    def _construir_tabla_costos(self) -> List[np.ndarray]:
        """
        Calcula costos solo para j = estado actual del mecanismo.
        Reduce complejidad de O(n·S²·m) a O(n·S·m).
        """
        S        = 1 << self.m
        popcount = np.array([bin(x).count('1') for x in range(S)], dtype=np.int8)
        estados  = np.arange(S, dtype=np.int32)

        # Estado actual del mecanismo como índice entero (little-endian)
        dims   = self.sia_subsistema.dims_ncubos
        estado = self.sia_subsistema.estado_inicial
        self._j_actual = int(sum(
            int(estado[d]) * (1 << local)
            for local, d in enumerate(dims)
        ))

        tabla: List[np.ndarray] = []
        for x in range(self.n):
            costos_j = np.zeros(S, dtype=np.float64)
            tensor   = self.tensors[x]
            j        = self._j_actual

            diff          = estados ^ j
            dist          = popcount[diff]
            costo_directo = np.abs(tensor - tensor[j])

            for d in range(1, self.m + 1):
                gamma    = 2.0 ** (-d)
                states_d = np.where(dist == d)[0]
                if states_d.size == 0:
                    continue
                diff_d        = diff[states_d]
                costo_vecinos = np.zeros(states_d.size, dtype=np.float64)
                for b in range(self.m):
                    tiene_bit     = (diff_d >> b) & 1
                    vecinos       = states_d ^ (1 << b)
                    costo_vecinos += tiene_bit * costos_j[vecinos]
                costos_j[states_d] = gamma * (costo_directo[states_d] + costo_vecinos)

            tabla.append(costos_j)
        return tabla

    def _identificar_candidatos(self, tabla: List[np.ndarray]) -> list:
        """
        Identifica biparticiones candidatas usando patrones de costo mínimo en T.

        Para cada variable x y cada estado j, el estado i con menor costo
        t_x(i,j) indica qué bits del mecanismo se pueden separar con menor pérdida.
        También agrega cortes totales (sub_mecanismo vacío) que el patrón de
        costo mínimo no detecta pero son biparticiones válidas.
        """
        indices = self.sia_subsistema.indices_ncubos  # variables de alcance (futuro)
        dims    = self.sia_subsistema.dims_ncubos     # variables de mecanismo (presente)
        candidatos = set()

        for x in range(self.n):
            T_x = tabla[x]
            for j in range(1 << self.m):
                costos = T_x[:, j].copy()
                costos[j] = np.inf                    # excluir i == j
                i_min = int(np.argmin(costos))

                mascara = i_min ^ j                   # bits que menos cuestan separar
                if mascara == 0:
                    continue

                sub_alcance   = (int(indices[x]),)
                sub_mecanismo = tuple(
                    int(dims[b]) for b in range(self.m) if (mascara >> b) & 1
                )

                if sub_alcance and sub_mecanismo:
                    candidatos.add((sub_alcance, sub_mecanismo))

            # Corte total: variable x sin ningún presente (sub_mecanismo vacío)
            candidatos.add(((int(indices[x]),), ()))

        return list(candidatos)
    
    def _evaluar_candidatos(self, candidatos: list) -> tuple:
        """
        Evalúa cada bipartición candidata y retorna la de menor pérdida φ.
        Usa la EMD configurada en el proyecto (emd_efecto por defecto).
        """
        futuros  = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos

        mejor_phi  = np.inf
        mejor_dist = None
        mejor_fmt  = None

        for sub_alcance, sub_mecanismo in candidatos:
            arr_alcance   = np.array(sub_alcance,   dtype=np.int8)
            arr_mecanismo = np.array(sub_mecanismo, dtype=np.int8)

            particion      = self.sia_subsistema.bipartir(arr_alcance, arr_mecanismo)
            dist_particion = particion.distribucion_marginal()
            phi            = self.distancia_metrica(dist_particion, self.sia_dists_marginales)

            if phi < mejor_phi:
                mejor_phi  = phi
                mejor_dist = dist_particion

                biparticion_prim = (sub_mecanismo, sub_alcance)
                biparticion_dual = (
                    tuple(set(presentes.tolist()) - set(sub_mecanismo)),
                    tuple(set(futuros.tolist())   - set(sub_alcance)),
                )
                mejor_fmt = fmt_biparticion_fuerza_bruta(
                    [biparticion_prim[ACTUAL], biparticion_prim[EFFECT]],
                    [biparticion_dual[ACTUAL], biparticion_dual[EFFECT]],
                )

        return mejor_phi, mejor_dist, mejor_fmt


        

