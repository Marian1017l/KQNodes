from typing import List

import numpy as np
import time

from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_STRAREGY_TAG
from typing import Callable
from src.funcs.iit import seleccionar_emd, count_bits, hamming_distance
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
        return ncube.data.flatten().astype(np.float64)
    
    def _construir_tabla_costos(self) -> List[np.ndarray]:
        """
        Calcula costos solo para j = estado actual del mecanismo.
        """
        S        = 1 << self.m
        popcount = np.array([count_bits(x) for x in range(S)], dtype=np.int8)
        estados  = np.arange(S, dtype=np.int32)

        # Estado actual del mecanismo como índice entero (little-endian)
        dims   = self.sia_subsistema.dims_ncubos
        estado = self.sia_subsistema.estado_inicial
        self._j_actual = int(sum(
            int(estado[d]) * (1 << local)
            for local, d in enumerate(dims)
        ))

        tabla: List[np.ndarray] = []
        pos_global = {d: i for i, d in enumerate(dims)}
        for x in range(self.n):
            costos_j = np.zeros(S, dtype=np.float64)
            tensor   = self.tensors[x]
            j        = self._j_actual

            ncubo = self.sia_subsistema.ncubos[x]
            dims_local = ncubo.dims

            j_local = 0
            for pos_local, d in enumerate(dims_local):
                bit = (j >> pos_global[d]) & 1
                j_local |= (bit << pos_local)

            # Proyección de todos los estados globales al espacio local del n-cubo x.
            # Reutiliza el mismo patrón bit a bit que j_local, vectorizado sobre estados.
            # Necesario cuando |dims_local| < m: tensor vive en 2^|dims_local|,
            # pero costos_j y states_d operan en el espacio global 2^m.
            estados_local = np.zeros(S, dtype=np.int32)
            for pos_local, d in enumerate(dims_local):
                bit_col        = (estados >> pos_global[d]) & 1
                estados_local |= (bit_col << pos_local)

            diff          = estados ^ j
            dist          = popcount[diff]
            costo_directo = np.abs(tensor[estados_local] - tensor[j_local])

            for d in range(1, self.m + 1):
                gamma    = 2.0 ** (-d)
                states_d = np.where(dist == d)[0]
                if states_d.size == 0:
                    continue
                costo_vecinos = np.zeros(states_d.size, dtype=np.float64)
                for b in range(self.m):
                    vecinos = states_d ^ (1 << b)
                    mask_validos = (popcount[vecinos ^ j] == d - 1)
                    costo_vecinos += mask_validos * costos_j[vecinos]
                costos_j[states_d] = gamma * (costo_directo[states_d] + costo_vecinos)
            tabla.append(costos_j)

        return tabla

    def _identificar_candidatos(self, tabla: List[np.ndarray]) -> list:
        """
        Selecciona candidatos de bipartición a partir de los estados
        con menor costo de transición hacia j_actual.
        """
        indices    = self.sia_subsistema.indices_ncubos
        dims       = self.sia_subsistema.dims_ncubos
        j          = self._j_actual
        candidatos = set()

        for x in range(self.n):
            costos    = tabla[x].copy()
            costos[j] = np.inf

           # Encontrar estados con costo mínimo (con tolerancia numérica)
            costo_min   = costos.min()
            estados_min = np.where(
                np.isclose(costos, costo_min, rtol=1e-9, atol=1e-15)
            )[0]

            # Si hay demasiados empates → quedarse con los más cercanos (Hamming)
            if estados_min.size > self.m + 1:
                hamming     = np.array(
                    [bin(int(i) ^ j).count('1') for i in estados_min], dtype=np.int32
                )
                estados_min = estados_min[hamming == hamming.min()]

            # Convertir estados a particiones usando XOR
            for i_cand in estados_min:
                mascara = int(i_cand) ^ j
                if mascara == 0:
                    continue
                sub_alcance   = (int(indices[x]),)
                sub_mecanismo = tuple(
                    int(dims[b]) for b in range(self.m) if (mascara >> b) & 1
                )
                if sub_alcance and sub_mecanismo:
                    candidatos.add((sub_alcance, sub_mecanismo))

            # Corte total: variable x sin ningún presente
            candidatos.add(((int(indices[x]),), ()))

            # Pares simples: variable x vs cada variable del mecanismo
            for b in range(self.m):
                candidatos.add(((int(indices[x]),), (int(dims[b]),)))

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

            #print("Candidato:", sub_alcance, sub_mecanismo, "phi=", phi)
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