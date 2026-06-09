import time
from typing import Union
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.iit import emd_efecto, ABECEDARY
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_parte_q
from src.models.base.sia import SIA

from src.models.core.solution import Solution
from src.constants.models import (
    KQNODES_ANALYSIS_TAG,
    KQNODES_LABEL,
    KQNODES_STRAREGY_TAG,
)
from src.constants.base import (
    COLS_IDX,
    INT_ZERO,
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    LAST_IDX,
    EFFECT,
    ACTUAL,
)
from src.models.base.application import aplicacion


class KQNodes(SIA):
    """Extiende QNodes a k-particiones mediante biparticiones sucesivas del grupo más grande."""

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        self.memoria_delta = {}
        self.memoria_union = {}
        self.memoria_grupo_candidato = {}
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
        self.logger = SafeLogger(KQNODES_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ):
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuro = tuple(
            (EFFECT, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )

        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        self.memoria_delta = {}
        self.memoria_union = {}
        self.memoria_grupo_candidato = {}

        phi_total, dist_mip, grupos = self._algorithm_k(vertices, k)
        fmt_particion = self._fmt_grupos(grupos)

        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=phi_total,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_mip,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_particion,
        )

    def _algorithm_k(self, vertices: list, k: int) -> tuple:
        """Biparticiones sucesivas hasta k grupos, subdividiendo siempre el más grande.

        phi_total = suma de φ de cada paso; mejor_dist = dist del paso con menor φ.
        """
        grupos = [list(vertices)]
        phi_total = 0.0
        mejor_phi = INFTY_POS
        mejor_dist = None

        for _ in range(k - 1):
            idx_mayor = max(range(len(grupos)), key=lambda i: len(grupos[i]))
            sub_vertices = grupos[idx_mayor]

            self.vertices = set(sub_vertices)
            self.memoria_grupo_candidato = {}

            mip = self.algorithm(sub_vertices)
            phi_mip = self.memoria_grupo_candidato[mip]
            phi_total += phi_mip

            clave = self.definir_clave(list(mip))
            particion = self.sia_subsistema.bipartir(
                np.array(clave[EFFECT], dtype=np.int8),
                np.array(clave[ACTUAL], dtype=np.int8),
            )
            dist = particion.distribucion_marginal()

            if phi_mip < mejor_phi:
                mejor_phi = phi_mip
                mejor_dist = dist

            mip_flat = KQNodes._flatten_grupo(list(mip))
            mip_set = set(mip_flat)
            complemento = [v for v in sub_vertices if v not in mip_set]

            grupos.pop(idx_mayor)
            grupos.append(mip_flat)
            grupos.append(complemento)

        if mejor_dist is None:
            mejor_dist = self.sia_dists_marginales

        return phi_total, mejor_dist, grupos

    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
    def algorithm(self, vertices: list[tuple[int, int]]):
        """Algoritmo Q: construye incrementalmente la bipartición de menor φ por submodularidad.

        Fases → ciclos → iteraciones. Cada fase agrega a omega el delta de menor costo marginal
        y registra la partición candidata en memoria_grupo_candidato. Retorna la clave de menor EMD.
        """
        for i in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = INFTY_POS

            for _ in range(len(deltas_ciclo) - 1):
                emd_local = 1e5
                indice_mip: int

                for k in range(len(deltas_ciclo)):
                    emd_union, emd_delta = self.funcion_submodular(
                        deltas_ciclo[k], omegas_ciclo
                    )
                    emd_iteracion = emd_union - emd_delta

                    if emd_iteracion < emd_local:
                        if emd_delta == INT_ZERO:
                            clave = (
                                tuple(deltas_ciclo[k])
                                if isinstance(deltas_ciclo[k], list)
                                else (deltas_ciclo[k],)
                            )
                            self.memoria_grupo_candidato[clave] = emd_delta
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = k
                        emd_particion_candidata = emd_delta

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            self.memoria_grupo_candidato[
                tuple(
                    deltas_ciclo[LAST_IDX]
                    if isinstance(deltas_ciclo[LAST_IDX], list)
                    else deltas_ciclo
                )
            ] = emd_particion_candidata

            par_candidato = (
                [omegas_ciclo[LAST_IDX]]
                if isinstance(omegas_ciclo[LAST_IDX], tuple)
                else omegas_ciclo[LAST_IDX]
            ) + (
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )

            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)

            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k],
        )

    def funcion_submodular(
        self, deltas: Union[tuple, list[tuple]], omegas: list[Union[tuple, list[tuple]]]
    ):
        """Calcula (emd_union, emd_delta): EMD de delta solo y de delta∪omega, con caché por bitmask."""
        # Delta #
        nodos_delta = [deltas] if isinstance(deltas, tuple) else deltas
        mask_da, mask_de = KQNodes._nodos_a_bitmask(nodos_delta)
        clave_delta = (mask_da, mask_de)
        dims_mecanismo_delta, idxs_alcance_delta = KQNodes._bitmask_a_indices(mask_da, mask_de)

        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(idxs_alcance_delta, dtype=np.int8),
                np.array(dims_mecanismo_delta, dtype=np.int8),
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
            self.memoria_delta[clave_delta] = emd_delta
            del vector_delta_marginal, particion_delta
        else:
            emd_delta = self.memoria_delta[clave_delta]

        # Unión #
        mask_ua, mask_ue = mask_da, mask_de
        for omega in omegas:
            nodos_omega = [omega] if isinstance(omega, tuple) else omega
            for tiempo, indice in nodos_omega:
                if tiempo == ACTUAL:
                    mask_ua |= 1 << int(indice)
                else:
                    mask_ue |= 1 << int(indice)
        clave_union = (mask_ua, mask_ue)
        dims_mecanismo_union, idxs_alcance_union = KQNodes._bitmask_a_indices(mask_ua, mask_ue)

        if clave_union not in self.memoria_union:
            particion_union = self.sia_subsistema.bipartir(
                np.array(idxs_alcance_union, dtype=np.int8),
                np.array(dims_mecanismo_union, dtype=np.int8),
            )
            vector_union_marginal = particion_union.distribucion_marginal()
            emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)
            self.memoria_union[clave_union] = emd_union
            del particion_union, vector_union_marginal
        else:
            emd_union = self.memoria_union[clave_union]

        return emd_union, emd_delta

    def _fmt_grupos(self, grupos: list[list]) -> str:
        """Formatea k grupos con fmt_parte_q, separados por ‖."""
        partes = [fmt_parte_q(g) for g in grupos]
        tops    = "‖".join(top    for top,    _ in partes)
        bottoms = "‖".join(bottom for _, bottom in partes)
        return f"{tops}\n{bottoms}\n"

    @staticmethod
    def definir_clave(
        conjuntos: list[Union[tuple[int, int], list[tuple[int, int]]]]
    ) -> tuple[tuple, tuple]:
        actual, efecto = [], []
        for conjunto in conjuntos:
            pares = [conjunto] if isinstance(conjunto, tuple) else conjunto
            for tiempo, indice in pares:
                (actual if tiempo == ACTUAL else efecto).append(indice)
        return tuple(sorted(actual)), tuple(sorted(efecto))

    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))

    @staticmethod
    def _flatten_grupo(grupo: list) -> list[tuple[int, int]]:
        """Aplana grupo anidado a lista plana de (tiempo, índice)."""
        result = []
        for item in grupo:
            if isinstance(item, tuple):
                result.append(item)
            else:
                result.extend(KQNodes._flatten_grupo(item))
        return result

    @staticmethod
    def _nodos_a_bitmask(nodos: list[tuple[int, int]]) -> tuple[int, int]:
        """Convierte [(tiempo, índice)] a (mask_actual, mask_efecto)."""
        mask_actual = 0
        mask_efecto = 0
        for tiempo, indice in nodos:
            if tiempo == ACTUAL:
                mask_actual |= 1 << indice
            else:
                mask_efecto |= 1 << indice
        return int(mask_actual), int(mask_efecto)

    @staticmethod
    def _bitmask_a_indices(mask_actual: int, mask_efecto: int) -> tuple[list[int], list[int]]:
        """Extrae índices de bits activos: retorna (indices_actual, indices_efecto)."""
        indices_actual = [i for i in range(mask_actual.bit_length()) if (mask_actual >> i) & 1]
        indices_efecto = [i for i in range(mask_efecto.bit_length()) if (mask_efecto >> i) & 1]
        return indices_actual, indices_efecto
