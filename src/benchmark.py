"""
Benchmark comparativo: QNodes vs KQNodes(k=2,3,4) sobre redes N3A–N6A.

Métricas recogidas por run:
  tiempo_ms       — duración de aplicar_estrategia (mejor de N_TRIALS)
  Δ hits/misses   — accesos a memoria_delta en funcion_submodular
  ∪ hits/misses   — accesos a memoria_union  en funcion_submodular
  bipartir        — llamadas totales a sia_subsistema.bipartir
  dist_mg         — llamadas totales a distribucion_marginal sobre resultados de bipartir

Los contadores se reinician en cada llamada a aplicar_estrategia; el timing
reportado es el mínimo de N_TRIALS para mayor estabilidad.

Ejecución:
    cd QNodes && python -m src.benchmark
"""

import time
from dataclasses import dataclass
from typing import Optional

from src.controllers.manager import Manager
from src.strategies.q_nodes import QNodes
from src.strategies.kq_nodes import KQNodes

# ─── Configuración ───────────────────────────────────────────────────────────

CASOS: list[tuple[str, str, str, str]] = [
    ("100",       "111",       "111",       "111"),       # N3A  — 6 vértices
    ("1000",      "1111",      "1111",      "1111"),      # N4A  — 8 vértices
    ("10000",     "11111",     "11111",     "11111"),     # N5A  — 10 vértices
    ("100000",    "111111",    "111111",    "111111"),    # N6A  — 12 vértices
    # ("10000000", "11111111", "11111111", "11111111"),  # N8A  — descomentar si disponible
]

K_VALORES: list[int] = [2, 3, 4]
N_TRIALS:  int       = 3   # runs por combinación; stats del último, tiempo = mínimo


# ─── Contenedor de estadísticas ──────────────────────────────────────────────

@dataclass
class Stats:
    estrategia:          str
    red:                 str
    k:                   Optional[int]
    tiempo_ms:           float = 0.0
    perdida:             float = 0.0
    delta_hits:          int   = 0
    delta_misses:        int   = 0
    union_hits:          int   = 0
    union_misses:        int   = 0
    bipartir_calls:      int   = 0
    dist_marginal_calls: int   = 0

    @property
    def delta_total(self) -> int:
        return self.delta_hits + self.delta_misses

    @property
    def union_total(self) -> int:
        return self.union_hits + self.union_misses

    @property
    def hit_pct_delta(self) -> str:
        return f"{100 * self.delta_hits / self.delta_total:5.1f}%" if self.delta_total else "   —  "

    @property
    def hit_pct_union(self) -> str:
        return f"{100 * self.union_hits / self.union_total:5.1f}%" if self.union_total else "   —  "


# ─── Mixin de instrumentación ─────────────────────────────────────────────────
#
# Envuelve sia_subsistema.bipartir y el distribucion_marginal del resultado
# DESPUÉS de que sia_preparar_subsistema haya construido el subsistema.
# No modifica ninguna clase base.

class _Instrumented:

    def _reset_stats(self, estrategia: str, red: str, k: Optional[int]) -> None:
        self._bench = Stats(estrategia=estrategia, red=red, k=k)

    def sia_preparar_subsistema(self, *args, **kwargs) -> None:
        super().sia_preparar_subsistema(*args, **kwargs)
        bench = self._bench
        orig_bipartir = self.sia_subsistema.bipartir

        def _bipartir_contado(alcance, mecanismo):
            bench.bipartir_calls += 1
            sistema = orig_bipartir(alcance, mecanismo)
            orig_dm = sistema.distribucion_marginal

            def _dm_contada():
                bench.dist_marginal_calls += 1
                return orig_dm()

            sistema.distribucion_marginal = _dm_contada
            return sistema

        self.sia_subsistema.bipartir = _bipartir_contado


# ─── QNodes instrumentado ────────────────────────────────────────────────────

class BenchQNodes(_Instrumented, QNodes):

    def aplicar_estrategia(self, estado, cond, alc, mec):
        self._reset_stats("QNodes", _red_label(estado), None)
        t0 = time.perf_counter()
        r  = super().aplicar_estrategia(estado, cond, alc, mec)
        self._bench.tiempo_ms = (time.perf_counter() - t0) * 1_000
        self._bench.perdida   = r.perdida
        return r

    def funcion_submodular(self, deltas, omegas):
        # len() antes/después: O(1), sin recomputar claves
        d_pre = len(self.memoria_delta)
        u_pre = len(self.memoria_union)
        result = super().funcion_submodular(deltas, omegas)
        if len(self.memoria_delta) > d_pre:
            self._bench.delta_misses += 1
        else:
            self._bench.delta_hits += 1
        if len(self.memoria_union) > u_pre:
            self._bench.union_misses += 1
        else:
            self._bench.union_hits += 1
        return result


# ─── KQNodes instrumentado ───────────────────────────────────────────────────

class BenchKQNodes(_Instrumented, KQNodes):

    def aplicar_estrategia(self, estado, cond, alc, mec, k: int = 2):
        self._reset_stats("KQNodes", _red_label(estado), k)
        t0 = time.perf_counter()
        r  = super().aplicar_estrategia(estado, cond, alc, mec, k=k)
        self._bench.tiempo_ms = (time.perf_counter() - t0) * 1_000
        self._bench.perdida   = r.perdida
        return r

    def funcion_submodular(self, deltas, omega_mask_a: int, omega_mask_e: int):
        d_pre = len(self.memoria_delta)
        u_pre = len(self.memoria_union)
        result = super().funcion_submodular(deltas, omega_mask_a, omega_mask_e)
        if len(self.memoria_delta) > d_pre:
            self._bench.delta_misses += 1
        else:
            self._bench.delta_hits += 1
        if len(self.memoria_union) > u_pre:
            self._bench.union_misses += 1
        else:
            self._bench.union_hits += 1
        return result


# ─── Utilidades ──────────────────────────────────────────────────────────────

def _red_label(estado: str) -> str:
    return f"N{len(estado)}A"


def _asegurar_red(estado: str) -> None:
    gestor = Manager(estado)
    if not gestor.tpm_filename.exists():
        print(f"\n  Generando red {_red_label(estado)} (puede tardar)…", flush=True)
        gestor.generar_red(len(estado))


def _correr(clase, tpm, estado: str, cond: str, alc: str, mec: str,
            k: Optional[int] = None) -> Stats:
    best_ms = float("inf")
    stats: Optional[Stats] = None
    for _ in range(N_TRIALS):
        inst = clase(tpm)
        if k is None:
            inst.aplicar_estrategia(estado, cond, alc, mec)
        else:
            inst.aplicar_estrategia(estado, cond, alc, mec, k=k)
        if inst._bench.tiempo_ms < best_ms:
            best_ms = inst._bench.tiempo_ms
        stats = inst._bench          # stats del trial más reciente (deterministas)
    stats.tiempo_ms = best_ms        # sobreescribir con el mejor tiempo
    return stats


def _imprimir_tabla(filas: list[Stats]) -> None:
    COL = (
        f"{'Red':5s}  {'Estrategia':9s}  {'k':>3s}  "
        f"{'perdida':>8s}  {'t(ms)':>7s}  "
        f"{'Δtotal':>7s}  {'Δhit%':>7s}  {'Δmiss':>6s}  "
        f"{'∪total':>7s}  {'∪hit%':>7s}  {'∪miss':>6s}  "
        f"{'bipartir':>8s}  {'dist_mg':>7s}"
    )
    SEP = "─" * len(COL)
    print(SEP)
    print(COL)
    print(SEP)
    prev_red = None
    for s in filas:
        if prev_red and s.red != prev_red:
            print()                  # línea en blanco entre redes
        prev_red = s.red
        k_str = str(s.k) if s.k is not None else " — "
        print(
            f"{s.red:5s}  {s.estrategia:9s}  {k_str:>3s}  "
            f"{s.perdida:8.4f}  {s.tiempo_ms:7.1f}  "
            f"{s.delta_total:7d}  {s.hit_pct_delta:>7s}  {s.delta_misses:6d}  "
            f"{s.union_total:7d}  {s.hit_pct_union:>7s}  {s.union_misses:6d}  "
            f"{s.bipartir_calls:8d}  {s.dist_marginal_calls:7d}"
        )
    print(SEP)
    print(f"  N_TRIALS={N_TRIALS} — timing = mínimo; stats = último trial")


# ─── Punto de entrada ────────────────────────────────────────────────────────

def ejecutar() -> None:
    resultados: list[Stats] = []

    for estado, cond, alc, mec in CASOS:
        red = _red_label(estado)
        print(f"→ {red} ", end="", flush=True)
        _asegurar_red(estado)
        tpm = Manager(estado).cargar_red()

        print("Q.. ", end="", flush=True)
        resultados.append(_correr(BenchQNodes, tpm, estado, cond, alc, mec))

        for k in K_VALORES:
            print(f"KQ{k}.. ", end="", flush=True)
            resultados.append(_correr(BenchKQNodes, tpm, estado, cond, alc, mec, k=k))

        del tpm
        print()

    print()
    _imprimir_tabla(resultados)


if __name__ == "__main__":
    ejecutar()
