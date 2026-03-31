from typing import List

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_STRAREGY_TAG


class GeometricSIA(SIA):
    """
    Estrategia geométrica para encontrar la bipartición de mínima información (MIP).
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self.n: int = 0
        self.m: int = 0
        self.tensors: List[np.ndarray] = []
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        
        raise NotImplementedError
    
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
            self.sia_subsistema.ncubos[i].data.flatten()
            for i in range(self.n)
    ]
    

