from abc import ABC, abstractmethod
from typing import Dict, Any, List
from uuid import UUID

from ..models.solution import Solution
from ..models.employee import Employee
from ..models.shift import Shift


class OptimizerStrategy(ABC):
    """Interfaz de estrategia para algoritmos de optimización.
    
    Esto sigue el patrón Strategy para permitir algoritmos metaheurísticos intercambiables.
    Cada implementación de algoritmo debe seguir esta interfaz.
    """
    
    @abstractmethod
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimizar asignaciones de turnos basado en los empleados y turnos dados.
        
        Args:
            employees: Lista de empleados disponibles
            shifts: Lista de turnos a ser asignados
            config: Parámetros de configuración específicos del algoritmo
            
        Returns:
            Un objeto Solution que contiene las asignaciones optimizadas
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Obtener el nombre de la estrategia de optimización."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Obtener los parámetros de configuración predeterminados para este algoritmo."""
        pass