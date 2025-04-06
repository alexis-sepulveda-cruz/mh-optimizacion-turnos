from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.value_objects.algorithm_type import AlgorithmType
from mh_optimizacion_turnos.domain.services.optimizer_strategy import OptimizerStrategy


class ShiftAssignmentServicePort(ABC):
    """Puerto de entrada para el servicio de asignación de turnos.
    
    Define la interfaz que los adaptadores de entrada utilizarán para interactuar con
    el núcleo de la aplicación para asignar turnos a empleados.
    """
    
    @abstractmethod
    def generate_schedule(self,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         algorithm: Union[AlgorithmType, str] = None,
                         algorithm_config: Dict[str, Any] = None) -> Solution:
        """Genera un cronograma de asignación de turnos.
        
        Args:
            start_date: Fecha de inicio del período de programación
            end_date: Fecha de fin del período de programación
            algorithm: Algoritmo de optimización a utilizar como enum AlgorithmType o string
            algorithm_config: Configuración específica para el algoritmo
            
        Returns:
            Solución de asignación de turnos optimizada
        """
        pass
    
    @abstractmethod
    def get_available_algorithms(self) -> List[str]:
        """Obtiene la lista de algoritmos de optimización disponibles como strings.
        
        Returns:
            Lista de nombres de algoritmos disponibles
        """
        pass
    
    @abstractmethod
    def set_algorithm(self, algorithm_name: Union[AlgorithmType, str]) -> None:
        """Establece el algoritmo de optimización a utilizar.
        
        Args:
            algorithm_name: Algoritmo a utilizar como enum AlgorithmType o string
            
        Raises:
            ValueError: Si el algoritmo no está disponible
        """
        pass
    
    @abstractmethod
    def get_algorithm_default_config(self, algorithm_name: Union[AlgorithmType, str]) -> Dict[str, Any]:
        """Obtiene la configuración predeterminada para un algoritmo.
        
        Args:
            algorithm_name: Algoritmo como enum AlgorithmType o string
            
        Returns:
            Diccionario con la configuración predeterminada
            
        Raises:
            ValueError: Si el algoritmo no está disponible
        """
        pass