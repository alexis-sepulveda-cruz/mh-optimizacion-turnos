from abc import ABC, abstractmethod
from typing import List

from ....domain.models.solution import Solution


class ScheduleExportPort(ABC):
    """Puerto de salida para exportar soluciones de asignación de turnos.
    
    Define la interfaz que los adaptadores de salida utilizarán para exportar
    las soluciones generadas por el sistema.
    """
    
    @abstractmethod
    def export_solution(self, solution: Solution, format_type: str, output_path: str = None, **kwargs) -> str:
        """Exporta una solución de asignación de turnos a varios formatos.
        
        Args:
            solution: La solución a exportar
            format_type: Tipo de formato ('csv', 'json', 'excel', etc.)
            output_path: Ruta para guardar el archivo (opcional)
            kwargs: Parámetros adicionales específicos del formato
            
        Returns:
            Ruta al archivo exportado o representación en cadena de la solución
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Obtiene la lista de formatos de exportación soportados.
        
        Returns:
            Lista de formatos soportados
        """
        pass