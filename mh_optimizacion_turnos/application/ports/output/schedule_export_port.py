from abc import ABC, abstractmethod
from typing import Union, List

from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.value_objects.export_format import ExportFormat


class ScheduleExportPort(ABC):
    """Puerto de salida para la exportación de cronogramas.
    
    Define la interfaz que los adaptadores de salida utilizarán para exportar
    cronogramas a diferentes formatos.
    """
    
    @abstractmethod
    def export_solution(self, solution: Solution, export_format: Union[ExportFormat, str]) -> str:
        """Exporta una solución a un formato específico.
        
        Args:
            solution: Solución a exportar
            export_format: Formato de exportación como enum ExportFormat o string
                           (text, csv, json, excel)
            
        Returns:
            Cadena con la representación de la solución en el formato especificado
            
        Raises:
            ValueError: Si el formato no está soportado
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Obtiene la lista de formatos de exportación soportados.
        
        Returns:
            Lista de formatos soportados
        """
        pass