"""Puerto de entrada para la configuración de datos de prueba."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from mh_optimizacion_turnos.domain.repositories.employee_repository import EmployeeRepository
from mh_optimizacion_turnos.domain.repositories.shift_repository import ShiftRepository


class TestDataSetupPort(ABC):
    """Puerto para configurar datos de prueba para el sistema."""
    
    @abstractmethod
    def setup_test_data(self, config: Dict[str, Any] = None) -> Tuple[EmployeeRepository, ShiftRepository]:
        """
        Configura datos de ejemplo para pruebas.
        
        Args:
            config: Configuración opcional con parámetros para la generación de datos
            
        Returns:
            Tupla con repositorios de empleados y turnos configurados
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración por defecto para la generación de datos.
        
        Returns:
            Diccionario con la configuración por defecto
        """
        pass