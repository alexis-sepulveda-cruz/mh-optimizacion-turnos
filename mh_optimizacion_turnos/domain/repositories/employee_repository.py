from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from mh_optimizacion_turnos.domain.models.employee import Employee


class EmployeeRepository(ABC):
    """Interfaz de repositorio para entidades Empleado."""
    
    @abstractmethod
    def get_by_id(self, employee_id: UUID) -> Optional[Employee]:
        """Obtener un empleado por su ID."""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Employee]:
        """Obtener todos los empleados."""
        pass
    
    @abstractmethod
    def save(self, employee: Employee) -> Employee:
        """Guardar un empleado (crear o actualizar)."""
        pass
    
    @abstractmethod
    def delete(self, employee_id: UUID) -> None:
        """Eliminar un empleado."""
        pass
    
    @abstractmethod
    def get_available_employees_for_shift(self, day: str, shift: str) -> List[Employee]:
        """Obtener todos los empleados disponibles para un turno específico en un día específico."""
        pass