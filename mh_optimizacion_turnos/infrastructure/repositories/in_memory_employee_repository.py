from typing import List, Optional, Dict
from uuid import UUID

from ...domain.models.employee import Employee
from ...domain.repositories.employee_repository import EmployeeRepository


class InMemoryEmployeeRepository(EmployeeRepository):
    """Implementación en memoria del repositorio de empleados para pruebas y desarrollo."""
    
    def __init__(self):
        self.employees: Dict[UUID, Employee] = {}
    
    def get_by_id(self, employee_id: UUID) -> Optional[Employee]:
        """Obtener un empleado por su ID."""
        return self.employees.get(employee_id)
    
    def get_all(self) -> List[Employee]:
        """Obtener todos los empleados."""
        return list(self.employees.values())
    
    def save(self, employee: Employee) -> Employee:
        """Guardar un empleado (crear o actualizar)."""
        self.employees[employee.id] = employee
        return employee
    
    def delete(self, employee_id: UUID) -> None:
        """Eliminar un empleado."""
        if employee_id in self.employees:
            del self.employees[employee_id]
    
    def get_available_employees_for_shift(self, day: str, shift: str) -> List[Employee]:
        """Obtener todos los empleados disponibles para un turno específico en un día específico."""
        return [
            employee for employee in self.employees.values()
            if employee.is_available(day, shift)
        ]