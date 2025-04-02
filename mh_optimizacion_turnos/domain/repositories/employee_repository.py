from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..models.employee import Employee


class EmployeeRepository(ABC):
    """Repository interface for Employee entities."""
    
    @abstractmethod
    def get_by_id(self, employee_id: UUID) -> Optional[Employee]:
        """Get an employee by ID."""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Employee]:
        """Get all employees."""
        pass
    
    @abstractmethod
    def save(self, employee: Employee) -> Employee:
        """Save an employee (create or update)."""
        pass
    
    @abstractmethod
    def delete(self, employee_id: UUID) -> None:
        """Delete an employee."""
        pass
    
    @abstractmethod
    def get_available_employees_for_shift(self, day: str, shift: str) -> List[Employee]:
        """Get all employees available for a specific shift on a specific day."""
        pass