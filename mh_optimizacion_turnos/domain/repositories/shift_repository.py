from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..models.shift import Shift


class ShiftRepository(ABC):
    """Interfaz de repositorio para entidades Turno."""
    
    @abstractmethod
    def get_by_id(self, shift_id: UUID) -> Optional[Shift]:
        """Obtener un turno por su ID."""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Shift]:
        """Obtener todos los turnos."""
        pass
    
    @abstractmethod
    def save(self, shift: Shift) -> Shift:
        """Guardar un turno (crear o actualizar)."""
        pass
    
    @abstractmethod
    def delete(self, shift_id: UUID) -> None:
        """Eliminar un turno."""
        pass
    
    @abstractmethod
    def get_shifts_by_day(self, day: str) -> List[Shift]:
        """Obtener todos los turnos para un día específico."""
        pass
    
    @abstractmethod
    def get_shifts_by_period(self, start_date: str, end_date: str) -> List[Shift]:
        """Obtener todos los turnos dentro de un rango de fechas."""
        pass