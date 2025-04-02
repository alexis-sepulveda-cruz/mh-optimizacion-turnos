from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime

from ...domain.models.shift import Shift
from ...domain.repositories.shift_repository import ShiftRepository


class InMemoryShiftRepository(ShiftRepository):
    """Implementación en memoria del repositorio de turnos para pruebas y desarrollo."""
    
    def __init__(self):
        self.shifts: Dict[UUID, Shift] = {}
    
    def get_by_id(self, shift_id: UUID) -> Optional[Shift]:
        """Obtener un turno por su ID."""
        return self.shifts.get(shift_id)
    
    def get_all(self) -> List[Shift]:
        """Obtener todos los turnos."""
        return list(self.shifts.values())
    
    def save(self, shift: Shift) -> Shift:
        """Guardar un turno (crear o actualizar)."""
        self.shifts[shift.id] = shift
        return shift
    
    def delete(self, shift_id: UUID) -> None:
        """Eliminar un turno."""
        if shift_id in self.shifts:
            del self.shifts[shift_id]
    
    def get_shifts_by_day(self, day: str) -> List[Shift]:
        """Obtener todos los turnos para un día específico."""
        return [shift for shift in self.shifts.values() if shift.day == day]
    
    def get_shifts_by_period(self, start_date: str, end_date: str) -> List[Shift]:
        """Obtener todos los turnos dentro de un rango de fechas.
        
        En una implementación real, esto convertiría las fechas a objetos datetime
        y filtraría los turnos según sus fechas. Para esta implementación simplificada,
        asumiremos que day es una cadena que representará la fecha.
        """
        # En una implementación real, haríamos una comparación de fechas
        # Por ahora, asumimos que day es un string que contiene la fecha en cualquier formato
        return list(self.shifts.values())  # Devolvemos todos los turnos para esta implementación básica