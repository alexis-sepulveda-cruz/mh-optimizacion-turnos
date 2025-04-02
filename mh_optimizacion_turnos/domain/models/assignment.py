from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass
class Assignment:
    """Entidad de dominio Asignación que representa la asignación de un empleado a un turno."""
    
    employee_id: UUID
    shift_id: UUID
    id: UUID = uuid4()
    cost: float = 0.0  # Costo calculado de esta asignación
    
    def __eq__(self, other):
        if not isinstance(other, Assignment):
            return False
        return self.employee_id == other.employee_id and self.shift_id == other.shift_id
    
    def __hash__(self):
        return hash((self.employee_id, self.shift_id))