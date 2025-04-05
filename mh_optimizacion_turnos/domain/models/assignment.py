from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class Assignment:
    """Representa una asignación de un empleado a un turno."""
    
    employee_id: UUID = field(
        metadata={"description": "Identificador único del empleado asignado"}
    )
    shift_id: UUID = field(
        metadata={"description": "Identificador único del turno asignado"}
    )
    cost: float = field(
        default=0.0, 
        metadata={"description": "Costo asociado a esta asignación específica"}
    )
    
    def __hash__(self) -> int:
        return hash((self.employee_id, self.shift_id))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Assignment):
            return False
        return (self.employee_id == other.employee_id and
                self.shift_id == other.shift_id)