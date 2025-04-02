from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass
class Assignment:
    """Assignment domain entity representing the assignment of an employee to a shift."""
    
    employee_id: UUID
    shift_id: UUID
    id: UUID = uuid4()
    cost: float = 0.0  # Calculated cost of this assignment
    
    def __eq__(self, other):
        if not isinstance(other, Assignment):
            return False
        return self.employee_id == other.employee_id and self.shift_id == other.shift_id
    
    def __hash__(self):
        return hash((self.employee_id, self.shift_id))