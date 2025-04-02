from dataclasses import dataclass, field
from typing import Dict, List, Set
from uuid import UUID, uuid4


@dataclass
class Employee:
    """Entidad de dominio Empleado que representa un trabajador que puede ser asignado a turnos."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    max_hours_per_week: int = 40
    max_consecutive_days: int = 5
    skills: Set[str] = field(default_factory=set)
    availability: Dict[str, List[str]] = field(default_factory=dict)
    preferences: Dict[str, int] = field(default_factory=dict)
    hourly_cost: float = 0.0

    def is_available(self, day: str, shift: str) -> bool:
        """Verificar si un empleado está disponible para un día y turno específicos."""
        if day not in self.availability:
            return False
        return shift in self.availability[day]
    
    def get_preference_score(self, day: str, shift: str) -> int:
        """Obtener la puntuación de preferencia para una combinación de día y turno."""
        key = f"{day}_{shift}"
        return self.preferences.get(key, 0)