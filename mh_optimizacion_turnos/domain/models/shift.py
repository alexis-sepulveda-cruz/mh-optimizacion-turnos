from dataclasses import dataclass, field
from typing import Set
from uuid import UUID, uuid4
from datetime import datetime, timedelta


@dataclass
class Shift:
    """Entidad de dominio Turno que representa un período de trabajo que necesita ser cubierto."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    day: str = ""
    start_time: datetime = None
    end_time: datetime = None
    required_employees: int = 1
    required_skills: Set[str] = field(default_factory=set)
    priority: int = 1  # Un número más alto significa mayor prioridad
    
    @property
    def duration_hours(self) -> float:
        """Calcular la duración del turno en horas."""
        if not self.start_time or not self.end_time:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 3600.0
        
    def __str__(self) -> str:
        return f"{self.name} ({self.day}: {self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')})"