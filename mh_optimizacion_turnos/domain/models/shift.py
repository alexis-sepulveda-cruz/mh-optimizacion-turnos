from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Set, Optional
from uuid import UUID, uuid4


@dataclass
class Shift:
    """Representa un turno en el sistema de asignación de turnos."""
    
    name: str
    day: str  # puede ser 'lunes' o una fecha específica como '2025-04-01'
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    duration_hours: float = 8.0  # duración en horas
    required_employees: int = 1
    priority: int = 1  # 1-10, con 10 siendo la mayor prioridad
    required_skills: Set[str] = field(default_factory=set)
    id: UUID = field(default_factory=uuid4)
    
    def add_required_skill(self, skill: str) -> None:
        """Añade una habilidad requerida para el turno.
        
        Args:
            skill: Nombre de la habilidad a añadir como requisito
        """
        self.required_skills.add(skill)
    
    def remove_required_skill(self, skill: str) -> None:
        """Elimina una habilidad requerida del turno.
        
        Args:
            skill: Nombre de la habilidad a eliminar de los requisitos
        """
        if skill in self.required_skills:
            self.required_skills.remove(skill)
    
    def set_time_range(self, start: time, end: time) -> None:
        """Establece el rango horario del turno y actualiza la duración.
        
        Args:
            start: Hora de inicio del turno
            end: Hora de finalización del turno
        """
        self.start_time = start
        self.end_time = end
        
        # Calcular la duración en horas
        if start and end:
            # Convertir a datetime para poder calcular la diferencia
            start_dt = datetime.combine(datetime.today(), start)
            end_dt = datetime.combine(datetime.today(), end)
            
            # Si el turno termina al día siguiente
            if end_dt < start_dt:
                end_dt = datetime.combine(datetime.today().replace(day=datetime.today().day + 1), end)
            
            # Calcular la diferencia en horas
            self.duration_hours = (end_dt - start_dt).total_seconds() / 3600.0