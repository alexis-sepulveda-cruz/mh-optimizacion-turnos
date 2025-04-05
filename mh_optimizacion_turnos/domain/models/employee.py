from dataclasses import dataclass, field
from typing import Dict, Set, List
from uuid import UUID, uuid4


@dataclass
class Employee:
    """Representa un empleado en el sistema de asignación de turnos."""
    
    name: str = field(
        metadata={"description": "Nombre del empleado"}
    )
    skills: Set[str] = field(
        default_factory=set,
        metadata={"description": "Conjunto de habilidades que posee el empleado"}
    )
    hourly_cost: float = field(
        default=10.0,
        metadata={"description": "Costo por hora de trabajo del empleado"}
    )
    max_hours_per_week: float = field(
        default=40.0,
        metadata={
            "description": "Máximo de horas que el empleado puede trabajar en una semana"
        }
    )
    max_consecutive_days: int = field(
        default=5,
        metadata={
            "description": "Máximo número de días consecutivos que el empleado puede "
                        "trabajar"
        }
    )
    availability: Dict[str, List[str]] = field(
        default_factory=dict,
        metadata={
            "description": "Diccionario que relaciona días con turnos disponibles. "
                        "Ejemplo: {'lunes': ['mañana', 'tarde']}"
        }
    )
    preferences: Dict[str, Dict[str, int]] = field(
        default_factory=dict,
        metadata={
            "description": "Preferencias del empleado para días y turnos específicos. "
                        "Ejemplo: {'lunes': {'mañana': 8, 'tarde': 5}}"
        }
    )
    id: UUID = field(
        default_factory=uuid4,
        metadata={"description": "Identificador único del empleado"}
    )
    
    def is_available(self, day: str, shift_name: str) -> bool:
        """Verifica si el empleado está disponible para un turno y día específicos.
        
        Args:
            day: El día (por ejemplo, 'martes', '2025-04-01')
            shift_name: El nombre del turno (por ejemplo, 'mañana', 'noche')
            
        Returns:
            True si el empleado está disponible, False en caso contrario
        """
        # Si no hay información de disponibilidad para este día, el empleado no está disponible
        if day not in self.availability:
            return False
        
        # Verificar si el turno específico está en la lista de turnos disponibles para este día
        return shift_name in self.availability[day]
    
    def add_availability(self, day: str, shift_name: str) -> None:
        """Añade disponibilidad para un turno en un día específico.
        
        Args:
            day: El día (por ejemplo, 'martes', '2025-04-01')
            shift_name: El nombre del turno (por ejemplo, 'mañana', 'noche')
        """
        if day not in self.availability:
            self.availability[day] = []
        
        if shift_name not in self.availability[day]:
            self.availability[day].append(shift_name)
    
    def remove_availability(self, day: str, shift_name: str) -> None:
        """Elimina disponibilidad para un turno en un día específico.
        
        Args:
            day: El día (por ejemplo, 'martes', '2025-04-01')
            shift_name: El nombre del turno (por ejemplo, 'mañana', 'noche')
        """
        if day in self.availability and shift_name in self.availability[day]:
            self.availability[day].remove(shift_name)
    
    def set_preference(self, day: str, shift_name: str, score: int) -> None:
        """Establece una preferencia para un turno en un día específico.
        
        Args:
            day: El día (por ejemplo, 'martes', '2025-04-01')
            shift_name: El nombre del turno (por ejemplo, 'mañana', 'noche')
            score: Puntuación de preferencia (más alto es mejor, típicamente 1-10)
        """
        if day not in self.preferences:
            self.preferences[day] = {}
        
        self.preferences[day][shift_name] = score
    
    def get_preference_score(self, day: str, shift_name: str) -> int:
        """Obtiene la puntuación de preferencia para un turno en un día específico.
        
        Args:
            day: El día (por ejemplo, 'martes', '2025-04-01')
            shift_name: El nombre del turno (por ejemplo, 'mañana', 'noche')
            
        Returns:
            Puntuación de preferencia (0 si no hay preferencia establecida)
        """
        if day in self.preferences and shift_name in self.preferences[day]:
            return self.preferences[day][shift_name]
        return 0  # Valor predeterminado si no hay preferencia establecida
    
    def add_skill(self, skill: str) -> None:
        """Añade una habilidad al empleado.
        
        Args:
            skill: Nombre de la habilidad a añadir
        """
        self.skills.add(skill)
    
    def remove_skill(self, skill: str) -> None:
        """Elimina una habilidad del empleado.
        
        Args:
            skill: Nombre de la habilidad a eliminar
        """
        if skill in self.skills:
            self.skills.remove(skill)