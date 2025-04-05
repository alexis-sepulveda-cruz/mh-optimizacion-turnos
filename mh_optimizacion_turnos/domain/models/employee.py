from dataclasses import dataclass, field
from typing import Dict, Set, List, Union
from uuid import UUID, uuid4

from mh_optimizacion_turnos.domain.models.day import Day
from mh_optimizacion_turnos.domain.models.shift_type import ShiftType
from mh_optimizacion_turnos.domain.models.skill import Skill


@dataclass
class Employee:
    """Representa un empleado en el sistema de asignación de turnos."""
    
    name: str = field(
        metadata={"description": "Nombre del empleado"}
    )
    skills: Set[Skill] = field(
        default_factory=set,
        metadata={"description": "Conjunto de habilidades (enums Skill) que posee el empleado"}
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
    availability: Dict[Day, List[ShiftType]] = field(
        default_factory=dict,
        metadata={
            "description": "Diccionario que relaciona días (enum Day) con turnos disponibles (enum ShiftType). "
                        "Ejemplo: {Day.LUNES: [ShiftType.MAÑANA, ShiftType.TARDE]}"
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
    
    def is_available(self, day: Union[Day, str], shift_name: Union[ShiftType, str]) -> bool:
        """Verifica si el empleado está disponible para un turno y día específicos.
        
        Args:
            day: El día como enum Day o como cadena 'martes', '2025-04-01'
            shift_name: El turno como enum ShiftType o como cadena 'mañana', 'noche'
            
        Returns:
            True si el empleado está disponible, False en caso contrario
        """
        # Convertir a Day si se proporciona como cadena
        day_key = day
        if isinstance(day, str):
            try:
                day_key = Day.from_string(day)
            except ValueError:
                # Si no es un día de la semana válido, asumimos que es una fecha específica
                # En este caso, mantenemos la compatibilidad con el comportamiento anterior
                if day not in self.availability:
                    return False
                
                # En este caso, verificamos la disponibilidad para un día específico (fecha)
                # que no es un enum Day
                shift_key = shift_name
                if isinstance(shift_name, str):
                    try:
                        shift_key = ShiftType.from_string(shift_name)
                    except ValueError:
                        # Si no es un tipo de turno válido, no está disponible
                        return False
                
                # Verificamos si el turno está en la lista de disponibilidad para este día
                available_shifts = self.availability[day]
                for avail_shift in available_shifts:
                    # Comparar con el nombre del enum si es necesario
                    if isinstance(avail_shift, ShiftType):
                        if avail_shift == shift_key:
                            return True
                    elif isinstance(avail_shift, str) and isinstance(shift_name, str):
                        if avail_shift.lower() == shift_name.lower():
                            return True
                
                return False
        
        # Convertir shift_name a ShiftType si es una cadena
        shift_key = shift_name
        if isinstance(shift_name, str):
            try:
                shift_key = ShiftType.from_string(shift_name)
            except ValueError:
                # Si no es un tipo de turno válido, no está disponible
                return False
        
        # Verificar disponibilidad con los enums
        if day_key not in self.availability:
            return False
        
        # Verificar si el turno específico está en la lista de turnos disponibles para este día
        return shift_key in self.availability[day_key]
    
    def add_availability(self, day: Union[Day, str], shift_name: Union[ShiftType, str]) -> None:
        """Añade disponibilidad para un turno en un día específico.
        
        Args:
            day: El día como enum Day o como cadena 'martes', '2025-04-01'
            shift_name: El turno como enum ShiftType o como cadena 'mañana', 'noche'
        """
        # Convertir a Day si se proporciona como cadena
        day_key = day
        if isinstance(day, str):
            try:
                day_key = Day.from_string(day)
            except ValueError:
                # Si no es un día de la semana válido, asumimos que es una fecha específica
                pass
        
        # Convertir shift_name a ShiftType si es una cadena
        shift_key = shift_name
        if isinstance(shift_name, str):
            try:
                shift_key = ShiftType.from_string(shift_name)
            except ValueError:
                # Si no es un tipo de turno válido, usamos el string tal cual
                pass
                
        if day_key not in self.availability:
            self.availability[day_key] = []
        
        if shift_key not in self.availability[day_key]:
            self.availability[day_key].append(shift_key)
    
    def remove_availability(self, day: Union[Day, str], shift_name: Union[ShiftType, str]) -> None:
        """Elimina disponibilidad para un turno en un día específico.
        
        Args:
            day: El día como enum Day o como cadena 'martes', '2025-04-01'
            shift_name: El turno como enum ShiftType o como cadena 'mañana', 'noche'
        """
        # Convertir a Day si se proporciona como cadena
        day_key = day
        if isinstance(day, str):
            try:
                day_key = Day.from_string(day)
            except ValueError:
                # Si no es un día de la semana válido, asumimos que es una fecha específica
                pass
        
        # Convertir shift_name a ShiftType si es una cadena
        shift_key = shift_name
        if isinstance(shift_name, str):
            try:
                shift_key = ShiftType.from_string(shift_name)
            except ValueError:
                # Si no es un tipo de turno válido, usamos el string tal cual
                pass
                
        if day_key in self.availability and shift_key in self.availability[day_key]:
            self.availability[day_key].remove(shift_key)
    
    def set_preference(self, day: Union[Day, str], shift_name: Union[ShiftType, str], score: int) -> None:
        """Establece una preferencia para un turno en un día específico.
        
        Args:
            day: El día como enum Day o como cadena 'martes', '2025-04-01'
            shift_name: El turno como enum ShiftType o como cadena 'mañana', 'noche'
            score: Puntuación de preferencia (más alto es mejor, típicamente 1-10)
        """
        # Convertir a string para mantener compatibilidad con el formato de preferencias
        day_key = day.to_string() if isinstance(day, Day) else day
        shift_key = shift_name.to_string() if isinstance(shift_name, ShiftType) else shift_name
        
        if day_key not in self.preferences:
            self.preferences[day_key] = {}
        
        self.preferences[day_key][shift_key] = score
    
    def get_preference_score(self, day: Union[Day, str], shift_name: Union[ShiftType, str]) -> int:
        """Obtiene la puntuación de preferencia para un turno en un día específico.
        
        Args:
            day: El día como enum Day o como cadena 'martes', '2025-04-01'
            shift_name: El turno como enum ShiftType o como cadena 'mañana', 'noche'
            
        Returns:
            Puntuación de preferencia (0 si no hay preferencia establecida)
        """
        # Convertir a string para mantener compatibilidad con el formato de preferencias
        day_key = day.to_string() if isinstance(day, Day) else day
        shift_key = shift_name.to_string() if isinstance(shift_name, ShiftType) else shift_name
        
        if day_key in self.preferences and shift_key in self.preferences[day_key]:
            return self.preferences[day_key][shift_key]
        return 0  # Valor predeterminado si no hay preferencia establecida
    
    def add_skill(self, skill: Union[Skill, str]) -> None:
        """Añade una habilidad al empleado.
        
        Args:
            skill: La habilidad como enum Skill o como cadena
        """
        # Convertir skill a Skill si es una cadena
        skill_obj = skill
        if isinstance(skill, str):
            try:
                skill_obj = Skill.from_string(skill)
            except ValueError:
                # Si no es una habilidad válida como enum, mantenemos retrocompatibilidad
                # usando el string directamente (menos preferible)
                pass
        
        self.skills.add(skill_obj)
    
    def remove_skill(self, skill: Union[Skill, str]) -> None:
        """Elimina una habilidad del empleado.
        
        Args:
            skill: La habilidad como enum Skill o como cadena
        """
        # Si es un string, intentamos convertirlo a Skill
        if isinstance(skill, str):
            try:
                # Intentar encontrar por enum
                skill_obj = Skill.from_string(skill)
                if skill_obj in self.skills:
                    self.skills.remove(skill_obj)
                    return
            except ValueError:
                # Si no es una habilidad válida como enum, buscamos por string
                # para retrocompatibilidad
                for s in list(self.skills):
                    if isinstance(s, str) and s.lower() == skill.lower():
                        self.skills.remove(s)
                        return
        elif skill in self.skills:
            self.skills.remove(skill)
            
    def has_skill(self, skill: Union[Skill, str]) -> bool:
        """Verifica si el empleado tiene una habilidad específica.
        
        Args:
            skill: La habilidad como enum Skill o como cadena
            
        Returns:
            True si el empleado tiene la habilidad, False en caso contrario
        """
        # Si es un string, intentamos convertirlo a Skill
        if isinstance(skill, str):
            try:
                # Intentar encontrar por enum
                skill_obj = Skill.from_string(skill)
                return skill_obj in self.skills
            except ValueError:
                # Si no es una habilidad válida como enum, buscamos por string
                # para retrocompatibilidad
                for s in self.skills:
                    if isinstance(s, str) and s.lower() == skill.lower():
                        return True
                return False
        else:
            return skill in self.skills