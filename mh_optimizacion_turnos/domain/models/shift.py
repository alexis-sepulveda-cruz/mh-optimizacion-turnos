from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Set, Optional, Union
from uuid import UUID, uuid4

from mh_optimizacion_turnos.domain.models.day import Day
from mh_optimizacion_turnos.domain.models.shift_type import ShiftType
from mh_optimizacion_turnos.domain.models.skill import Skill


@dataclass
class Shift:
    """Representa un turno en el sistema de asignación de turnos."""
    
    name: Union[ShiftType, str] = field(
        metadata={"description": "Nombre o tipo de turno como enum ShiftType"}
    )
    day: Union[Day, str] = field(
        metadata={
            "description": "Día del turno, puede ser un enum Day o una fecha específica "
                        "como '2025-04-01'"
        }
    )
    start_time: Optional[time] = field(
        default=None,
        metadata={"description": "Hora de inicio del turno"}
    )
    end_time: Optional[time] = field(
        default=None,
        metadata={"description": "Hora de finalización del turno"}
    )
    duration_hours: float = field(
        default=8.0,
        metadata={"description": "Duración del turno en horas"}
    )
    required_employees: int = field(
        default=1,
        metadata={"description": "Número de empleados necesarios para este turno"}
    )
    priority: int = field(
        default=1,
        metadata={
            "description": "Prioridad del turno en escala 1-10, siendo 10 la mayor "
                        "prioridad"
        }
    )
    required_skills: Set[Union[Skill, str]] = field(
        default_factory=set,
        metadata={
            "description": "Conjunto de habilidades requeridas (enums Skill) para poder cubrir "
                        "este turno"
        }
    )
    id: UUID = field(
        default_factory=uuid4,
        metadata={"description": "Identificador único del turno"}
    )
    
    def add_required_skill(self, skill: Union[Skill, str]) -> None:
        """Añade una habilidad requerida para el turno.
        
        Args:
            skill: Habilidad a añadir como requisito, como enum Skill o cadena
        """
        # Convertir a Skill si es una cadena
        skill_obj = skill
        if isinstance(skill, str):
            try:
                skill_obj = Skill.from_string(skill)
            except ValueError:
                # Si no es una habilidad válida como enum, mantenemos retrocompatibilidad
                # usando el string directamente (menos preferible)
                pass
                
        self.required_skills.add(skill_obj)
    
    def remove_required_skill(self, skill: Union[Skill, str]) -> None:
        """Elimina una habilidad requerida del turno.
        
        Args:
            skill: Habilidad a eliminar, como enum Skill o cadena
        """
        # Si es un string, intentamos convertirlo a Skill
        if isinstance(skill, str):
            try:
                # Intentar encontrar por enum
                skill_obj = Skill.from_string(skill)
                if skill_obj in self.required_skills:
                    self.required_skills.remove(skill_obj)
                    return
            except ValueError:
                # Si no es una habilidad válida como enum, buscamos por string
                # para retrocompatibilidad
                for s in list(self.required_skills):
                    if isinstance(s, str) and s.lower() == skill.lower():
                        self.required_skills.remove(s)
                        return
        elif skill in self.required_skills:
            self.required_skills.remove(skill)
    
    def requires_skill(self, skill: Union[Skill, str]) -> bool:
        """Verifica si el turno requiere una habilidad específica.
        
        Args:
            skill: La habilidad como enum Skill o como cadena
            
        Returns:
            True si el turno requiere la habilidad, False en caso contrario
        """
        # Si es un string, intentamos convertirlo a Skill
        if isinstance(skill, str):
            try:
                # Intentar encontrar por enum
                skill_obj = Skill.from_string(skill)
                return skill_obj in self.required_skills
            except ValueError:
                # Si no es una habilidad válida como enum, buscamos por string
                # para retrocompatibilidad
                for s in self.required_skills:
                    if isinstance(s, str) and s.lower() == skill.lower():
                        return True
                return False
        else:
            return skill in self.required_skills
    
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
                end_dt = datetime.combine(datetime.today().replace(
                    day=datetime.today().day + 1), end)
            
            # Calcular la diferencia en horas
            self.duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
            
    def get_day_string(self) -> str:
        """Obtiene una representación en cadena del día del turno.
        
        Returns:
            Cadena de texto con el nombre del día
        """
        if isinstance(self.day, Day):
            return self.day.to_string()
        return str(self.day)
        
    def get_name_string(self) -> str:
        """Obtiene una representación en cadena del nombre o tipo de turno.
        
        Returns:
            Cadena de texto con el nombre del turno
        """
        if isinstance(self.name, ShiftType):
            return self.name.to_string()
        return str(self.name)
        
    def get_required_skills_as_strings(self) -> Set[str]:
        """Obtiene las habilidades requeridas como cadenas de texto.
        
        Returns:
            Conjunto de cadenas con los nombres de las habilidades requeridas
        """
        result = set()
        for skill in self.required_skills:
            if isinstance(skill, Skill):
                result.add(skill.to_string())
            else:
                result.add(str(skill))
        return result