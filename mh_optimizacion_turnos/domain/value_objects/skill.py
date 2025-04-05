from enum import Enum, auto
from typing import List


class Skill(Enum):
    """Representa las habilidades disponibles para empleados."""
    
    ATENCION_AL_CLIENTE = auto()
    ATENCION_CLIENTE = auto()  # Versión alternativa para compatibilidad
    CAJA = auto()
    INVENTARIO = auto()
    LIMPIEZA = auto()
    SUPERVISOR = auto()
    
    @classmethod
    def from_string(cls, skill_str: str) -> 'Skill':
        """Convierte una cadena de texto a un enum Skill.
        
        Args:
            skill_str: Cadena de texto que representa una habilidad
            
        Returns:
            Enum Skill correspondiente
            
        Raises:
            ValueError: Si la cadena no corresponde a una habilidad válida
        """
        skill_str_lower = skill_str.lower()
        
        if skill_str_lower in ('atención al cliente', 'atencion al cliente', 'atención cliente', 'atencion cliente'):
            return cls.ATENCION_AL_CLIENTE
        elif skill_str_lower == 'caja':
            return cls.CAJA
        elif skill_str_lower == 'inventario':
            return cls.INVENTARIO
        elif skill_str_lower == 'limpieza':
            return cls.LIMPIEZA
        elif skill_str_lower == 'supervisor':
            return cls.SUPERVISOR
        else:
            raise ValueError(f"'{skill_str}' no es una habilidad válida")
    
    def to_string(self) -> str:
        """Convierte el enum a una representación de cadena de texto.
        
        Returns:
            Cadena de texto con el nombre de la habilidad
        """
        if self in (self.ATENCION_AL_CLIENTE, self.ATENCION_CLIENTE):
            return "Atención al cliente"
        elif self == self.CAJA:
            return "Caja"
        elif self == self.INVENTARIO:
            return "Inventario"
        elif self == self.LIMPIEZA:
            return "Limpieza"
        elif self == self.SUPERVISOR:
            return "Supervisor"
        
    @classmethod
    def get_all_skills(cls) -> List['Skill']:
        """Obtiene una lista con todas las habilidades.
        
        Returns:
            Lista de enums Skill
        """
        # Excluimos la versión alternativa para evitar duplicados
        return [
            cls.ATENCION_AL_CLIENTE, 
            cls.CAJA, 
            cls.INVENTARIO, 
            cls.LIMPIEZA, 
            cls.SUPERVISOR
        ]