from enum import Enum, auto
from typing import List


class ShiftType(Enum):
    """Representa los tipos de turnos disponibles."""
    
    MAÑANA = auto()
    MANANA = auto()  # Versión sin tilde para compatibilidad
    TARDE = auto()
    NOCHE = auto()
    
    @classmethod
    def from_string(cls, shift_str: str) -> 'ShiftType':
        """Convierte una cadena de texto a un enum ShiftType.
        
        Args:
            shift_str: Cadena de texto que representa un tipo de turno
            
        Returns:
            Enum ShiftType correspondiente
            
        Raises:
            ValueError: Si la cadena no corresponde a un tipo de turno válido
        """
        shift_str_lower = shift_str.lower()
        
        if shift_str_lower in ('mañana', 'manana'):
            return cls.MAÑANA if shift_str_lower == 'mañana' else cls.MANANA
        elif shift_str_lower == 'tarde':
            return cls.TARDE
        elif shift_str_lower == 'noche':
            return cls.NOCHE
        else:
            raise ValueError(f"'{shift_str}' no es un tipo de turno válido")
    
    def to_string(self) -> str:
        """Convierte el enum a una representación de cadena de texto.
        
        Returns:
            Cadena de texto con el nombre del tipo de turno
        """
        if self in (self.MAÑANA, self.MANANA):
            return "Mañana"
        elif self == self.TARDE:
            return "Tarde"
        elif self == self.NOCHE:
            return "Noche"
        
    @classmethod
    def get_all_shift_types(cls) -> List['ShiftType']:
        """Obtiene una lista con todos los tipos de turnos.
        
        Returns:
            Lista de enums ShiftType
        """
        # Excluimos la versión sin tilde de Mañana para evitar duplicados
        return [cls.MAÑANA, cls.TARDE, cls.NOCHE]