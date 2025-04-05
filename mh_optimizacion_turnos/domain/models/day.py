from enum import Enum, auto
from typing import List


class Day(Enum):
    """Representa los días de la semana."""
    
    LUNES = auto()
    MARTES = auto()
    MIERCOLES = auto()
    MIERCOLES_CON_TILDE = auto()  # Para compatibilidad con "Miércoles"
    JUEVES = auto()
    VIERNES = auto()
    SABADO = auto()
    SABADO_CON_TILDE = auto()  # Para compatibilidad con "Sábado"
    DOMINGO = auto()
    
    @classmethod
    def from_string(cls, day_str: str) -> 'Day':
        """Convierte una cadena de texto a un enum Day.
        
        Args:
            day_str: Cadena de texto que representa un día
            
        Returns:
            Enum Day correspondiente
            
        Raises:
            ValueError: Si la cadena no corresponde a un día válido
        """
        day_str_lower = day_str.lower()
        
        if day_str_lower == 'lunes':
            return cls.LUNES
        elif day_str_lower == 'martes':
            return cls.MARTES
        elif day_str_lower in ('miercoles', 'miércoles'):
            return cls.MIERCOLES if day_str_lower == 'miercoles' else cls.MIERCOLES_CON_TILDE
        elif day_str_lower == 'jueves':
            return cls.JUEVES
        elif day_str_lower == 'viernes':
            return cls.VIERNES
        elif day_str_lower in ('sabado', 'sábado'):
            return cls.SABADO if day_str_lower == 'sabado' else cls.SABADO_CON_TILDE
        elif day_str_lower == 'domingo':
            return cls.DOMINGO
        else:
            raise ValueError(f"'{day_str}' no es un día de la semana válido")
    
    def to_string(self) -> str:
        """Convierte el enum a una representación de cadena de texto.
        
        Returns:
            Cadena de texto con el nombre del día
        """
        if self == self.LUNES:
            return "Lunes"
        elif self == self.MARTES:
            return "Martes"
        elif self in (self.MIERCOLES, self.MIERCOLES_CON_TILDE):
            return "Miércoles"
        elif self == self.JUEVES:
            return "Jueves"
        elif self == self.VIERNES:
            return "Viernes"
        elif self in (self.SABADO, self.SABADO_CON_TILDE):
            return "Sábado"
        elif self == self.DOMINGO:
            return "Domingo"
        
    @classmethod
    def get_all_days(cls) -> List['Day']:
        """Obtiene una lista con todos los días de la semana.
        
        Returns:
            Lista de enums Day, uno por cada día de la semana
        """
        # Excluimos las versiones alternativas con tilde para evitar duplicados
        return [
            cls.LUNES, cls.MARTES, cls.MIERCOLES, cls.JUEVES, 
            cls.VIERNES, cls.SABADO, cls.DOMINGO
        ]