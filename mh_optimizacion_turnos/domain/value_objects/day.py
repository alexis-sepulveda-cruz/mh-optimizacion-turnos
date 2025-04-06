from enum import Enum, auto
from typing import List


class Day(Enum):
    """Representa los días de la semana."""
    
    LUNES = auto()
    MARTES = auto()
    MIERCOLES = auto()
    JUEVES = auto()
    VIERNES = auto()
    SABADO = auto()
    DOMINGO = auto()
    
    @classmethod
    def from_string(cls, day_str: str) -> 'Day':
        """Convierte una cadena de texto a un enum Day.
        
        Args:
            day_str: Cadena de texto que representa un día de la semana
            
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
        elif day_str_lower in ('miércoles', 'miercoles'):
            return cls.MIERCOLES
        elif day_str_lower == 'jueves':
            return cls.JUEVES
        elif day_str_lower == 'viernes':
            return cls.VIERNES
        elif day_str_lower == 'sábado' or day_str_lower == 'sabado':
            return cls.SABADO
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
        elif self == self.MIERCOLES:
            return "Miércoles"
        elif self == self.JUEVES:
            return "Jueves"
        elif self == self.VIERNES:
            return "Viernes"
        elif self == self.SABADO:
            return "Sábado"
        elif self == self.DOMINGO:
            return "Domingo"
        
    @classmethod
    def get_weekdays(cls) -> List['Day']:
        """Obtiene una lista con los días laborables (lunes a viernes).
        
        Returns:
            Lista de enums Day correspondientes a los días laborables
        """
        return [cls.LUNES, cls.MARTES, cls.MIERCOLES, cls.JUEVES, cls.VIERNES]
    
    @classmethod
    def get_weekend(cls) -> List['Day']:
        """Obtiene una lista con los días de fin de semana (sábado y domingo).
        
        Returns:
            Lista de enums Day correspondientes al fin de semana
        """
        return [cls.SABADO, cls.DOMINGO]