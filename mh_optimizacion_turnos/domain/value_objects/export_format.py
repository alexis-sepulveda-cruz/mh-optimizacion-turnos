from enum import Enum, auto
from typing import List


class ExportFormat(Enum):
    """Representa los formatos de exportación disponibles."""
    
    TEXT = auto()
    CSV = auto()
    JSON = auto()
    EXCEL = auto()
    
    @classmethod
    def from_string(cls, format_str: str) -> 'ExportFormat':
        """Convierte una cadena de texto a un enum ExportFormat.
        
        Args:
            format_str: Cadena de texto que representa un formato de exportación
            
        Returns:
            Enum ExportFormat correspondiente
            
        Raises:
            ValueError: Si la cadena no corresponde a un formato válido
        """
        format_str_lower = format_str.lower()
        
        if format_str_lower == 'text':
            return cls.TEXT
        elif format_str_lower == 'csv':
            return cls.CSV
        elif format_str_lower == 'json':
            return cls.JSON
        elif format_str_lower == 'excel':
            return cls.EXCEL
        else:
            raise ValueError(f"'{format_str}' no es un formato de exportación válido")
    
    def to_string(self) -> str:
        """Convierte el enum a una representación de cadena de texto.
        
        Returns:
            Cadena de texto con el nombre del formato
        """
        if self == self.TEXT:
            return "text"
        elif self == self.CSV:
            return "csv"
        elif self == self.JSON:
            return "json"
        elif self == self.EXCEL:
            return "excel"
        
    @classmethod
    def get_all_formats(cls) -> List['ExportFormat']:
        """Obtiene una lista con todos los formatos disponibles.
        
        Returns:
            Lista de enums ExportFormat
        """
        return [cls.TEXT, cls.CSV, cls.JSON, cls.EXCEL]