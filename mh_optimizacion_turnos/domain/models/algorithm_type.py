from enum import Enum, auto
from typing import List


class AlgorithmType(Enum):
    """Representa los tipos de algoritmos de optimización disponibles."""
    
    GENETIC = auto()
    TABU = auto()
    GRASP = auto()
    
    @classmethod
    def from_string(cls, algorithm_str: str) -> 'AlgorithmType':
        """Convierte una cadena de texto a un enum AlgorithmType.
        
        Args:
            algorithm_str: Cadena de texto que representa un algoritmo
            
        Returns:
            Enum AlgorithmType correspondiente
            
        Raises:
            ValueError: Si la cadena no corresponde a un algoritmo válido
        """
        algorithm_str_lower = algorithm_str.lower()
        
        if algorithm_str_lower == 'genetic':
            return cls.GENETIC
        elif algorithm_str_lower == 'tabu':
            return cls.TABU
        elif algorithm_str_lower == 'grasp':
            return cls.GRASP
        else:
            raise ValueError(f"'{algorithm_str}' no es un tipo de algoritmo válido")
    
    def to_string(self) -> str:
        """Convierte el enum a una representación de cadena de texto.
        
        Returns:
            Cadena de texto con el nombre del algoritmo
        """
        if self == self.GENETIC:
            return "genetic"
        elif self == self.TABU:
            return "tabu"
        elif self == self.GRASP:
            return "grasp"
        
    @classmethod
    def get_all_algorithms(cls) -> List['AlgorithmType']:
        """Obtiene una lista con todos los algoritmos disponibles.
        
        Returns:
            Lista de enums AlgorithmType
        """
        return [cls.GENETIC, cls.TABU, cls.GRASP]