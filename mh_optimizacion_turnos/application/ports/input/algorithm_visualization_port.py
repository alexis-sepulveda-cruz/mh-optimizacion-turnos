"""Puerto de entrada para la visualización de comparación de algoritmos."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class AlgorithmVisualizationPort(ABC):
    """Puerto para visualización y comparación de algoritmos."""
    
    @abstractmethod
    def plot_comparison(self, results: Dict[str, Any], output_dir: str = "./assets/plots") -> Dict[str, str]:
        """
        Crea gráficos de comparación entre algoritmos y los guarda en el directorio especificado.
        
        Args:
            results: Resultados de algoritmos con métricas
            output_dir: Directorio donde guardar los gráficos generados
            
        Returns:
            Diccionario con rutas de los archivos generados
        """
        pass
    
    @abstractmethod
    def find_best_algorithm(self, results: Dict[str, Any]) -> tuple:
        """
        Determina el mejor algoritmo en base a múltiples criterios.
        
        Args:
            results: Resultados de algoritmos con métricas
            
        Returns:
            Tuple con (mejor_algoritmo, criterios_ranking, puntaje_algoritmos)
        """
        pass
    
    @abstractmethod
    def generate_algorithm_summary(self, algorithm_name: str, stats: Dict[str, Any], 
                                  scores: Dict[str, float], rankings: Dict[str, Dict[str, float]], 
                                  strengths: List[str], output_dir: str = "./assets/plots") -> Dict[str, str]:
        """
        Genera un resumen detallado del algoritmo y lo guarda en formato texto y JSON.
        
        Args:
            algorithm_name: Nombre del algoritmo
            stats: Estadísticas del algoritmo
            scores: Puntuaciones globales de algoritmos 
            rankings: Rankings por criterio
            strengths: Lista de fortalezas principales
            output_dir: Directorio donde guardar los resultados
            
        Returns:
            Diccionario con rutas de los archivos generados
        """
        pass