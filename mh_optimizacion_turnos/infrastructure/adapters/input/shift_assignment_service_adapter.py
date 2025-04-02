from typing import Dict, Any, List, Optional
import logging

from ....application.ports.input.shift_assignment_service_port import ShiftAssignmentServicePort
from ....domain.models.solution import Solution
from ....domain.services.shift_optimizer_service import ShiftOptimizerService
from ....domain.services.optimizers.genetic_algorithm_optimizer import GeneticAlgorithmOptimizer
from ....domain.services.optimizers.tabu_search_optimizer import TabuSearchOptimizer
from ....domain.services.optimizers.grasp_optimizer import GraspOptimizer


logger = logging.getLogger(__name__)


class ShiftAssignmentServiceAdapter(ShiftAssignmentServicePort):
    """Adaptador de entrada para el servicio de asignación de turnos.
    
    Implementa el puerto de entrada ShiftAssignmentServicePort y se comunica
    con el dominio de la aplicación.
    """
    
    def __init__(self, shift_optimizer_service: ShiftOptimizerService):
        self.shift_optimizer_service = shift_optimizer_service
        self.algorithms = {
            "genetic": GeneticAlgorithmOptimizer(),
            "tabu": TabuSearchOptimizer(),
            "grasp": GraspOptimizer()
        }
    
    def generate_schedule(self,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         algorithm: str = None,
                         algorithm_config: Dict[str, Any] = None) -> Solution:
        """Genera un cronograma de asignación de turnos.
        
        Implementación del método definido en el puerto de entrada.
        """
        # Si se especifica un algoritmo, lo establecemos
        if algorithm and algorithm in self.algorithms:
            self.set_algorithm(algorithm)
        
        # Si no hay un algoritmo establecido en el servicio, usamos el genético por defecto
        if not self.shift_optimizer_service.optimizer_strategy:
            self.set_algorithm("genetic")
        
        # Generamos el cronograma
        return self.shift_optimizer_service.generate_optimal_schedule(
            start_date=start_date,
            end_date=end_date,
            config=algorithm_config
        )
    
    def get_available_algorithms(self) -> List[str]:
        """Obtiene la lista de algoritmos de optimización disponibles."""
        return list(self.algorithms.keys())
    
    def set_algorithm(self, algorithm_name: str) -> None:
        """Establece el algoritmo de optimización a utilizar."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algoritmo no disponible: {algorithm_name}. "
                           f"Opciones: {', '.join(self.algorithms.keys())}")
        
        self.shift_optimizer_service.set_optimizer_strategy(self.algorithms[algorithm_name])
        logger.info(f"Algoritmo establecido: {algorithm_name}")
    
    def get_algorithm_default_config(self, algorithm_name: str) -> Dict[str, Any]:
        """Obtiene la configuración predeterminada para un algoritmo."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algoritmo no disponible: {algorithm_name}. "
                           f"Opciones: {', '.join(self.algorithms.keys())}")
        
        return self.algorithms[algorithm_name].get_default_config()