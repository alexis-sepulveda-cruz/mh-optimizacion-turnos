from typing import Dict, Any, List, Optional, Union
import logging

from mh_optimizacion_turnos.application.ports.input.shift_assignment_service_port import ShiftAssignmentServicePort
from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.value_objects.algorithm_type import AlgorithmType
from mh_optimizacion_turnos.domain.services.shift_optimizer_service import ShiftOptimizerService
from mh_optimizacion_turnos.domain.services.optimizers.genetic_algorithm_optimizer import GeneticAlgorithmOptimizer
from mh_optimizacion_turnos.domain.services.optimizers.tabu_search_optimizer import TabuSearchOptimizer
from mh_optimizacion_turnos.domain.services.optimizers.grasp_optimizer import GraspOptimizer


logger = logging.getLogger(__name__)


class ShiftAssignmentServiceAdapter(ShiftAssignmentServicePort):
    """Adaptador de entrada para el servicio de asignación de turnos.
    
    Implementa el puerto de entrada ShiftAssignmentServicePort y se comunica
    con el dominio de la aplicación.
    """
    
    def __init__(self, shift_optimizer_service: ShiftOptimizerService):
        self.shift_optimizer_service = shift_optimizer_service
        self.algorithms = {
            AlgorithmType.GENETIC: GeneticAlgorithmOptimizer(),
            AlgorithmType.TABU: TabuSearchOptimizer(),
            AlgorithmType.GRASP: GraspOptimizer()
        }
    
    def _convert_to_algorithm_enum(self, algorithm: Union[AlgorithmType, str]) -> AlgorithmType:
        """
        Convierte un nombre de algoritmo (string o enum) a un enum AlgorithmType.
        
        Args:
            algorithm: Algoritmo como enum AlgorithmType o string
            
        Returns:
            Instancia de AlgorithmType
            
        Raises:
            ValueError: Si el algoritmo no está disponible
        """
        if algorithm is None:
            return None
            
        # Si ya es un enum, lo devolvemos tal cual
        if isinstance(algorithm, AlgorithmType):
            return algorithm
            
        # Convertir de string a enum
        try:
            return AlgorithmType.from_string(algorithm)
        except ValueError:
            available_algorithms = self.get_available_algorithms()
            raise ValueError(f"Algoritmo no disponible: {algorithm}. "
                           f"Opciones: {', '.join(available_algorithms)}")
    
    def _validate_algorithm(self, algorithm_enum: AlgorithmType) -> None:
        """
        Valida que el algoritmo esté disponible.
        
        Args:
            algorithm_enum: Algoritmo como enum AlgorithmType
            
        Raises:
            ValueError: Si el algoritmo no está disponible
        """
        if algorithm_enum not in self.algorithms:
            available_algorithms = self.get_available_algorithms()
            raise ValueError(f"Algoritmo no disponible: {algorithm_enum}. "
                           f"Opciones: {', '.join(available_algorithms)}")
    
    def generate_schedule(self,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         algorithm: Union[AlgorithmType, str] = None,
                         algorithm_config: Dict[str, Any] = None) -> Solution:
        """Genera un cronograma de asignación de turnos.
        
        Implementación del método definido en el puerto de entrada.
        """
        # Convertir algorithm a enum si es necesario
        algorithm_enum = self._convert_to_algorithm_enum(algorithm)
        
        # Si se especifica un algoritmo válido, lo establecemos
        if algorithm_enum and algorithm_enum in self.algorithms:
            self.set_algorithm(algorithm_enum)
        
        # Si no hay un algoritmo establecido en el servicio, usamos el genético por defecto
        if not self.shift_optimizer_service.optimizer_strategy:
            self.set_algorithm(AlgorithmType.GENETIC)
        
        # Generamos el cronograma
        return self.shift_optimizer_service.generate_optimal_schedule(
            start_date=start_date,
            end_date=end_date,
            config=algorithm_config
        )
    
    def get_available_algorithms(self) -> List[str]:
        """Obtiene la lista de algoritmos de optimización disponibles como strings."""
        return [alg.to_string() for alg in self.algorithms.keys()]
    
    def get_available_algorithm_enums(self) -> List[AlgorithmType]:
        """Obtiene la lista de algoritmos de optimización disponibles como enums."""
        return list(self.algorithms.keys())
    
    def set_algorithm(self, algorithm_name: Union[AlgorithmType, str]) -> None:
        """Establece el algoritmo de optimización a utilizar."""
        # Convertir y validar el algoritmo
        algorithm_enum = self._convert_to_algorithm_enum(algorithm_name)
        self._validate_algorithm(algorithm_enum)
        
        # Establecer el algoritmo en el servicio
        self.shift_optimizer_service.set_optimizer_strategy(self.algorithms[algorithm_enum])
        logger.info(f"Algoritmo establecido: {algorithm_enum.to_string()}")
    
    def get_algorithm_default_config(self, algorithm_name: Union[AlgorithmType, str]) -> Dict[str, Any]:
        """Obtiene la configuración predeterminada para un algoritmo."""
        # Convertir y validar el algoritmo
        algorithm_enum = self._convert_to_algorithm_enum(algorithm_name)
        self._validate_algorithm(algorithm_enum)
        
        return self.algorithms[algorithm_enum].get_default_config()