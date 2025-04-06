from typing import Dict, Any, Optional, List
import logging
import time
from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.repositories.employee_repository import EmployeeRepository
from mh_optimizacion_turnos.domain.repositories.shift_repository import ShiftRepository
from mh_optimizacion_turnos.domain.services.optimizer_strategy import OptimizerStrategy
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.value_objects.algorithm_type import AlgorithmType

logger = logging.getLogger(__name__)


class ShiftOptimizerService:
    """Servicio para optimizar asignaciones de turnos usando algoritmos metaheurísticos configurables.
    
    Este servicio implementa un enfoque de restricciones duras donde todas las soluciones
    generadas deben cumplir con todas las restricciones definidas.
    """
    
    def __init__(
        self,
        employee_repository: EmployeeRepository,
        shift_repository: ShiftRepository,
        solution_validator: SolutionValidator,
        optimizer_strategy: Optional[OptimizerStrategy] = None
    ):
        self.employee_repository = employee_repository
        self.shift_repository = shift_repository
        self.solution_validator = solution_validator
        self.optimizer_strategy = optimizer_strategy
        self.metrics = {
            "total_optimizations": 0,
            "valid_solutions_generated": 0,
            "total_execution_time": 0,
            "algorithm_metrics": {}
        }
    
    def set_optimizer_strategy(self, optimizer_strategy: OptimizerStrategy) -> None:
        """Establecer la estrategia de optimización a usar.
        
        Esto permite cambiar entre diferentes algoritmos metaheurísticos
        en tiempo de ejecución basado en preferencias del usuario o características del problema.
        """
        self.optimizer_strategy = optimizer_strategy
        logger.info(f"Estrategia de optimización establecida a: {optimizer_strategy.get_name()}")
    
    def generate_optimal_schedule(
        self,
        start_date: str = None,
        end_date: str = None,
        config: Dict[str, Any] = None
    ) -> Solution:
        """Generar un horario óptimo para el período dado usando la estrategia seleccionada.
        
        Implementa un enfoque de restricciones duras donde todas las soluciones deben
        cumplir con todas las restricciones definidas. No se aceptan soluciones inválidas.
        
        Args:
            start_date: Fecha de inicio para el período de programación
            end_date: Fecha de fin para el período de programación
            config: Parámetros de configuración para el algoritmo de optimización
            
        Returns:
            Un objeto Solution que contiene las asignaciones optimizadas
            
        Raises:
            ValueError: Si no se ha establecido una estrategia de optimización
                      o si la solución generada viola restricciones (restricciones duras)
        """
        if not self.optimizer_strategy:
            raise ValueError("No se ha establecido ninguna estrategia de optimización")
        
        # Obtener todos los empleados y turnos relevantes
        employees: List[Employee] = self.employee_repository.get_all()
        
        if start_date and end_date:
            shifts: List[Shift] = self.shift_repository.get_shifts_by_period(start_date, end_date)
        else:
            shifts: List[Shift] = self.shift_repository.get_all()
        
        logger.info(f"Iniciando optimización con {len(employees)} empleados y {len(shifts)} turnos")
        logger.info(f"Usando enfoque de restricciones duras - solo se aceptarán soluciones válidas")
        
        # Registrar métricas
        self.metrics["total_optimizations"] += 1
        algorithm_name = self.optimizer_strategy.get_name()
        
        if algorithm_name not in self.metrics["algorithm_metrics"]:
            self.metrics["algorithm_metrics"][algorithm_name] = {
                "runs": 0,
                "total_time": 0,
                "avg_time": 0,
                "valid_solutions": 0
            }
        
        self.metrics["algorithm_metrics"][algorithm_name]["runs"] += 1
        
        # Medir tiempo de ejecución
        start_time = time.time()
        
        # Usar la estrategia para optimizar
        solution = self.optimizer_strategy.optimize(employees, shifts, config)
        
        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        self.metrics["total_execution_time"] += execution_time
        self.metrics["algorithm_metrics"][algorithm_name]["total_time"] += execution_time
        self.metrics["algorithm_metrics"][algorithm_name]["avg_time"] = (
            self.metrics["algorithm_metrics"][algorithm_name]["total_time"] / 
            self.metrics["algorithm_metrics"][algorithm_name]["runs"]
        )
        
        # Validar solución - con restricciones duras, la solución debe ser válida
        validation_result = self.solution_validator.validate(solution, employees, shifts)
        if not validation_result.is_valid:
            logger.error(f"La solución generada tiene {validation_result.violations} violaciones de restricciones:")
            for detail in validation_result.violation_details[:5]:  # Mostrar hasta 5 violaciones
                logger.error(f"- {detail}")
                
            if len(validation_result.violation_details) > 5:
                logger.error(f"- ...y {len(validation_result.violation_details) - 5} más")
                
            # Con restricciones duras, una solución inválida es inaceptable
            raise ValueError("La solución generada viola restricciones duras y no puede ser aceptada")
        
        # Actualizar métricas de éxito
        self.metrics["valid_solutions_generated"] += 1
        self.metrics["algorithm_metrics"][algorithm_name]["valid_solutions"] += 1
        
        # Establecer que no hay violaciones de restricciones (enfoque de restricciones duras)
        solution.constraint_violations = 0
        
        # Calcular el costo real de la solución
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        # Costo: Costo real de las asignaciones basado en el costo por hora y duración del turno
        total_cost = 0.0
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Cálculo del costo de esta asignación
                assignment_cost = employee.hourly_cost * shift.duration_hours
                assignment.cost = assignment_cost  # Actualizar el costo en la asignación
                total_cost += assignment_cost
        
        # Asegurar que haya al menos un costo mínimo para evitar divisiones por cero
        solution.total_cost = max(0.1, total_cost)
        
        # Calcular estadísticas de cobertura
        total_shifts = len(shifts)
        required_positions = sum(s.required_employees for s in shifts)
        
        # Contar asignaciones por turno
        shift_employees_count = {}
        for assignment in solution.assignments:
            if assignment.shift_id not in shift_employees_count:
                shift_employees_count[assignment.shift_id] = 0
            shift_employees_count[assignment.shift_id] += 1
        
        # Calcular cobertura
        covered_shifts = sum(1 for s in shifts if shift_employees_count.get(s.id, 0) >= s.required_employees)
        covered_positions = sum(min(shift_employees_count.get(s.id, 0), s.required_employees) for s in shifts)
        
        coverage_percent = 100.0 * (covered_positions / required_positions) if required_positions > 0 else 0
        
        logger.info(f"Optimización completada en {execution_time:.2f} segundos")
        logger.info(f"Costo total: {solution.total_cost:.2f}, Asignaciones: {len(solution.assignments)}")
        logger.info(f"Cobertura: {covered_shifts}/{total_shifts} turnos completos ({coverage_percent:.1f}% posiciones cubiertas)")
        
        return solution
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento del servicio de optimización."""
        return self.metrics
        
    def set_algorithm(self, algorithm_type: AlgorithmType) -> None:
        """Configurar el algoritmo a utilizar basado en un tipo de algoritmo."""
        from mh_optimizacion_turnos.domain.services.optimizers.genetic_algorithm_optimizer import GeneticAlgorithmOptimizer
        from mh_optimizacion_turnos.domain.services.optimizers.tabu_search_optimizer import TabuSearchOptimizer
        from mh_optimizacion_turnos.domain.services.optimizers.grasp_optimizer import GraspOptimizer
        
        optimizer = None
        
        if algorithm_type == AlgorithmType.GENETIC:
            optimizer = GeneticAlgorithmOptimizer()
        elif algorithm_type == AlgorithmType.TABU:
            optimizer = TabuSearchOptimizer()
        elif algorithm_type == AlgorithmType.GRASP:
            optimizer = GraspOptimizer()
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm_type}")
        
        self.set_optimizer_strategy(optimizer)
        
    def generate_schedule(self, algorithm_config: Dict[str, Any] = None) -> Solution:
        """Método simplificado para generar un horario con la configuración actual."""
        return self.generate_optimal_schedule(config=algorithm_config)