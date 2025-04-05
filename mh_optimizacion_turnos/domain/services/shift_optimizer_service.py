from typing import Dict, Any, Optional, List
import logging

from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.repositories.employee_repository import EmployeeRepository
from mh_optimizacion_turnos.domain.repositories.shift_repository import ShiftRepository
from mh_optimizacion_turnos.domain.services.optimizer_strategy import OptimizerStrategy
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator


logger = logging.getLogger(__name__)


class ShiftOptimizerService:
    """Servicio para optimizar asignaciones de turnos usando algoritmos metaheurísticos configurables."""
    
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
        
        Args:
            start_date: Fecha de inicio para el período de programación
            end_date: Fecha de fin para el período de programación
            config: Parámetros de configuración para el algoritmo de optimización
            
        Returns:
            Un objeto Solution que contiene las asignaciones optimizadas
            
        Raises:
            ValueError: Si no se ha establecido una estrategia de optimización
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
        
        # Usar la estrategia para optimizar
        solution = self.optimizer_strategy.optimize(employees, shifts, config)
        
        # Validar solución
        validation_result = self.solution_validator.validate(solution, employees, shifts)
        if not validation_result.is_valid:
            logger.warning(f"La solución generada tiene {validation_result.violations} violaciones de restricciones")
        
        solution.constraint_violations = validation_result.violations
        
        # Calcular el costo real de la solución de una manera más precisa
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        # Costo base: Costo real de las asignaciones basado en el costo por hora y duración del turno
        base_cost = 0.0
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Cálculo real del costo de esta asignación
                assignment_cost = employee.hourly_cost * shift.duration_hours
                assignment.cost = assignment_cost  # Actualizar el costo en la asignación
                base_cost += assignment_cost
        
        # Asegurar que haya al menos un costo base mínimo
        base_cost = max(10.0, base_cost)
        
        # Añadir penalización por violaciones de restricciones, más significativa ahora
        # Cada violación incrementa el costo en un porcentaje del costo base
        violation_penalty = 0.0
        if validation_result.violations > 0:
            # 15% del costo base por cada violación
            violation_factor = 0.15 * validation_result.violations
            violation_penalty = base_cost * violation_factor
        
        # Establecer el costo total
        total_cost = base_cost + violation_penalty
        solution.total_cost = total_cost
        
        logger.info(f"Optimización completada. Costo base: {base_cost:.2f}, Penalización: {violation_penalty:.2f}, " 
                   f"Costo total: {total_cost:.2f}, Violaciones: {solution.constraint_violations}")
        
        return solution