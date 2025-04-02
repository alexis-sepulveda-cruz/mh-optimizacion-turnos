from typing import Dict, Any, List, Optional
import logging

from ..models.solution import Solution
from ..models.employee import Employee
from ..models.shift import Shift
from ..repositories.employee_repository import EmployeeRepository
from ..repositories.shift_repository import ShiftRepository
from .optimizer_strategy import OptimizerStrategy
from .solution_validator import SolutionValidator


logger = logging.getLogger(__name__)


class ShiftOptimizerService:
    """Service for optimizing shift assignments using configurable metaheuristic algorithms."""
    
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
        """Set the optimization strategy to use.
        
        This allows switching between different metaheuristic algorithms
        at runtime based on user preferences or problem characteristics.
        """
        self.optimizer_strategy = optimizer_strategy
        logger.info(f"Optimizer strategy set to: {optimizer_strategy.get_name()}")
    
    def generate_optimal_schedule(
        self,
        start_date: str = None,
        end_date: str = None,
        config: Dict[str, Any] = None
    ) -> Solution:
        """Generate an optimal schedule for the given period using the selected strategy.
        
        Args:
            start_date: Start date for the scheduling period
            end_date: End date for the scheduling period
            config: Configuration parameters for the optimization algorithm
            
        Returns:
            A Solution object containing the optimized assignments
            
        Raises:
            ValueError: If no optimizer strategy has been set
        """
        if not self.optimizer_strategy:
            raise ValueError("No optimizer strategy has been set")
        
        # Get all relevant employees and shifts
        employees = self.employee_repository.get_all()
        
        if start_date and end_date:
            shifts = self.shift_repository.get_shifts_by_period(start_date, end_date)
        else:
            shifts = self.shift_repository.get_all()
        
        logger.info(f"Starting optimization with {len(employees)} employees and {len(shifts)} shifts")
        
        # Use the strategy to optimize
        solution = self.optimizer_strategy.optimize(employees, shifts, config)
        
        # Validate solution
        validation_result = self.solution_validator.validate(solution, employees, shifts)
        if not validation_result.is_valid:
            logger.warning(f"Generated solution has {validation_result.violations} constraint violations")
        
        solution.constraint_violations = validation_result.violations
        
        # Calcular el costo real de la solución de una manera más precisa
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        # Base cost: Costo real de las asignaciones basado en el costo por hora y duración del turno
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
        
        logger.info(f"Optimization complete. Base cost: {base_cost:.2f}, Penalty: {violation_penalty:.2f}, " 
                   f"Total cost: {total_cost:.2f}, Violations: {solution.constraint_violations}")
        
        return solution