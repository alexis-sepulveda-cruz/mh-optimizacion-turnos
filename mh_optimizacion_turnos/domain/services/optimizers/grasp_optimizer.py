import random
import logging
from typing import List, Dict, Any, Tuple
import copy

from ..optimizer_strategy import OptimizerStrategy
from ..solution_validator import SolutionValidator
from ...models.solution import Solution
from ...models.employee import Employee
from ...models.shift import Shift
from ...models.assignment import Assignment


logger = logging.getLogger(__name__)


class GraspOptimizer(OptimizerStrategy):
    """GRASP (Greedy Randomized Adaptive Search Procedure) implementation for shift optimization."""
    
    def get_name(self) -> str:
        return "GRASP Optimizer"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "max_iterations": 100,
            "alpha": 0.3,  # Restricción de la lista de candidatos (0=completamente greedy, 1=completamente aleatorio)
            "local_search_iterations": 50
        }
    
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimize shift assignments using GRASP."""
        if config is None:
            config = self.get_default_config()
        
        # Extract configuration parameters
        max_iterations = config.get("max_iterations", 100)
        alpha = config.get("alpha", 0.3)
        local_search_iterations = config.get("local_search_iterations", 50)
        
        logger.info(f"Starting GRASP optimization with {max_iterations} iterations and alpha={alpha}")
        
        best_solution = None
        best_fitness = float('-inf')
        
        # Main GRASP loop
        for iteration in range(max_iterations):
            # Phase 1: Construction - Greedy randomized construction of a solution
            solution = self._greedy_randomized_construction(employees, shifts, alpha)
            
            # Phase 2: Local Search - Improve the solution using local search
            solution = self._local_search(solution, employees, shifts, local_search_iterations)
            
            # Evaluate solution
            fitness = self._calculate_fitness(solution, employees, shifts)
            
            # Update best solution if improved
            if best_solution is None or fitness > best_fitness:
                best_solution = solution.clone()
                best_fitness = fitness
                logger.info(f"Iteration {iteration}: Found new best solution with fitness {best_fitness:.4f}")
            
            # Log progress periodically
            if iteration % 10 == 0 and iteration > 0:
                logger.info(f"Completed {iteration}/{max_iterations} iterations. Best fitness: {best_fitness:.4f}")
        
        logger.info(f"GRASP optimization complete. Best solution fitness: {best_fitness:.4f}")
        
        return best_solution
    
    def _greedy_randomized_construction(self, employees: List[Employee], shifts: List[Shift], alpha: float) -> Solution:
        """Construye una solución de manera greedy pero con componente aleatorio."""
        solution = Solution()
        
        # Sort shifts by priority (descending)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        for shift in sorted_shifts:
            # Find qualified employees for this shift
            qualified_employees = [
                e for e in employees 
                if e.is_available(shift.day, shift.name) and shift.required_skills.issubset(e.skills)
            ]
            
            if not qualified_employees:
                continue
                
            # Sort employees by cost (ascending) - lower cost is better
            sorted_employees = sorted(qualified_employees, key=lambda e: e.hourly_cost)
            
            # Determine RCL (Restricted Candidate List) size
            rcl_size = max(1, int(alpha * len(sorted_employees)))
            
            # Assign required number of employees from RCL
            needed_employees = min(shift.required_employees, len(qualified_employees))
            for _ in range(needed_employees):
                # Choose randomly from the best candidates (RCL)
                candidate_pool = sorted_employees[:rcl_size]
                if not candidate_pool:
                    break
                    
                selected_employee = random.choice(candidate_pool)
                
                # Add assignment to solution
                assignment = Assignment(
                    employee_id=selected_employee.id,
                    shift_id=shift.id,
                    cost=selected_employee.hourly_cost * shift.duration_hours
                )
                solution.add_assignment(assignment)
                
                # Remove selected employee from candidates to avoid double assignment
                sorted_employees.remove(selected_employee)
        
        solution.calculate_total_cost()
        return solution
    
    def _local_search(self, solution: Solution, employees: List[Employee], 
                     shifts: List[Shift], max_iterations: int) -> Solution:
        """Mejora la solución mediante búsqueda local."""
        current_solution = solution.clone()
        current_fitness = self._calculate_fitness(current_solution, employees, shifts)
        
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try to improve by replacing employees
            for i, assignment in enumerate(current_solution.assignments):
                shift = shift_dict.get(assignment.shift_id)
                current_employee_id = assignment.employee_id
                
                if not shift:
                    continue
                    
                # Find potential replacements
                potential_replacements = [
                    e for e in employees 
                    if e.id != current_employee_id and
                    e.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(e.skills)
                ]
                
                if not potential_replacements:
                    continue
                
                # Try each potential replacement
                for new_employee in potential_replacements:
                    # Create a new solution with this replacement
                    new_solution = current_solution.clone()
                    
                    # Replace the assignment
                    for j, a in enumerate(new_solution.assignments):
                        if j == i:  # This is the assignment we want to replace
                            new_solution.assignments[j] = Assignment(
                                employee_id=new_employee.id,
                                shift_id=shift.id,
                                cost=new_employee.hourly_cost * shift.duration_hours
                            )
                            break
                    
                    # Calculate fitness of new solution
                    new_fitness = self._calculate_fitness(new_solution, employees, shifts)
                    
                    # Accept if improved
                    if new_fitness > current_fitness:
                        current_solution = new_solution
                        current_fitness = new_fitness
                        improved = True
                        break
                
                if improved:
                    break
            
            # If no improvement was found by replacement, try adding new assignments
            if not improved:
                for shift in shifts:
                    # Check if this shift needs more employees
                    assigned_count = len([a for a in current_solution.assignments if a.shift_id == shift.id])
                    
                    if assigned_count < shift.required_employees:
                        # Find employees not already assigned to this shift
                        current_assigned_ids = set(
                            a.employee_id for a in current_solution.assignments if a.shift_id == shift.id
                        )
                        
                        available_employees = [
                            e for e in employees 
                            if e.id not in current_assigned_ids and
                            e.is_available(shift.day, shift.name) and
                            shift.required_skills.issubset(e.skills)
                        ]
                        
                        if available_employees:
                            # Sort by cost (ascending)
                            sorted_employees = sorted(available_employees, key=lambda e: e.hourly_cost)
                            
                            # Try the best candidate
                            if sorted_employees:
                                new_solution = current_solution.clone()
                                new_assignment = Assignment(
                                    employee_id=sorted_employees[0].id,
                                    shift_id=shift.id,
                                    cost=sorted_employees[0].hourly_cost * shift.duration_hours
                                )
                                new_solution.add_assignment(new_assignment)
                                
                                # Calculate fitness of new solution
                                new_fitness = self._calculate_fitness(new_solution, employees, shifts)
                                
                                # Accept if improved
                                if new_fitness > current_fitness:
                                    current_solution = new_solution
                                    current_fitness = new_fitness
                                    improved = True
                                    break
                
                if improved:
                    continue
            
            # If no improvement was found, stop local search
            if not improved:
                break
        
        return current_solution
    
    def _calculate_fitness(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> float:
        """Calculate fitness score for a solution."""
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        # Calcular el costo base de la solución de manera más realista
        base_cost = 0.0
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Actualizar el costo real de esta asignación
                assignment_cost = employee.hourly_cost * shift.duration_hours
                assignment.cost = assignment_cost
                base_cost += assignment_cost
        
        # Asegurar un costo base mínimo
        base_cost = max(10.0, base_cost)
        
        # Inicializar contador de violaciones
        violations_count = 0
        
        # Penalizaciones por falta de cobertura (más severas)
        coverage_penalty = 0
        for shift in shifts:
            assigned_count = len(solution.get_shift_employees(shift.id))
            if assigned_count < shift.required_employees:
                shortage = shift.required_employees - assigned_count
                # Penalización basada en la importancia del turno y la cantidad de personal faltante
                penalty_factor = shift.priority * 25.0 * shortage
                coverage_penalty += penalty_factor
                violations_count += shortage
        
        # Tracking de horas y días por empleado
        employee_hours = {}
        employee_days = {}
        employee_availability_violations = 0
        employee_skills_violations = 0
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Track hours
                if emp_id not in employee_hours:
                    employee_hours[emp_id] = 0
                employee_hours[emp_id] += shift.duration_hours
                
                # Track days
                if emp_id not in employee_days:
                    employee_days[emp_id] = set()
                employee_days[emp_id].add(shift.day)
                
                # Verificar disponibilidad del empleado
                if not employee.is_available(shift.day, shift.name):
                    employee_availability_violations += 1
                
                # Verificar habilidades requeridas
                if not shift.required_skills.issubset(employee.skills):
                    employee_skills_violations += len(shift.required_skills - employee.skills)
        
        # Contar violaciones de horas máximas
        hours_violations = 0
        for emp_id, hours in employee_hours.items():
            if emp_id in employee_dict:
                max_hours = employee_dict[emp_id].max_hours_per_week
                if hours > max_hours:
                    hours_violations += (hours - max_hours)
        
        # Contar violaciones de días consecutivos
        consecutive_days_violations = 0
        for emp_id, days in employee_days.items():
            if emp_id in employee_dict:
                max_consecutive = employee_dict[emp_id].max_consecutive_days
                if len(days) > max_consecutive:
                    consecutive_days_violations += (len(days) - max_consecutive)
        
        # Actualizar contador total de violaciones
        violations_count += (
            employee_availability_violations +
            employee_skills_violations +
            hours_violations +
            consecutive_days_violations
        )
        
        # Calcular penalización total
        total_penalty = coverage_penalty + (base_cost * 0.1 * violations_count)
        
        # Establecer el costo total y el número de violaciones en la solución
        total_cost = base_cost + total_penalty
        solution.total_cost = total_cost
        solution.constraint_violations = violations_count
        
        # Calcular fitness (inversamente proporcional al costo)
        # Usamos una función más suave que 1/x para evitar valores extremos
        fitness_score = 1000.0 / (1.0 + total_cost / 100.0)
        
        # Añadir bonificaciones por preferencias de empleados
        preference_bonus = 0.0
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                preference_score = employee.get_preference_score(shift.day, shift.name)
                # Los beneficios por preferencias ahora son proporcionales al fitness
                preference_bonus += preference_score * 0.01 * fitness_score
        
        # Añadir el bono de preferencias al fitness
        final_fitness = fitness_score + preference_bonus
        
        # Guardar el fitness en la solución para referencias futuras
        solution.fitness_score = final_fitness
        
        return max(0.1, final_fitness)