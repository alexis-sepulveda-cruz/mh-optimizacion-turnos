import random
import logging
import copy
from typing import List, Dict, Any, Set, Tuple
from collections import deque

from ..optimizer_strategy import OptimizerStrategy
from ...models.solution import Solution
from ...models.employee import Employee
from ...models.shift import Shift
from ...models.assignment import Assignment


logger = logging.getLogger(__name__)


class TabuSearchOptimizer(OptimizerStrategy):
    """Tabu Search implementation for shift optimization."""
    
    def get_name(self) -> str:
        return "Tabu Search Optimizer"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "max_iterations": 1000,
            "tabu_tenure": 20,
            "neighborhood_size": 30,
            "max_iterations_without_improvement": 200
        }
    
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimize shift assignments using Tabu Search."""
        if config is None:
            config = self.get_default_config()
        
        # Extract configuration parameters
        max_iterations = config.get("max_iterations", 1000)
        tabu_tenure = config.get("tabu_tenure", 20)
        neighborhood_size = config.get("neighborhood_size", 30)
        max_no_improve = config.get("max_iterations_without_improvement", 200)
        
        logger.info(f"Starting tabu search optimization with max {max_iterations} iterations")
        
        # Initialize tabu list as a fixed-size deque
        tabu_list = deque(maxlen=tabu_tenure)
        
        # Create initial solution
        current_solution = self._create_initial_solution(employees, shifts)
        best_solution = current_solution.clone()
        
        current_fitness = self._calculate_fitness(current_solution, employees, shifts)
        best_fitness = current_fitness
        
        iterations_without_improvement = 0
        
        for iteration in range(max_iterations):
            # Generate neighborhood
            neighbors = self._generate_neighborhood(current_solution, employees, shifts, neighborhood_size)
            
            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_fitness = float('-inf')
            best_move = None
            
            for neighbor, move in neighbors:
                # Check if move is tabu
                if move in tabu_list:
                    continue
                    
                neighbor_fitness = self._calculate_fitness(neighbor, employees, shifts)
                
                # Accept if better than the best found so far
                if neighbor_fitness > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
                
                # Aspiration criterion: accept if better than the global best, even if tabu
                if neighbor_fitness > best_fitness and move in tabu_list:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
                    break
            
            # If no valid moves found
            if best_neighbor is None:
                logger.info(f"No valid moves found at iteration {iteration}")
                break
                
            # Move to the best neighbor
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            
            # Add the move to the tabu list
            tabu_list.append(best_move)
            
            # Update best solution if improved
            if current_fitness > best_fitness:
                best_solution = current_solution.clone()
                best_fitness = current_fitness
                iterations_without_improvement = 0
                logger.info(f"Iteration {iteration}: Found new best solution with fitness {best_fitness:.4f}")
            else:
                iterations_without_improvement += 1
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Current fitness = {current_fitness:.4f}, Best fitness = {best_fitness:.4f}")
            
            # Stop if no improvement for a while
            if iterations_without_improvement >= max_no_improve:
                logger.info(f"Stopping early - no improvement for {max_no_improve} iterations")
                break
        
        logger.info(f"Tabu search complete after {iteration+1} iterations. Best fitness: {best_fitness:.4f}")
        
        return best_solution
    
    def _create_initial_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Create an initial solution using a greedy approach."""
        solution = Solution()
        
        # Sort shifts by priority (descending)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        for shift in sorted_shifts:
            # Find qualified employees for this shift
            qualified_employees = [
                e for e in employees 
                if e.is_available(shift.day, shift.name) and shift.required_skills.issubset(e.skills)
            ]
            
            # Sort by hourly cost (ascending)
            qualified_employees.sort(key=lambda e: e.hourly_cost)
            
            # Assign required number of employees
            needed = min(shift.required_employees, len(qualified_employees))
            for i in range(needed):
                employee = qualified_employees[i]
                assignment = Assignment(
                    employee_id=employee.id,
                    shift_id=shift.id,
                    cost=employee.hourly_cost * shift.duration_hours
                )
                solution.add_assignment(assignment)
        
        solution.calculate_total_cost()
        return solution
    
    def _generate_neighborhood(self, current_solution: Solution, 
                             employees: List[Employee], shifts: List[Shift], 
                             neighborhood_size: int) -> List[Tuple[Solution, Tuple]]:
        """Generate neighborhood solutions by applying different moves."""
        neighbors = []
        shift_dict = {s.id: s for s in shifts}
        
        # Group assignments by shift
        shift_assignments = {}
        for a in current_solution.assignments:
            if a.shift_id not in shift_assignments:
                shift_assignments[a.shift_id] = []
            shift_assignments[a.shift_id].append(a)
        
        # Strategy 1: Replace Employee
        for i in range(min(neighborhood_size // 2, len(current_solution.assignments))):
            # Pick random assignment to replace
            assignment = random.choice(current_solution.assignments)
            shift = shift_dict.get(assignment.shift_id)
            
            if shift:
                # Find potential replacements
                available_employees = [
                    e for e in employees 
                    if e.id != assignment.employee_id and
                    e.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(e.skills)
                ]
                
                if available_employees:
                    new_employee = random.choice(available_employees)
                    
                    # Create new solution with this replacement
                    new_solution = current_solution.clone()
                    
                    # Find and replace the assignment
                    for j, a in enumerate(new_solution.assignments):
                        if a.employee_id == assignment.employee_id and a.shift_id == assignment.shift_id:
                            new_solution.assignments[j] = Assignment(
                                employee_id=new_employee.id,
                                shift_id=assignment.shift_id,
                                cost=new_employee.hourly_cost * shift.duration_hours
                            )
                            break
                    
                    # Generate a tuple that uniquely identifies this move: (original_emp_id, shift_id, new_emp_id)
                    move = (assignment.employee_id, assignment.shift_id, new_employee.id)
                    neighbors.append((new_solution, move))
        
        # Strategy 2: Swap Assignments
        for i in range(min(neighborhood_size // 2, len(shift_assignments))):
            if len(shift_assignments) < 2:
                continue
                
            # Pick two different shifts
            shift_ids = random.sample(list(shift_assignments.keys()), 2)
            
            if len(shift_assignments[shift_ids[0]]) > 0 and len(shift_assignments[shift_ids[1]]) > 0:
                # Pick one assignment from each shift
                assignment1 = random.choice(shift_assignments[shift_ids[0]])
                assignment2 = random.choice(shift_assignments[shift_ids[1]])
                
                shift1 = shift_dict.get(assignment1.shift_id)
                shift2 = shift_dict.get(assignment2.shift_id)
                
                employee_dict = {e.id: e for e in employees}
                employee1 = employee_dict.get(assignment1.employee_id)
                employee2 = employee_dict.get(assignment2.employee_id)
                
                # Check if swap is valid
                if (employee1 and employee2 and shift1 and shift2 and
                    employee1.is_available(shift2.day, shift2.name) and
                    employee2.is_available(shift1.day, shift1.name) and
                    shift1.required_skills.issubset(employee2.skills) and
                    shift2.required_skills.issubset(employee1.skills)):
                    
                    # Create new solution with the swap
                    new_solution = current_solution.clone()
                    
                    # Find and replace the assignments
                    for j, a in enumerate(new_solution.assignments):
                        if a.employee_id == assignment1.employee_id and a.shift_id == assignment1.shift_id:
                            new_solution.assignments[j] = Assignment(
                                employee_id=employee2.id,
                                shift_id=shift1.id,
                                cost=employee2.hourly_cost * shift1.duration_hours
                            )
                        elif a.employee_id == assignment2.employee_id and a.shift_id == assignment2.shift_id:
                            new_solution.assignments[j] = Assignment(
                                employee_id=employee1.id,
                                shift_id=shift2.id,
                                cost=employee1.hourly_cost * shift2.duration_hours
                            )
                    
                    # Generate a tuple that uniquely identifies this swap:
                    # ((emp1_id, shift1_id), (emp2_id, shift2_id))
                    move = ((assignment1.employee_id, assignment1.shift_id),
                           (assignment2.employee_id, assignment2.shift_id))
                    
                    neighbors.append((new_solution, move))
        
        # If we couldn't generate enough neighbors, add some random solutions
        while len(neighbors) < neighborhood_size:
            new_solution = self._create_initial_solution(employees, shifts)
            # Use a unique identifier for this random solution
            move = ("random", random.randint(0, 10000))
            neighbors.append((new_solution, move))
        
        return neighbors
    
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