import random
import logging
from typing import List, Dict, Any
from collections import defaultdict

from mh_optimizacion_turnos.domain.services.optimizer_strategy import OptimizerStrategy
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.models.assignment import Assignment


logger = logging.getLogger(__name__)


class GraspOptimizer(OptimizerStrategy):
    """Implementación de GRASP (Greedy Randomized Adaptive Search Procedure) para la optimización de turnos."""
    
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
        """Optimiza la asignación de turnos usando GRASP."""
        if config is None:
            config = self.get_default_config()
        
        # Extraer parámetros de configuración
        max_iterations = config.get("max_iterations", 100)
        alpha = config.get("alpha", 0.3)
        local_search_iterations = config.get("local_search_iterations", 50)
        
        logger.info(f"Iniciando optimización GRASP con {max_iterations} iteraciones y alpha={alpha}")
        
        best_solution = None
        best_fitness = float('-inf')
        
        # Ciclo principal de GRASP
        for iteration in range(max_iterations):
            # Fase 1: Construcción - Construcción greedy aleatorizada de una solución
            solution = self._greedy_randomized_construction(employees, shifts, alpha)
            
            # Fase 2: Búsqueda Local - Mejorar la solución usando búsqueda local
            solution = self._local_search(solution, employees, shifts, local_search_iterations)
            
            # Evaluar solución
            fitness = self._calculate_fitness(solution, employees, shifts)
            
            # Actualizar mejor solución si ha mejorado
            if best_solution is None or fitness > best_fitness:
                best_solution = solution.clone()
                best_fitness = fitness
                logger.info(f"Iteración {iteration}: Se encontró nueva mejor solución con fitness {best_fitness:.4f}")
            
            # Registrar progreso periódicamente
            if iteration % 10 == 0 and iteration > 0:
                logger.info(f"Completadas {iteration}/{max_iterations} iteraciones. Mejor fitness: {best_fitness:.4f}")
        
        logger.info(f"Optimización GRASP completada. Fitness de la mejor solución: {best_fitness:.4f}")
        
        return best_solution
    
    def _greedy_randomized_construction(self, employees: List[Employee], shifts: List[Shift], alpha: float) -> Solution:
        """Construye una solución de manera greedy pero con componente aleatorio."""
        solution = Solution()
        
        # Ordenar turnos por prioridad (descendente)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        # Rastrear qué empleados ya han sido asignados a cada tipo de turno en cada día
        day_shift_employees = defaultdict(set)
        
        for shift in sorted_shifts:
            # Encontrar empleados calificados para este turno
            qualified_employees = [
                e for e in employees 
                if e.is_available(shift.day, shift.name) and shift.required_skills.issubset(e.skills)
            ]
            
            # Clave para identificar un tipo de turno en un día específico
            day_shift_key = (shift.day, shift.name)
            
            # Filtrar empleados que ya están asignados a este turno en este día
            available_employees = [e for e in qualified_employees if e.id not in day_shift_employees[day_shift_key]]
            
            if not available_employees:
                continue
                
            # Ordenar empleados por costo (ascendente) - menor costo es mejor
            sorted_employees = sorted(available_employees, key=lambda e: e.hourly_cost)
            
            # Determinar el tamaño de la RCL (Lista Restringida de Candidatos)
            rcl_size = max(1, int(alpha * len(sorted_employees)))
            
            # Asignar el número requerido de empleados desde la RCL
            needed_employees = min(shift.required_employees, len(available_employees))
            for _ in range(needed_employees):
                # Elegir aleatoriamente de los mejores candidatos (RCL)
                candidate_pool = sorted_employees[:rcl_size]
                if not candidate_pool:
                    break
                    
                selected_employee = random.choice(candidate_pool)
                
                # Añadir asignación a la solución
                assignment = Assignment(
                    employee_id=selected_employee.id,
                    shift_id=shift.id,
                    cost=selected_employee.hourly_cost * shift.duration_hours
                )
                solution.add_assignment(assignment)
                
                # Registrar que este empleado ya ha sido asignado a este turno en este día
                day_shift_employees[day_shift_key].add(selected_employee.id)
                
                # Eliminar el empleado seleccionado de los candidatos para evitar doble asignación
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
        
        # Rastrear asignaciones actuales por día y tipo de turno
        day_shift_employees = defaultdict(set)
        for assignment in current_solution.assignments:
            shift = shift_dict.get(assignment.shift_id)
            if shift:
                day_shift_key = (shift.day, shift.name)
                day_shift_employees[day_shift_key].add(assignment.employee_id)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Intentar mejorar reemplazando empleados
            for i, assignment in enumerate(current_solution.assignments):
                shift = shift_dict.get(assignment.shift_id)
                current_employee_id = assignment.employee_id
                
                if not shift:
                    continue
                
                # Clave para identificar un tipo de turno en un día específico
                day_shift_key = (shift.day, shift.name)
                    
                # Encontrar reemplazos potenciales que no estén ya asignados a este turno en este día
                potential_replacements = [
                    e for e in employees 
                    if e.id != current_employee_id and
                    e.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(e.skills) and
                    e.id not in day_shift_employees[day_shift_key]
                ]
                
                if not potential_replacements:
                    continue
                
                # Probar cada reemplazo potencial
                for new_employee in potential_replacements:
                    # Crear una nueva solución con este reemplazo
                    new_solution = current_solution.clone()
                    
                    # Reemplazar la asignación
                    for j, a in enumerate(new_solution.assignments):
                        if j == i:  # Esta es la asignación que queremos reemplazar
                            new_solution.assignments[j] = Assignment(
                                employee_id=new_employee.id,
                                shift_id=shift.id,
                                cost=new_employee.hourly_cost * shift.duration_hours
                            )
                            break
                    
                    # Calcular fitness de la nueva solución
                    new_fitness = self._calculate_fitness(new_solution, employees, shifts)
                    
                    # Aceptar si mejora
                    if new_fitness > current_fitness:
                        current_solution = new_solution
                        current_fitness = new_fitness
                        
                        # Actualizar registro de asignaciones
                        day_shift_employees[day_shift_key].remove(current_employee_id)
                        day_shift_employees[day_shift_key].add(new_employee.id)
                        
                        improved = True
                        break
                
                if improved:
                    break
            
            # Si no se encontró mejora mediante reemplazo, intentar añadir nuevas asignaciones
            if not improved:
                for shift in shifts:
                    # Verificar si este turno necesita más empleados
                    assigned_count = len([a for a in current_solution.assignments if a.shift_id == shift.id])
                    
                    if assigned_count < shift.required_employees:
                        # Clave para identificar un tipo de turno en un día específico
                        day_shift_key = (shift.day, shift.name)
                        
                        # Encontrar empleados que no estén ya asignados a este turno
                        available_employees = [
                            e for e in employees 
                            if e.id not in day_shift_employees[day_shift_key] and
                            e.is_available(shift.day, shift.name) and
                            shift.required_skills.issubset(e.skills)
                        ]
                        
                        if available_employees:
                            # Ordenar por costo (ascendente)
                            sorted_employees = sorted(available_employees, key=lambda e: e.hourly_cost)
                            
                            # Probar el mejor candidato
                            if sorted_employees:
                                new_employee = sorted_employees[0]
                                new_solution = current_solution.clone()
                                new_assignment = Assignment(
                                    employee_id=new_employee.id,
                                    shift_id=shift.id,
                                    cost=new_employee.hourly_cost * shift.duration_hours
                                )
                                new_solution.add_assignment(new_assignment)
                                
                                # Calcular fitness de la nueva solución
                                new_fitness = self._calculate_fitness(new_solution, employees, shifts)
                                
                                # Aceptar si mejora
                                if new_fitness > current_fitness:
                                    current_solution = new_solution
                                    current_fitness = new_fitness
                                    
                                    # Actualizar registro de asignaciones
                                    day_shift_employees[day_shift_key].add(new_employee.id)
                                    
                                    improved = True
                                    break
                
                if improved:
                    continue
            
            # Si no se encontró ninguna mejora, detener la búsqueda local
            if not improved:
                break
        
        return current_solution
    
    def _calculate_fitness(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> float:
        """Calcula la puntuación de fitness para una solución."""
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
        
        # Tracking de asignaciones duplicadas
        day_shift_employees = defaultdict(set)
        duplicate_assignments = 0
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Clave para identificar un tipo de turno en un día específico
                day_shift_key = (shift.day, shift.name)
                
                # Verificar si este empleado ya está asignado a este turno en este día
                if emp_id in day_shift_employees[day_shift_key]:
                    duplicate_assignments += 1
                
                # Registrar esta asignación
                day_shift_employees[day_shift_key].add(emp_id)
                
                # Registrar horas
                if emp_id not in employee_hours:
                    employee_hours[emp_id] = 0
                employee_hours[emp_id] += shift.duration_hours
                
                # Registrar días
                if emp_id not in employee_days:
                    employee_days[emp_id] = set()
                employee_days[emp_id].add(shift.day)
                
                # Verificar disponibilidad del empleado
                if not employee.is_available(shift.day, shift.name):
                    employee_availability_violations += 1
                
                # Verificar habilidades requeridas
                if not shift.required_skills.issubset(employee.skills):
                    employee_skills_violations += len(shift.required_skills - employee.skills)
        
        # Penalizar asignaciones duplicadas severamente
        if duplicate_assignments > 0:
            violations_count += duplicate_assignments * 10
        
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