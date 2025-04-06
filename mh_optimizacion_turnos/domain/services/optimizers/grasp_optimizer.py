import random
import logging
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict

from mh_optimizacion_turnos.domain.services.optimizer_strategy import OptimizerStrategy
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.models.assignment import Assignment


logger = logging.getLogger(__name__)


class GraspOptimizer(OptimizerStrategy):
    """Implementación de GRASP (Greedy Randomized Adaptive Search Procedure) para la optimización de turnos.
    
    Implementa restricciones duras (hard constraints) para garantizar soluciones factibles.
    """
    
    def get_name(self) -> str:
        return "GRASP Optimizer"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "max_iterations": 100,
            "alpha": 0.3,  # Restricción de la lista de candidatos (0=completamente greedy, 1=completamente aleatorio)
            "local_search_iterations": 50,
            "max_construction_attempts": 100,  # Máximo de intentos para construir una solución válida
            "validation_timeout": 10,  # Tiempo máximo (segundos) para generar una solución válida
            "metrics": {
                "enabled": True,
                "track_evaluations": True,
                "track_construction_success": True
            }
        }
    
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimiza la asignación de turnos usando GRASP con restricciones duras."""
        if config is None:
            config = self.get_default_config()
        
        # Extraer parámetros de configuración
        max_iterations = config.get("max_iterations", 100)
        alpha = config.get("alpha", 0.3)
        local_search_iterations = config.get("local_search_iterations", 50)
        max_construction_attempts = config.get("max_construction_attempts", 100)
        validation_timeout = config.get("validation_timeout", 10)
        
        # Inicialización de métricas
        metrics = {
            "objective_evaluations": 0,
            "construction_attempts": 0,
            "construction_success_rate": 0,
            "valid_solutions_found": 0,
            "local_search_improvements": 0
        }
        
        # Crear validador
        validator = SolutionValidator()
        
        logger.info(f"Iniciando optimización GRASP con {max_iterations} iteraciones y alpha={alpha}")
        logger.info(f"Usando enfoque de restricciones duras (solo soluciones factibles)")
        
        best_solution = None
        best_fitness = float('-inf')
        
        # Ciclo principal de GRASP
        start_time = time.time()
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Fase 1: Construcción - Construcción greedy aleatorizada de una solución VÁLIDA
            solution = None
            construction_attempts = 0
            
            while solution is None and construction_attempts < max_construction_attempts:
                construction_attempts += 1
                metrics["construction_attempts"] += 1
                
                # Intentar construir una solución candidata
                candidate = self._greedy_randomized_construction(employees, shifts, alpha)
                
                # Validar solución
                validation_result = validator.validate(candidate, employees, shifts)
                
                if validation_result.is_valid:
                    solution = candidate
                    metrics["valid_solutions_found"] += 1
                    
                # Verificar timeout
                if time.time() - iteration_start > validation_timeout:
                    logger.warning(f"Timeout en la construcción de solución válida en iteración {iteration}")
                    break
            
            # Actualizar métrica de tasa de éxito
            current_success_rate = metrics["valid_solutions_found"] / metrics["construction_attempts"]
            if metrics["construction_success_rate"] == 0:
                metrics["construction_success_rate"] = current_success_rate
            else:
                metrics["construction_success_rate"] = (
                    metrics["construction_success_rate"] * 0.7 + current_success_rate * 0.3
                )
            
            # Si no se pudo construir una solución válida, pasar a la siguiente iteración
            if solution is None:
                logger.warning(f"No se pudo construir una solución válida en la iteración {iteration} "
                             f"después de {construction_attempts} intentos")
                continue
            
            # Fase 2: Búsqueda Local - Mejorar la solución usando búsqueda local (manteniendo factibilidad)
            improved_solution = self._local_search(
                solution, employees, shifts, local_search_iterations, validator
            )
            
            # Si la búsqueda local generó una solución inválida, usar la original
            if improved_solution is None:
                improved_solution = solution
            else:
                metrics["local_search_improvements"] += 1
            
            # Evaluar solución
            fitness = self._calculate_fitness(improved_solution, employees, shifts)
            metrics["objective_evaluations"] += 1
            
            # Actualizar mejor solución si ha mejorado
            if best_solution is None or fitness > best_fitness:
                best_solution = improved_solution.clone()
                best_fitness = fitness
                logger.info(f"Iteración {iteration}: Nueva mejor solución con fitness {best_fitness:.4f}")
            
            # Registrar progreso periódicamente
            if iteration % 10 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                logger.info(f"Completadas {iteration}/{max_iterations} iteraciones en {elapsed:.2f}s. "
                          f"Mejor fitness: {best_fitness:.4f}, "
                          f"Tasa de construcción: {metrics['construction_success_rate']:.2f}")
        
        if best_solution is None:
            raise ValueError("No se pudo encontrar ninguna solución válida. Las restricciones pueden ser demasiado estrictas.")
        
        # Registrar métricas finales
        logger.info(f"Optimización GRASP completada. Fitness de la mejor solución: {best_fitness:.4f}")
        logger.info(f"Métricas: {metrics['valid_solutions_found']}/{metrics['construction_attempts']} "
                  f"soluciones válidas construidas. {metrics['local_search_improvements']} mejoras por búsqueda local.")
        
        return best_solution
    
    def _greedy_randomized_construction(self, employees: List[Employee], shifts: List[Shift], alpha: float) -> Solution:
        """Construye una solución de manera greedy aleatorizada, intentando respetar restricciones."""
        solution = Solution()
        
        # Ordenar turnos por prioridad (descendente)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        # Rastrear qué empleados ya han sido asignados a cada tipo de turno en cada día
        day_shift_employees = defaultdict(set)
        
        # Rastrear horas y días asignados por empleado
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
        for shift in sorted_shifts:
            # Clave para identificar un tipo de turno en un día específico
            day_shift_key = (shift.day, shift.name)
            
            # Determinar cuántos empleados se necesitan asignar para este turno
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue  # Este turno ya tiene los empleados requeridos
            
            # Encontrar empleados calificados para este turno que:
            # 1. No estén ya asignados a este turno en este día
            # 2. Estén disponibles para este turno
            # 3. Tengan las habilidades requeridas
            # 4. No excedan sus horas máximas
            # 5. No excedan sus días consecutivos
            qualified_employees = []
            
            for employee in employees:
                # Verificar si ya está asignado a este turno en este día
                if employee.id in day_shift_employees[day_shift_key]:
                    continue
                    
                # Verificar disponibilidad
                if not employee.is_available(shift.day, shift.name):
                    continue
                
                # Verificar habilidades
                if not shift.required_skills.issubset(employee.skills):
                    continue
                
                # Verificar horas máximas
                if employee_hours[employee.id] + shift.duration_hours > employee.max_hours_per_week:
                    continue
                
                # Verificar días consecutivos (simplificado)
                if shift.day in employee_days[employee.id]:
                    # Ya tiene un turno ese día, no es problema
                    pass
                elif len(employee_days[employee.id]) >= employee.max_consecutive_days:
                    # Ya alcanzó el máximo de días, pero podría ser compatible
                    # Simplificación: permitimos siempre que no exceda el número total de días
                    if len(employee_days[employee.id]) < 7:  # Asumiendo una semana
                        pass
                    else:
                        continue
                
                # Este empleado cumple con todas las condiciones
                qualified_employees.append(employee)
            
            if not qualified_employees and needed > 0:
                # No hay suficientes empleados calificados para este turno
                # En un enfoque de restricciones duras, esto significa que la solución no será válida
                # Pero continuamos para intentar generar una solución lo más completa posible
                # (será rechazada en la validación)
                continue
            
            # Ordenar empleados por costo (ascendente) - menor costo es mejor
            sorted_employees = sorted(qualified_employees, key=lambda e: e.hourly_cost)
            
            if not sorted_employees:
                continue
            
            # Determinar el tamaño de la RCL (Lista Restringida de Candidatos)
            rcl_size = max(1, int(alpha * len(sorted_employees)))
            rcl = sorted_employees[:rcl_size]
            
            # Asignar empleados
            for _ in range(min(needed, len(rcl))):
                if not rcl:
                    break
                
                # Seleccionar aleatoriamente de la RCL
                selected_employee = random.choice(rcl)
                rcl.remove(selected_employee)  # Evitar seleccionar el mismo empleado dos veces
                
                # Añadir asignación a la solución
                assignment = Assignment(
                    employee_id=selected_employee.id,
                    shift_id=shift.id,
                    cost=selected_employee.hourly_cost * shift.duration_hours
                )
                solution.add_assignment(assignment)
                
                # Actualizar registros
                day_shift_employees[day_shift_key].add(selected_employee.id)
                employee_hours[selected_employee.id] += shift.duration_hours
                employee_days[selected_employee.id].add(shift.day)
        
        return solution
    
    def _local_search(self, solution: Solution, employees: List[Employee], shifts: List[Shift], 
                     max_iterations: int, validator: SolutionValidator) -> Optional[Solution]:
        """Mejora la solución mediante búsqueda local, manteniendo factibilidad."""
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
                    
                # Encontrar reemplazos potenciales que:
                # 1. No estén ya asignados a este turno en este día
                # 2. Estén disponibles para este turno
                # 3. Tengan las habilidades requeridas
                # 4. Tengan mejor costo (o alguna otra característica que mejore la solución)
                potential_replacements = [
                    e for e in employees 
                    if e.id != current_employee_id and
                    e.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(e.skills) and
                    e.id not in day_shift_employees[day_shift_key] and
                    e.hourly_cost < employee_dict[current_employee_id].hourly_cost  # Mejora el costo
                ]
                
                if not potential_replacements:
                    continue
                
                # Probar cada reemplazo potencial
                for new_employee in potential_replacements:
                    # Crear una nueva solución con este reemplazo
                    new_solution = current_solution.clone()
                    
                    # Reemplazar la asignación
                    new_assignments = []
                    for a in new_solution.assignments:
                        if a.shift_id == assignment.shift_id and a.employee_id == current_employee_id:
                            # Reemplazar esta asignación
                            new_assignments.append(Assignment(
                                employee_id=new_employee.id,
                                shift_id=shift.id,
                                cost=new_employee.hourly_cost * shift.duration_hours
                            ))
                        else:
                            # Mantener esta asignación
                            new_assignments.append(a)
                    
                    new_solution.assignments = new_assignments
                    
                    # Validar la nueva solución
                    validation_result = validator.validate(new_solution, employees, shifts)
                    
                    if not validation_result.is_valid:
                        continue  # Ignorar reemplazos que generan soluciones inválidas
                    
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
                # Intentar añadir asignaciones para turnos con cobertura insuficiente
                for shift in shifts:
                    # Verificar si este turno necesita más empleados
                    day_shift_key = (shift.day, shift.name)
                    assigned_count = len(day_shift_employees[day_shift_key])
                    
                    if assigned_count < shift.required_employees:
                        # Encontrar empleados que no estén ya asignados a este turno y cumplan requisitos
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
                                
                                # Validar la nueva solución
                                validation_result = validator.validate(new_solution, employees, shifts)
                                
                                if not validation_result.is_valid:
                                    continue
                                
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
        
        # Validar la solución final (por precaución)
        validation_result = validator.validate(current_solution, employees, shifts)
        
        if validation_result.is_valid:
            return current_solution
        else:
            # Si de algún modo la solución final es inválida, devolver None
            logger.warning("La búsqueda local generó una solución inválida. Usando solución original.")
            return None
    
    def _calculate_fitness(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> float:
        """Calcula la puntuación de fitness para una solución factible (solo considera costo, no violaciones)."""
        # Calcular el costo total de la solución
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        total_cost = 0.0
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Cálculo real del costo de esta asignación
                assignment_cost = employee.hourly_cost * shift.duration_hours
                assignment.cost = assignment_cost
                total_cost += assignment_cost
        
        # Asegurar un costo mínimo
        total_cost = max(10.0, total_cost)
        solution.total_cost = total_cost
        
        # Con restricciones duras, las violaciones siempre son 0
        solution.constraint_violations = 0
        
        # El fitness es inversamente proporcional al costo
        # Usamos esta fórmula para evitar valores extremos
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
                # Los beneficios por preferencias son proporcionales al fitness
                preference_bonus += preference_score * 0.01 * fitness_score
        
        # Añadir el bono de preferencias al fitness
        final_fitness = fitness_score + preference_bonus
        
        # Calcular cobertura de turnos (porcentaje de turnos cubiertos completamente)
        total_shifts = len(shifts)
        covered_shifts = 0
        
        for shift in shifts:
            assigned_count = len(solution.get_shift_employees(shift.id))
            if assigned_count >= shift.required_employees:
                covered_shifts += 1
        
        coverage_rate = covered_shifts / total_shifts if total_shifts > 0 else 0
        
        # Aumentar el fitness para soluciones con mejor cobertura
        coverage_bonus = coverage_rate * 0.5 * fitness_score
        
        # Fitness final incluye preferencias y cobertura
        final_fitness += coverage_bonus
        
        # Guardar el fitness en la solución para referencias futuras
        solution.fitness_score = max(0.1, final_fitness)
        
        return solution.fitness_score