import random
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import deque, defaultdict

from mh_optimizacion_turnos.domain.services.optimizer_strategy import OptimizerStrategy
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.models.assignment import Assignment


logger = logging.getLogger(__name__)


class TabuSearchOptimizer(OptimizerStrategy):
    """Implementación de Búsqueda Tabú para la optimización de turnos con restricciones duras."""
    
    def get_name(self) -> str:
        return "Tabu Search Optimizer"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "max_iterations": 1000,
            "tabu_tenure": 20,
            "neighborhood_size": 30,
            "max_iterations_without_improvement": 200,
            "max_initial_solution_attempts": 100,  # Máximos intentos para generar solución inicial válida
            "max_neighbor_attempts": 100,  # Máximos intentos para generar vecinos válidos por iteración
            "validation_timeout": 10,  # Tiempo máximo (segundos) para generar soluciones válidas
            "metrics": {
                "enabled": True,
                "track_evaluations": True,
                "track_rejection_rate": True
            }
        }
    
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimiza la asignación de turnos usando Búsqueda Tabú con restricciones duras."""
        if config is None:
            config = self.get_default_config()
        
        # Extraer parámetros de configuración
        max_iterations = config.get("max_iterations", 1000)
        tabu_tenure = config.get("tabu_tenure", 20)
        neighborhood_size = config.get("neighborhood_size", 30)
        max_no_improve = config.get("max_iterations_without_improvement", 200)
        max_initial_solution_attempts = config.get("max_initial_solution_attempts", 100)
        max_neighbor_attempts = config.get("max_neighbor_attempts", 100)
        validation_timeout = config.get("validation_timeout", 10)
        
        # Inicialización de métricas
        metrics = {
            "objective_evaluations": 0,
            "valid_neighbors_rate": 0,
            "initial_solution_attempts": 0,
            "neighbor_attempts": 0,
            "neighbors_generated": 0
        }
        
        # Crear validador
        validator = SolutionValidator()
        
        logger.info(f"Iniciando optimización con búsqueda tabú con máximo {max_iterations} iteraciones")
        logger.info(f"Usando enfoque de restricciones duras (solo soluciones factibles)")
        
        # Inicializar lista tabú como una cola de tamaño fijo
        tabu_list = deque(maxlen=tabu_tenure)
        
        # Crear solución inicial válida
        start_time = time.time()
        current_solution = None
        
        for attempt in range(max_initial_solution_attempts):
            metrics["initial_solution_attempts"] += 1
            
            # Intentar generar una solución inicial
            candidate = self._create_initial_solution(employees, shifts)
            
            # Validar la solución
            validation_result = validator.validate(candidate, employees, shifts)
            
            if validation_result.is_valid:
                current_solution = candidate
                break
                
            # Verificar timeout
            if time.time() - start_time > validation_timeout:
                logger.warning("Timeout en la generación de solución inicial válida")
                break
        
        # Si no se pudo generar una solución inicial válida
        if current_solution is None:
            raise ValueError(f"No se pudo generar una solución inicial válida después de "
                           f"{metrics['initial_solution_attempts']} intentos. "
                           f"Las restricciones pueden ser demasiado estrictas.")
        
        # Inicializar mejor solución
        best_solution = current_solution.clone()
        
        # Calcular fitness
        current_fitness = self._calculate_fitness(current_solution, employees, shifts)
        best_fitness = current_fitness
        metrics["objective_evaluations"] += 1
        
        iterations_without_improvement = 0
        
        logger.info(f"Solución inicial válida generada con fitness {current_fitness:.4f} "
                  f"después de {metrics['initial_solution_attempts']} intentos")
        
        # Bucle principal de búsqueda tabú
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Generar vecindario válido (vecinos que cumplen con todas las restricciones)
            neighbors = []
            neighbor_attempts = 0
            
            while len(neighbors) < neighborhood_size and neighbor_attempts < max_neighbor_attempts:
                # Generar un conjunto de vecinos candidatos
                candidate_neighbors = self._generate_neighborhood_candidates(
                    current_solution, employees, shifts, 
                    min(neighborhood_size, max_neighbor_attempts - neighbor_attempts)
                )
                
                neighbor_attempts += len(candidate_neighbors)
                metrics["neighbor_attempts"] += len(candidate_neighbors)
                
                # Validar cada vecino
                for neighbor, move in candidate_neighbors:
                    validation_result = validator.validate(neighbor, employees, shifts)
                    
                    if validation_result.is_valid:
                        neighbors.append((neighbor, move))
                        metrics["neighbors_generated"] += 1
                
                # Verificar timeout
                if time.time() - iteration_start > validation_timeout:
                    logger.warning(f"Timeout en la generación de vecindario en iteración {iteration}")
                    break
            
            # Actualizar métrica de tasa de vecinos válidos
            if neighbor_attempts > 0:
                current_valid_rate = len(neighbors) / neighbor_attempts
                if metrics["valid_neighbors_rate"] == 0:
                    metrics["valid_neighbors_rate"] = current_valid_rate
                else:
                    metrics["valid_neighbors_rate"] = (
                        metrics["valid_neighbors_rate"] * 0.7 + current_valid_rate * 0.3
                    )
            
            # Si no se encontraron vecinos válidos, finalizar
            if not neighbors:
                logger.warning(f"No se encontraron movimientos válidos en la iteración {iteration} "
                             f"después de {neighbor_attempts} intentos")
                break
                
            # Encontrar el mejor vecino no tabú
            best_neighbor = None
            best_neighbor_fitness = float('-inf')
            best_move = None
            
            for neighbor, move in neighbors:
                # Verificar si el movimiento es tabú
                if move in tabu_list:
                    continue
                    
                neighbor_fitness = self._calculate_fitness(neighbor, employees, shifts)
                metrics["objective_evaluations"] += 1
                
                # Aceptar si es mejor que el mejor encontrado hasta ahora
                if neighbor_fitness > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
                
                # Criterio de aspiración: aceptar si es mejor que el mejor global, incluso si es tabú
                if neighbor_fitness > best_fitness and move in tabu_list:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
                    break
            
            # Si no se encuentran movimientos no tabú
            if best_neighbor is None:
                logger.info(f"Todos los movimientos están en la lista tabú en la iteración {iteration}")
                # Diversificación: reiniciar desde una nueva solución
                new_solution = self._create_diversification_solution(
                    employees, shifts, validator, max_initial_solution_attempts
                )
                if new_solution:
                    current_solution = new_solution
                    current_fitness = self._calculate_fitness(current_solution, employees, shifts)
                    metrics["objective_evaluations"] += 1
                    
                    # Limpiar lista tabú
                    tabu_list.clear()
                    continue
                else:
                    # Si no se puede crear una solución diversa, terminar
                    break
                
            # Moverse al mejor vecino
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            
            # Añadir el movimiento a la lista tabú
            tabu_list.append(best_move)
            
            # Actualizar mejor solución si ha mejorado
            if current_fitness > best_fitness:
                best_solution = current_solution.clone()
                best_fitness = current_fitness
                iterations_without_improvement = 0
                logger.info(f"Iteración {iteration}: Nueva mejor solución con fitness {best_fitness:.4f}")
            else:
                iterations_without_improvement += 1
            
            # Registrar progreso periódicamente
            if iteration % 10 == 0:
                logger.info(f"Iteración {iteration}: Fitness actual = {current_fitness:.4f}, "
                          f"Mejor fitness = {best_fitness:.4f}, "
                          f"Tasa de vecinos válidos = {metrics['valid_neighbors_rate']:.2f}")
            
            # Detener si no hay mejora durante un tiempo
            if iterations_without_improvement >= max_no_improve:
                logger.info(f"Detención temprana - no hay mejora durante {max_no_improve} iteraciones")
                break
        
        logger.info(f"Búsqueda tabú completada. Mejor fitness: {best_fitness:.4f}")
        logger.info(f"Métricas: {metrics['objective_evaluations']} evaluaciones de función objetivo, "
                  f"Tasa de vecinos válidos: {metrics['valid_neighbors_rate']:.2f}")
        
        return best_solution
    
    def _create_initial_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Crear una solución inicial que intenta respetar restricciones."""
        solution = Solution()
        
        # Ordenar turnos por prioridad (descendente)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        # Rastrear qué empleados ya han sido asignados a cada tipo de turno en cada día
        day_shift_employees = defaultdict(set)
        
        # Rastrear horas y días asignados por empleado
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
        for shift in sorted_shifts:
            # Determinar cuántos empleados se necesitan asignar para este turno
            day_shift_key = (shift.day, shift.name)
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue  # Este turno ya tiene los empleados requeridos
                
            # Encontrar empleados calificados para este turno
            qualified_employees = []
            
            for employee in employees:
                # Verificar si ya está asignado a este turno en este día
                if employee.id in day_shift_employees[day_shift_key]:
                    continue
                    
                # Verificar disponibilidad y habilidades
                if not (employee.is_available(shift.day, shift.name) and 
                      shift.required_skills.issubset(employee.skills)):
                    continue
                
                # Verificar horas máximas
                if employee_hours[employee.id] + shift.duration_hours > employee.max_hours_per_week:
                    continue
                
                # Verificar días consecutivos (simplificado)
                if shift.day in employee_days[employee.id]:
                    # Ya tiene un turno ese día, no es problema
                    pass
                elif len(employee_days[employee.id]) >= employee.max_consecutive_days:
                    # Simplificación: permitimos siempre que no exceda una semana
                    if len(employee_days[employee.id]) < 7:
                        pass
                    else:
                        continue
                
                # Este empleado cumple con todas las condiciones
                qualified_employees.append(employee)
            
            # Ordenar por costo por hora (ascendente)
            qualified_employees.sort(key=lambda e: e.hourly_cost)
            
            # Asignar el número requerido de empleados
            for i in range(min(needed, len(qualified_employees))):
                employee = qualified_employees[i]
                assignment = Assignment(
                    employee_id=employee.id,
                    shift_id=shift.id,
                    cost=employee.hourly_cost * shift.duration_hours
                )
                solution.add_assignment(assignment)
                
                # Actualizar registros
                day_shift_employees[day_shift_key].add(employee.id)
                employee_hours[employee.id] += shift.duration_hours
                employee_days[employee.id].add(shift.day)
        
        return solution
    
    def _create_diversification_solution(self, employees: List[Employee], shifts: List[Shift], 
                                        validator: SolutionValidator, max_attempts: int) -> Optional[Solution]:
        """Crear una solución diversificada para escapar de óptimos locales."""
        for attempt in range(max_attempts):
            # Crear solución con perturbación aleatoria en la selección de empleados
            solution = Solution()
            
            # Ordenar turnos aleatoriamente (para diversificar)
            random_shifts = list(shifts)
            random.shuffle(random_shifts)
            
            # Rastrear asignaciones por día y tipo de turno
            day_shift_employees = defaultdict(set)
            employee_hours = {e.id: 0.0 for e in employees}
            employee_days = {e.id: set() for e in employees}
            
            for shift in random_shifts:
                day_shift_key = (shift.day, shift.name)
                needed = shift.required_employees - len(day_shift_employees[day_shift_key])
                
                if needed <= 0:
                    continue
                    
                # Encontrar empleados calificados
                qualified_employees = [
                    e for e in employees
                    if e.id not in day_shift_employees[day_shift_key] and
                    e.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(e.skills) and
                    employee_hours[e.id] + shift.duration_hours <= e.max_hours_per_week
                ]
                
                # Mezclar aleatoriamente (para diversificar)
                random.shuffle(qualified_employees)
                
                # Asignar empleados
                for i in range(min(needed, len(qualified_employees))):
                    employee = qualified_employees[i]
                    assignment = Assignment(
                        employee_id=employee.id,
                        shift_id=shift.id,
                        cost=employee.hourly_cost * shift.duration_hours
                    )
                    solution.add_assignment(assignment)
                    
                    # Actualizar registros
                    day_shift_employees[day_shift_key].add(employee.id)
                    employee_hours[employee.id] += shift.duration_hours
                    employee_days[employee.id].add(shift.day)
            
            # Validar solución
            validation_result = validator.validate(solution, employees, shifts)
            
            if validation_result.is_valid:
                return solution
        
        return None
    
    def _generate_neighborhood_candidates(self, current_solution: Solution,
                                        employees: List[Employee], shifts: List[Shift],
                                        neighborhood_size: int) -> List[Tuple[Solution, Tuple]]:
        """Generar candidatos a soluciones vecinas (pueden incluir soluciones inválidas)."""
        candidates = []
        
        # Crear diccionarios para búsquedas rápidas
        shift_dict = {s.id: s for s in shifts}
        employee_dict = {e.id: e for e in employees}
        
        # Agrupar asignaciones por turno
        shift_assignments = defaultdict(list)
        for a in current_solution.assignments:
            shift_assignments[a.shift_id].append(a)
        
        # Rastrear asignaciones actuales por día y tipo de turno
        day_shift_employees = defaultdict(set)
        for assignment in current_solution.assignments:
            if assignment.shift_id in shift_dict:
                shift = shift_dict[assignment.shift_id]
                day_shift_key = (shift.day, shift.name)
                day_shift_employees[day_shift_key].add(assignment.employee_id)
        
        # Estrategia 1: Reemplazar Empleado (con restricciones)
        if current_solution.assignments:
            for _ in range(min(neighborhood_size // 2, len(current_solution.assignments))):
                # Elegir asignación aleatoria para reemplazar
                assignment = random.choice(current_solution.assignments)
                shift = shift_dict.get(assignment.shift_id)
                
                if shift:
                    # Encontrar posibles reemplazos que respeten restricciones
                    day_shift_key = (shift.day, shift.name)
                    
                    replacement_candidates = [
                        e for e in employees
                        if e.id != assignment.employee_id and
                        e.id not in day_shift_employees[day_shift_key] and
                        e.is_available(shift.day, shift.name) and
                        shift.required_skills.issubset(e.skills)
                    ]
                    
                    if replacement_candidates:
                        new_employee = random.choice(replacement_candidates)
                        
                        # Crear nueva solución con este reemplazo
                        new_solution = current_solution.clone()
                        
                        # Encontrar y reemplazar la asignación
                        for j, a in enumerate(new_solution.assignments):
                            if a.employee_id == assignment.employee_id and a.shift_id == assignment.shift_id:
                                new_solution.assignments[j] = Assignment(
                                    employee_id=new_employee.id,
                                    shift_id=assignment.shift_id,
                                    cost=new_employee.hourly_cost * shift.duration_hours
                                )
                                break
                        
                        # Identificador único para este movimiento
                        move = (assignment.employee_id, assignment.shift_id, new_employee.id)
                        candidates.append((new_solution, move))
        
        # Estrategia 2: Intercambiar Asignaciones (con restricciones)
        if len(shift_assignments) >= 2:
            shift_ids = list(shift_assignments.keys())
            
            for _ in range(min(neighborhood_size // 2, len(shift_assignments))):
                if len(shift_ids) < 2:
                    break
                    
                # Elegir dos turnos diferentes
                shift_id_pair = random.sample(shift_ids, 2)
                
                if (shift_assignments[shift_id_pair[0]] and shift_assignments[shift_id_pair[1]]):
                    # Elegir una asignación de cada turno
                    assignment1 = random.choice(shift_assignments[shift_id_pair[0]])
                    assignment2 = random.choice(shift_assignments[shift_id_pair[1]])
                    
                    shift1 = shift_dict.get(assignment1.shift_id)
                    shift2 = shift_dict.get(assignment2.shift_id)
                    
                    employee1 = employee_dict.get(assignment1.employee_id)
                    employee2 = employee_dict.get(assignment2.employee_id)
                    
                    # Verificar compatibilidad básica (será validado después)
                    if (employee1 and employee2 and shift1 and shift2):
                        # Crear nueva solución con el intercambio
                        new_solution = current_solution.clone()
                        
                        # Realizar el intercambio
                        new_assignments = []
                        for a in new_solution.assignments:
                            if a.employee_id == assignment1.employee_id and a.shift_id == assignment1.shift_id:
                                new_assignments.append(Assignment(
                                    employee_id=employee2.id,
                                    shift_id=shift1.id,
                                    cost=employee2.hourly_cost * shift1.duration_hours
                                ))
                            elif a.employee_id == assignment2.employee_id and a.shift_id == assignment2.shift_id:
                                new_assignments.append(Assignment(
                                    employee_id=employee1.id,
                                    shift_id=shift2.id,
                                    cost=employee1.hourly_cost * shift2.duration_hours
                                ))
                            else:
                                new_assignments.append(a)
                        
                        new_solution.assignments = new_assignments
                        
                        # Identificador único para este intercambio
                        move = ((assignment1.employee_id, assignment1.shift_id),
                              (assignment2.employee_id, assignment2.shift_id))
                        
                        candidates.append((new_solution, move))
        
        # Si tenemos pocos candidatos, añadir más operaciones
        if len(candidates) < neighborhood_size // 2:
            # Estrategia 3: Añadir asignaciones faltantes
            missing_assignments = []
            
            # Identificar turnos con asignaciones insuficientes
            for shift in shifts:
                day_shift_key = (shift.day, shift.name)
                assigned = len(day_shift_employees[day_shift_key])
                missing = max(0, shift.required_employees - assigned)
                
                if missing > 0:
                    missing_assignments.append((shift, missing))
            
            # Intentar añadir asignaciones faltantes
            if missing_assignments:
                for _ in range(min(neighborhood_size // 4, len(missing_assignments))):
                    shift, needed = random.choice(missing_assignments)
                    day_shift_key = (shift.day, shift.name)
                    
                    # Encontrar empleados que podrían asignarse a este turno
                    available_employees = [
                        e for e in employees
                        if e.id not in day_shift_employees[day_shift_key] and
                        e.is_available(shift.day, shift.name) and
                        shift.required_skills.issubset(e.skills)
                    ]
                    
                    if available_employees:
                        new_employee = random.choice(available_employees)
                        
                        # Crear nueva solución añadiendo esta asignación
                        new_solution = current_solution.clone()
                        new_assignment = Assignment(
                            employee_id=new_employee.id,
                            shift_id=shift.id,
                            cost=new_employee.hourly_cost * shift.duration_hours
                        )
                        new_solution.add_assignment(new_assignment)
                        
                        # Identificador único para este movimiento
                        move = ("add", shift.id, new_employee.id)
                        candidates.append((new_solution, move))
        
        return candidates
    
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
        final_fitness = fitness_score + preference_bonus + coverage_bonus
        
        # Guardar el fitness en la solución para referencias futuras
        solution.fitness_score = max(0.1, final_fitness)
        
        return solution.fitness_score