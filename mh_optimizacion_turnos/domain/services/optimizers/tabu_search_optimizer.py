import random
import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Set
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
            "max_iterations": 100,            # Número de iteraciones de la búsqueda tabú
            "tabu_tenure": 15,                # Duración de permanencia en la lista tabú
            "neighborhood_size": 20,          # Número de vecinos a generar por iteración
            "max_iterations_without_improvement": 30, # Detención temprana
            "max_initial_solution_attempts": 1500, # Máximos intentos para generar solución inicial
            "max_neighbor_attempts": 100,     # Máximos intentos para generar vecinos válidos por iteración
            "validation_timeout": 15,         # Tiempo máximo (segundos) para generar soluciones válidas
            "use_constructive_approach": True, # Usar enfoque constructivo si aleatorio falla
            "relaxation_factor": 0.7,         # Factor para relajar restricciones temporalmente
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
        max_iterations = config.get("max_iterations", 100)
        tabu_tenure = config.get("tabu_tenure", 15)
        neighborhood_size = config.get("neighborhood_size", 20)
        max_no_improve = config.get("max_iterations_without_improvement", 30)
        max_initial_solution_attempts = config.get("max_initial_solution_attempts", 1500)
        max_neighbor_attempts = config.get("max_neighbor_attempts", 100)
        validation_timeout = config.get("validation_timeout", 15)
        use_constructive_approach = config.get("use_constructive_approach", True)
        relaxation_factor = config.get("relaxation_factor", 0.7)
        
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
        
        # Intentar generar una solución inicial válida usando diferentes estrategias
        current_solution = self._create_initial_solution(
            employees, shifts, validator, max_initial_solution_attempts, 
            validation_timeout, use_constructive_approach, relaxation_factor
        )
        
        if current_solution is None:
            raise ValueError(f"No se pudo generar una solución inicial válida después de varios intentos. "
                           f"Las restricciones pueden ser demasiado estrictas.")
        
        # Inicializar mejor solución
        best_solution = current_solution.clone()
        
        # Calcular fitness
        current_fitness = self._calculate_fitness(current_solution, employees, shifts)
        best_fitness = current_fitness
        metrics["objective_evaluations"] += 1
        
        iterations_without_improvement = 0
        
        logger.info(f"Solución inicial válida generada con fitness {current_fitness:.4f}")
        
        # Bucle principal de búsqueda tabú
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Generar vecindario válido (vecinos que cumplen con todas las restricciones)
            neighbors = self._generate_valid_neighborhood(
                current_solution, employees, shifts, validator,
                neighborhood_size, max_neighbor_attempts, validation_timeout
            )
            
            metrics["neighbor_attempts"] += max_neighbor_attempts
            metrics["neighbors_generated"] += len(neighbors)
            
            # Actualizar métrica de tasa de vecinos válidos
            current_valid_rate = len(neighbors) / max(1, max_neighbor_attempts)
            if metrics["valid_neighbors_rate"] == 0:
                metrics["valid_neighbors_rate"] = current_valid_rate
            else:
                metrics["valid_neighbors_rate"] = (
                    metrics["valid_neighbors_rate"] * 0.7 + current_valid_rate * 0.3
                )
            
            # Si no se encontraron vecinos válidos, intentar diversificación
            if not neighbors:
                logger.warning(f"No se encontraron movimientos válidos en la iteración {iteration}")
                # Diversificación: reiniciar desde una nueva solución
                diversified = self._create_diversification_solution(
                    employees, shifts, validator, max_initial_solution_attempts, relaxation_factor
                )
                if diversified:
                    current_solution = diversified
                    current_fitness = self._calculate_fitness(current_solution, employees, shifts)
                    metrics["objective_evaluations"] += 1
                    
                    # Actualizar mejor solución si es necesario
                    if current_fitness > best_fitness:
                        best_solution = current_solution.clone()
                        best_fitness = current_fitness
                        iterations_without_improvement = 0
                    else:
                        iterations_without_improvement += 1
                        
                    # Limpiar lista tabú
                    tabu_list.clear()
                    logger.info(f"Diversificación en iteración {iteration}: nuevo fitness = {current_fitness:.4f}")
                    continue
                else:
                    # Si no se puede crear una solución diversa, terminar
                    logger.warning("No se pudo diversificar. Finalizando búsqueda.")
                    break
                
            # Encontrar el mejor vecino no tabú
            best_neighbor = None
            best_neighbor_fitness = float('-inf')
            best_move = None
            
            for neighbor, move in neighbors:
                # Verificar si el movimiento es tabú
                is_tabu = move in tabu_list
                
                neighbor_fitness = self._calculate_fitness(neighbor, employees, shifts)
                metrics["objective_evaluations"] += 1
                
                # Criterio de aspiración: aceptar si es mejor que el mejor global, incluso si es tabú
                if neighbor_fitness > best_fitness and is_tabu:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
                    logger.info(f"Criterio de aspiración aplicado: movimiento tabú aceptado con fitness = {neighbor_fitness:.4f}")
                    break
                
                # Aceptar si no es tabú y es mejor que el mejor encontrado hasta ahora
                if not is_tabu and neighbor_fitness > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
            
            # Si no se encuentran movimientos no tabú
            if best_neighbor is None:
                logger.info(f"Todos los movimientos están en la lista tabú en la iteración {iteration}")
                # Intentar encontrar el mejor movimiento tabú como último recurso
                for neighbor, move in neighbors:
                    neighbor_fitness = self._calculate_fitness(neighbor, employees, shifts)
                    metrics["objective_evaluations"] += 1
                    
                    if neighbor_fitness > best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = neighbor_fitness
                        best_move = move
                        
                # Si aun así no hay vecinos, diversificar
                if best_neighbor is None:
                    continue
                
                logger.info(f"Seleccionando el mejor movimiento tabú con fitness = {best_neighbor_fitness:.4f}")
                
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
    
    def _create_initial_solution(self, 
                               employees: List[Employee], 
                               shifts: List[Shift],
                               validator: SolutionValidator,
                               max_attempts: int,
                               timeout: float,
                               use_constructive_approach: bool = True,
                               relaxation_factor: float = 0.7) -> Optional[Solution]:
        """Crear una solución inicial válida usando varias estrategias."""
        start_time = time.time()
        
        # Estrategia 1: Generar soluciones aleatorias que intenten respetar restricciones
        for attempt in range(max_attempts // 2):
            if time.time() - start_time > timeout / 2:
                break
                
            # Intentar crear una solución aleatoria que respete restricciones básicas
            solution = self._create_random_solution(employees, shifts)
            
            # Validar la solución
            validation_result = validator.validate(solution, employees, shifts)
            
            if validation_result.is_valid:
                logger.info(f"Solución inicial válida generada aleatoriamente en {attempt + 1} intentos")
                return solution
                
            if (attempt + 1) % 500 == 0:
                logger.info(f"Generando solución inicial: {attempt + 1} intentos realizados")
        
        # Estrategia 2: Enfoque constructivo
        if use_constructive_approach:
            logger.info("Intentando enfoque constructivo para generar solución inicial")
            constructive = self._create_constructive_solution(employees, shifts)
            
            validation_result = validator.validate(constructive, employees, shifts)
            
            if validation_result.is_valid:
                logger.info("Solución inicial válida generada usando enfoque constructivo")
                return constructive
        
        # Estrategia 3: Enfoque con relajación de restricciones
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time > 0:
            logger.info("Intentando generar solución con restricciones relajadas")
            attempt_limit = max(100, max_attempts // 4)
            
            for attempt in range(attempt_limit):
                if time.time() - start_time > timeout:
                    break
                    
                solution = self._create_relaxed_solution(employees, shifts, relaxation_factor)
                
                # Validar la solución
                validation_result = validator.validate(solution, employees, shifts)
                
                if validation_result.is_valid:
                    logger.info(f"Solución inicial válida generada con restricciones relajadas en {attempt + 1} intentos")
                    return solution
        
        logger.error("No se pudo generar una solución inicial válida")
        return None
    
    def _create_random_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
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
            
            # Si hay suficientes empleados calificados, tomar en cuenta preferencias y costo
            if len(qualified_employees) > 0:
                # Ordenar por preferencia (descendente) y luego por costo (ascendente)
                qualified_employees.sort(
                    key=lambda e: (
                        -e.get_preference_score(shift.day, shift.name),  # Preferencia alta primero
                        e.hourly_cost  # Costo bajo primero
                    )
                )
                
                # Asignar el número requerido de empleados o los disponibles
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
    
    def _create_constructive_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Crear una solución usando un enfoque constructivo greedy."""
        solution = Solution()
        
        # Crear mapeos para acceso rápido
        shift_dict = {s.id: s for s in shifts}
        employee_dict = {e.id: e for e in employees}
        
        # Rastrear asignaciones
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        day_shift_employees = defaultdict(set)
        
        # Ordenar turnos por prioridad (descendente)
        days_order = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO', 'DOMINGO']
        sorted_shifts = sorted(shifts, 
                              key=lambda s: (
                                  -s.priority,
                                  days_order.index(s.day.name) if s.day.name in days_order else 999,
                                  s.name.name  # ShiftType como MAÑANA, TARDE, NOCHE
                              ))
        
        # Asignar turnos uno por uno, priorizando los de mayor prioridad
        for shift in sorted_shifts:
            day_shift_key = (shift.day, shift.name)
            required = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if required <= 0:
                continue  # Este turno ya está cubierto
            
            # Encontrar candidatos para este turno basados en disponibilidad, habilidades y restricciones
            candidates = []
            for employee in employees:
                emp_id = employee.id
                
                # Verificar si ya está asignado a este turno
                if emp_id in day_shift_employees[day_shift_key]:
                    continue
                
                # Verificar disponibilidad y habilidades
                if not (employee.is_available(shift.day, shift.name) and 
                       shift.required_skills.issubset(employee.skills)):
                    continue
                
                # Verificar límite de horas
                if employee_hours[emp_id] + shift.duration_hours > employee.max_hours_per_week:
                    continue
                
                # Este empleado es un candidato
                preference = employee.get_preference_score(shift.day, shift.name)
                cost = employee.hourly_cost
                # Calculamos un score combinado: mayor preferencia y menor costo es mejor
                score = preference - (cost / 100.0)  # Normalizar el impacto del costo
                
                candidates.append((employee, score))
            
            # Ordenar candidatos por score (mayor primero)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Asignar los mejores candidatos
            for i in range(min(required, len(candidates))):
                employee = candidates[i][0]
                
                # Crear asignación
                assignment = Assignment(
                    employee_id=employee.id,
                    shift_id=shift.id,
                    cost=employee.hourly_cost * shift.duration_hours
                )
                solution.add_assignment(assignment)
                
                # Actualizar registros
                employee_hours[employee.id] += shift.duration_hours
                employee_days[employee.id].add(shift.day)
                day_shift_employees[day_shift_key].add(employee.id)
        
        return solution
    
    def _create_relaxed_solution(self, 
                              employees: List[Employee],
                              shifts: List[Shift],
                              relaxation_factor: float) -> Solution:
        """Crear una solución relajando algunas restricciones."""
        solution = Solution()
        
        # Rastrear asignaciones
        employee_hours = {e.id: 0.0 for e in employees}
        day_shift_employees = defaultdict(set)
        
        for shift in shifts:
            day_shift_key = (shift.day, shift.name)
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue  # Este turno ya está cubierto
            
            # Construir lista de candidatos con restricciones relajadas
            candidates = []
            
            for employee in employees:
                emp_id = employee.id
                
                # Verificar si ya está asignado a este turno
                if emp_id in day_shift_employees[day_shift_key]:
                    continue
                
                # Verificar disponibilidad con relajación
                available = (employee.is_available(shift.day, shift.name) or 
                           random.random() < relaxation_factor)
                
                # Verificar habilidades con relajación
                has_skills = (shift.required_skills.issubset(employee.skills) or 
                             random.random() < relaxation_factor)
                
                # Relajación de horas: permitir exceder un poco el máximo
                hours_ok = (employee_hours[emp_id] + shift.duration_hours <= 
                         employee.max_hours_per_week * (1 + relaxation_factor * 0.1))
                
                if available and has_skills and hours_ok:
                    # Este empleado es candidato
                    preference = employee.get_preference_score(shift.day, shift.name)
                    candidates.append((employee, preference))
            
            # Si aún así no hay suficientes candidatos, usar a cualquiera
            if len(candidates) < needed:
                all_employees = [e for e in employees if e.id not in day_shift_employees[day_shift_key]]
                if all_employees:
                    # Añadir empleados aleatorios
                    random.shuffle(all_employees)
                    for e in all_employees:
                        if (e.id, e.get_preference_score(shift.day, shift.name)) not in candidates:
                            candidates.append((e, 0))  # Preferencia neutral
                        if len(candidates) >= needed:
                            break
            
            # Ordenar por preferencia (mayor primero)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Asignar empleados
            for i in range(min(needed, len(candidates))):
                employee = candidates[i][0]
                
                # Crear asignación
                assignment = Assignment(
                    employee_id=employee.id,
                    shift_id=shift.id,
                    cost=employee.hourly_cost * shift.duration_hours
                )
                solution.add_assignment(assignment)
                
                # Actualizar registros
                employee_hours[employee.id] += shift.duration_hours
                day_shift_employees[day_shift_key].add(employee.id)
        
        return solution
    
    def _create_diversification_solution(self, 
                                    employees: List[Employee], 
                                    shifts: List[Shift], 
                                    validator: SolutionValidator, 
                                    max_attempts: int,
                                    relaxation_factor: float = 0.7) -> Optional[Solution]:
        """Crear una solución diversificada para escapar de óptimos locales."""
        # Estrategia 1: Enfoque constructivo con aleatorización
        constructive = self._create_constructive_solution(employees, shifts)
        validation_result = validator.validate(constructive, employees, shifts)
        
        if validation_result.is_valid:
            # Aplicar una mutación fuerte para diversificar
            self._mutate_solution(constructive, employees, shifts, mutation_rate=0.3)
            
            # Verificar que sigue siendo válida
            validation_result = validator.validate(constructive, employees, shifts)
            if validation_result.is_valid:
                return constructive
        
        # Estrategia 2: Generar soluciones relajadas y validarlas
        for attempt in range(max_attempts // 10):
            solution = self._create_relaxed_solution(employees, shifts, relaxation_factor)
            
            validation_result = validator.validate(solution, employees, shifts)
            if validation_result.is_valid:
                return solution
        
        # No se pudo generar una solución válida
        return None
    
    def _generate_valid_neighborhood(self,
                               current_solution: Solution,
                               employees: List[Employee],
                               shifts: List[Shift],
                               validator: SolutionValidator,
                               neighborhood_size: int,
                               max_attempts: int,
                               timeout: float) -> List[Tuple[Solution, Tuple]]:
        """Genera un conjunto de soluciones vecinas válidas."""
        valid_neighbors = []
        attempts = 0
        start_time = time.time()
        
        while len(valid_neighbors) < neighborhood_size and attempts < max_attempts:
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout en la generación de vecindario después de {attempts} intentos")
                break
                
            # Generar un vecino candidato
            candidate, move = self._generate_neighbor(current_solution, employees, shifts)
            attempts += 1
            
            # Validar la solución
            validation_result = validator.validate(candidate, employees, shifts)
            
            if validation_result.is_valid:
                valid_neighbors.append((candidate, move))
                
            # Informar progreso
            if attempts % 50 == 0 and attempts > 0:
                logger.debug(f"Vecinos generados: {len(valid_neighbors)}/{neighborhood_size} "
                           f"en {attempts} intentos ({time.time() - start_time:.2f}s)")
        
        return valid_neighbors
    
    def _generate_neighbor(self, 
                         solution: Solution, 
                         employees: List[Employee], 
                         shifts: List[Shift]) -> Tuple[Solution, Tuple]:
        """Genera un vecino usando diversos operadores de vecindad."""
        # Operadores de vecindad disponibles
        operators = [
            self._swap_operator,
            self._replace_operator,
            self._add_assignment_operator
        ]
        
        # Seleccionar un operador aleatoriamente
        operator = random.choice(operators)
        
        # Aplicar el operador
        return operator(solution, employees, shifts)
    
    def _swap_operator(self, 
                     solution: Solution, 
                     employees: List[Employee], 
                     shifts: List[Shift]) -> Tuple[Solution, Tuple]:
        """Operador de vecindad: intercambiar dos asignaciones entre empleados."""
        if len(solution.assignments) < 2:
            # No hay suficientes asignaciones para intercambiar, usar otro operador
            return self._replace_operator(solution, employees, shifts)
        
        # Clonar solución para modificarla
        new_solution = solution.clone()
        
        # Seleccionar dos asignaciones aleatorias
        indices = random.sample(range(len(new_solution.assignments)), 2)
        a1 = new_solution.assignments[indices[0]]
        a2 = new_solution.assignments[indices[1]]
        
        # Intercambiar los empleados
        emp1, emp2 = a1.employee_id, a2.employee_id
        
        # Crear nuevas asignaciones con los empleados intercambiados
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        if emp1 in employee_dict and emp2 in employee_dict and a1.shift_id in shift_dict and a2.shift_id in shift_dict:
            e1 = employee_dict[emp1]
            e2 = employee_dict[emp2]
            s1 = shift_dict[a1.shift_id]
            s2 = shift_dict[a2.shift_id]
            
            new_a1 = Assignment(
                employee_id=emp2,
                shift_id=a1.shift_id,
                cost=e2.hourly_cost * s1.duration_hours
            )
            
            new_a2 = Assignment(
                employee_id=emp1,
                shift_id=a2.shift_id,
                cost=e1.hourly_cost * s2.duration_hours
            )
            
            # Reemplazar las asignaciones originales
            new_solution.assignments[indices[0]] = new_a1
            new_solution.assignments[indices[1]] = new_a2
        
        # Identificador único para este movimiento
        move = ("swap", (emp1, a1.shift_id), (emp2, a2.shift_id))
        
        return new_solution, move
    
    def _replace_operator(self, 
                        solution: Solution, 
                        employees: List[Employee], 
                        shifts: List[Shift]) -> Tuple[Solution, Tuple]:
        """Operador de vecindad: reemplazar un empleado en una asignación."""
        if not solution.assignments:
            # No hay asignaciones para reemplazar, usar otro operador
            return self._add_assignment_operator(solution, employees, shifts)
        
        # Clonar solución para modificarla
        new_solution = solution.clone()
        
        # Seleccionar una asignación aleatoria para modificar
        index = random.randrange(len(new_solution.assignments))
        assignment = new_solution.assignments[index]
        
        # Obtener información del turno
        shift_dict = {s.id: s for s in shifts}
        if assignment.shift_id not in shift_dict:
            # Turno no encontrado, intentar con otro operador
            return self._add_assignment_operator(solution, employees, shifts)
            
        shift = shift_dict[assignment.shift_id]
        
        # Encontrar empleados que podrían reemplazar al actual
        current_employee_id = assignment.employee_id
        
        # Obtener todos los empleados asignados a este turno
        assigned_employees = set(
            a.employee_id for a in solution.assignments 
            if a.shift_id == assignment.shift_id
        )
        
        # Buscar candidatos para reemplazo
        candidates = []
        for employee in employees:
            if (employee.id != current_employee_id and 
                employee.id not in assigned_employees and
                employee.is_available(shift.day, shift.name) and
                shift.required_skills.issubset(employee.skills)):
                candidates.append(employee)
        
        if not candidates:
            # No hay candidatos para reemplazo, usar otro operador
            return self._add_assignment_operator(solution, employees, shifts)
        
        # Seleccionar un empleado aleatorio como reemplazo
        new_employee = random.choice(candidates)
        
        # Crear nueva asignación
        new_assignment = Assignment(
            employee_id=new_employee.id,
            shift_id=assignment.shift_id,
            cost=new_employee.hourly_cost * shift.duration_hours
        )
        
        # Reemplazar la asignación original
        new_solution.assignments[index] = new_assignment
        
        # Identificador único para este movimiento
        move = ("replace", current_employee_id, new_employee.id, assignment.shift_id)
        
        return new_solution, move
    
    def _add_assignment_operator(self, 
                              solution: Solution, 
                              employees: List[Employee], 
                              shifts: List[Shift]) -> Tuple[Solution, Tuple]:
        """Operador de vecindad: añadir una nueva asignación."""
        # Clonar solución para modificarla
        new_solution = solution.clone()
        
        # Encontrar turnos con cobertura insuficiente
        shift_dict = {s.id: s for s in shifts}
        undercover_shifts = []
        
        for shift in shifts:
            assigned = len(solution.get_shift_employees(shift.id))
            if assigned < shift.required_employees:
                undercover_shifts.append((shift, shift.required_employees - assigned))
        
        # Si todos los turnos están cubiertos, intentar otro operador
        if not undercover_shifts:
            # No hay turnos para cubrir, modificar una asignación existente
            if solution.assignments:
                return self._replace_operator(solution, employees, shifts)
            else:
                # Crear una asignación completamente nueva
                if shifts and employees:
                    shift = random.choice(shifts)
                    employee = random.choice(employees)
                    
                    new_assignment = Assignment(
                        employee_id=employee.id,
                        shift_id=shift.id,
                        cost=employee.hourly_cost * shift.duration_hours
                    )
                    
                    new_solution.add_assignment(new_assignment)
                    
                    # Identificador único para este movimiento
                    move = ("add_new", employee.id, shift.id)
                    
                    return new_solution, move
        
        # Seleccionar un turno con baja cobertura
        selected_shift, needed = random.choice(undercover_shifts)
        
        # Obtener empleados ya asignados a este turno
        current_assigned_ids = set(solution.get_shift_employees(selected_shift.id))
        
        # Buscar empleados disponibles que no estén ya asignados
        available_employees = [
            e for e in employees 
            if e.id not in current_assigned_ids and 
            e.is_available(selected_shift.day, selected_shift.name) and
            selected_shift.required_skills.issubset(e.skills)
        ]
        
        # Si no hay empleados disponibles, intentar otro operador
        if not available_employees:
            return self._replace_operator(solution, employees, shifts)
        
        # Seleccionar un empleado aleatorio
        selected_employee = random.choice(available_employees)
        
        # Añadir nueva asignación
        new_assignment = Assignment(
            employee_id=selected_employee.id,
            shift_id=selected_shift.id,
            cost=selected_employee.hourly_cost * selected_shift.duration_hours
        )
        
        new_solution.add_assignment(new_assignment)
        
        # Identificador único para este movimiento
        move = ("add", selected_employee.id, selected_shift.id)
        
        return new_solution, move
    
    def _mutate_solution(self, 
                       solution: Solution, 
                       employees: List[Employee], 
                       shifts: List[Shift], 
                       mutation_rate: float = 0.2) -> None:
        """Aplica mutación a una solución (modificación in-place)."""
        # Rastrear empleados por turno
        shift_employees = defaultdict(set)
        for assignment in solution.assignments:
            shift_employees[assignment.shift_id].add(assignment.employee_id)
        
        # Decidir qué asignaciones modificar
        for i, assignment in enumerate(solution.assignments[:]):  # Copiar lista para iterar seguramente
            if random.random() < mutation_rate:
                # Eliminar esta asignación
                solution.assignments.remove(assignment)
                shift_employees[assignment.shift_id].remove(assignment.employee_id)
                
                # Posiblemente reemplazar con otra
                if random.random() < 0.7:  # 70% de probabilidad de reemplazo
                    shift_id = assignment.shift_id
                    if shift_id in shift_employees:
                        # Buscar un nuevo empleado
                        shift = next((s for s in shifts if s.id == shift_id), None)
                        if shift:
                            qualified_employees = [
                                e for e in employees 
                                if e.id not in shift_employees[shift_id] and 
                                e.is_available(shift.day, shift.name) and
                                shift.required_skills.issubset(e.skills)
                            ]
                            
                            if qualified_employees:
                                new_employee = random.choice(qualified_employees)
                                new_assignment = Assignment(
                                    employee_id=new_employee.id,
                                    shift_id=shift_id,
                                    cost=new_employee.hourly_cost * shift.duration_hours
                                )
                                solution.add_assignment(new_assignment)
                                shift_employees[shift_id].add(new_employee.id)
        
        # Añadir algunas asignaciones completamente nuevas
        for shift in shifts:
            if random.random() < mutation_rate:
                # Verificar si hay espacio para más empleados
                assigned = len(shift_employees.get(shift.id, set()))
                if assigned < shift.required_employees:
                    # Buscar empleados disponibles
                    available = [
                        e for e in employees 
                        if e.id not in shift_employees.get(shift.id, set()) and
                        e.is_available(shift.day, shift.name) and
                        shift.required_skills.issubset(e.skills)
                    ]
                    
                    if available:
                        selected = random.choice(available)
                        new_assignment = Assignment(
                            employee_id=selected.id,
                            shift_id=shift.id,
                            cost=selected.hourly_cost * shift.duration_hours
                        )
                        solution.add_assignment(new_assignment)
                        shift_employees[shift.id].add(selected.id)
    
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