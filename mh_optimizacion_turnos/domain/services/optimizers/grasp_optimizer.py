import random
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
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
            "alpha": 0.5,  # Factor de aleatorización para construcción de soluciones
            "local_search_iterations": 50,
            "max_construction_attempts": 500, 
            "max_initial_solution_attempts": 1000,
            "validation_timeout": 30,
            "construction_strategies": ["greedy", "constructive", "hybrid", "minimal"],
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
        alpha = config.get("alpha", 0.5)
        local_search_iterations = config.get("local_search_iterations", 50)
        max_construction_attempts = config.get("max_construction_attempts", 500)
        max_initial_solution_attempts = config.get("max_initial_solution_attempts", 1000)
        validation_timeout = config.get("validation_timeout", 30)
        construction_strategies = config.get("construction_strategies", 
                                           ["greedy", "constructive", "hybrid", "minimal"])
        
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
        
        # Intentar generar solución inicial
        best_solution = self._create_initial_solution(
            employees, shifts, validator, 
            max_initial_solution_attempts, validation_timeout,
            alpha, construction_strategies
        )
        
        # Si aún no hay solución, lanzar error
        if best_solution is None:
            raise ValueError("No se pudo encontrar ninguna solución válida. Las restricciones pueden ser demasiado estrictas.")
        
        # Calcular fitness inicial
        best_fitness = self._calculate_fitness(best_solution, employees, shifts)
        logger.info(f"Solución inicial válida encontrada con fitness {best_fitness:.4f}")
        
        # Ciclo principal de GRASP
        start_time = time.time()
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Fase 1: Construcción - Construcción greedy aleatorizada de una solución VÁLIDA
            solution = None
            construction_attempts = 0
            
            # Ciclo adaptativo: intentar primero con la configuración normal,
            # luego con configuración más relajada si hay problemas
            for current_alpha in [alpha, alpha + 0.2, alpha + 0.4]:
                while solution is None and construction_attempts < max_construction_attempts // 3:
                    construction_attempts += 1
                    metrics["construction_attempts"] += 1
                    
                    # Intentar construir una solución candidata con parámetros actuales
                    candidate = self._construct_solution(
                        employees, shifts, current_alpha, "greedy"
                    )
                    
                    # Validar solución
                    validation_result = validator.validate(candidate, employees, shifts)
                    
                    if validation_result.is_valid:
                        solution = candidate
                        metrics["valid_solutions_found"] += 1
                        break
                        
                    # Verificar timeout
                    if time.time() - iteration_start > validation_timeout / 3:
                        break
                
                if solution is not None:
                    break
            
            # Si no se encontró solución en el ciclo adaptativo, usar la mejor encontrada hasta ahora
            if solution is None and best_solution is not None:
                logger.warning(f"Usando mejor solución conocida para iteración {iteration}")
                solution = best_solution.clone()
            
            # Actualizar métrica de tasa de éxito
            current_success_rate = metrics["valid_solutions_found"] / max(1, metrics["construction_attempts"])
            if metrics["construction_success_rate"] == 0:
                metrics["construction_success_rate"] = current_success_rate
            else:
                metrics["construction_success_rate"] = (
                    metrics["construction_success_rate"] * 0.7 + current_success_rate * 0.3
                )
            
            # Si no se pudo construir una solución válida, pasar a la siguiente iteración
            if solution is None:
                logger.warning(f"No se pudo construir una solución válida en la iteración {iteration}")
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
                          f"Mejor fitness: {best_fitness:.4f}")
        
        # Registrar métricas finales
        logger.info(f"Optimización GRASP completada. Fitness de la mejor solución: {best_fitness:.4f}")
        
        return best_solution
    
    def _create_initial_solution(self,
                              employees: List[Employee],
                              shifts: List[Shift],
                              validator: SolutionValidator,
                              max_attempts: int,
                              timeout: float,
                              alpha: float,
                              strategies: List[str]) -> Optional[Solution]:
        """Crear una solución inicial válida usando múltiples estrategias."""
        logger.info("Generando solución inicial para GRASP...")
        start_time = time.time()
        
        # Distribuir intentos entre estrategias disponibles
        attempts_per_strategy = max_attempts // len(strategies)
        timeout_per_strategy = timeout / len(strategies)
        
        for strategy in strategies:
            strategy_start = time.time()
            logger.info(f"Probando estrategia de construcción: {strategy}")
            
            for attempt in range(attempts_per_strategy):
                if time.time() - strategy_start > timeout_per_strategy:
                    logger.info(f"Timeout en estrategia {strategy} después de {attempt} intentos")
                    break
                
                # Construir solución con la estrategia actual
                candidate = self._construct_solution(employees, shifts, alpha, strategy)
                validation_result = validator.validate(candidate, employees, shifts)
                
                if validation_result.is_valid:
                    logger.info(f"Solución inicial válida generada con estrategia {strategy} en {attempt + 1} intentos")
                    return candidate
                
                if (attempt + 1) % 100 == 0:
                    logger.info(f"Estrategia {strategy}: {attempt + 1} intentos realizados")
        
        logger.error("No se pudo generar una solución inicial válida con ninguna estrategia")
        return None
    
    def _construct_solution(self, 
                         employees: List[Employee], 
                         shifts: List[Shift], 
                         alpha: float,
                         strategy: str = "greedy") -> Solution:
        """Construye una solución usando la estrategia especificada."""
        if strategy == "greedy":
            return self._greedy_randomized_construction(employees, shifts, alpha)
        elif strategy == "constructive":
            return self._constructive_solution(employees, shifts)
        elif strategy == "hybrid":
            return self._hybrid_solution(employees, shifts, alpha)
        elif strategy == "minimal":
            return self._minimal_solution(employees, shifts)
        else:
            # Estrategia por defecto
            return self._greedy_randomized_construction(employees, shifts, alpha)
    
    def _track_assignment_state(self, solution: Solution, employees: List[Employee], 
                             shifts: List[Shift]) -> Tuple[Dict, Dict, Dict]:
        """Calcula el estado actual de asignaciones (usado por múltiples métodos)."""
        shift_dict = {s.id: s for s in shifts}
        
        # Rastrear asignaciones
        day_shift_employees = defaultdict(set)
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
        # Poblar con asignaciones existentes en la solución
        for assignment in solution.assignments:
            shift = shift_dict.get(assignment.shift_id)
            if shift:
                day_shift_key = (shift.day, shift.name)
                day_shift_employees[day_shift_key].add(assignment.employee_id)
                employee_hours[assignment.employee_id] += shift.duration_hours
                employee_days[assignment.employee_id].add(shift.day)
        
        return day_shift_employees, employee_hours, employee_days
    
    def _greedy_randomized_construction(
        self, 
        employees: List[Employee], 
        shifts: List[Shift], 
        alpha: float
    ) -> Solution:
        """Construye una solución de manera greedy aleatorizada, intentando respetar restricciones."""
        solution = Solution()
        
        # Ordenar turnos por prioridad (descendente)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        # Inicializar estado
        day_shift_employees, employee_hours, employee_days = self._track_assignment_state(
            solution, employees, shifts
        )
        
        # Calcular valores promedio para normalización
        avg_hourly_cost = sum(e.hourly_cost for e in employees) / len(employees) if employees else 1.0
        avg_max_hours = sum(e.max_hours_per_week for e in employees) / len(employees) if employees else 40.0
        
        for shift in sorted_shifts:
            # Clave para identificar un tipo de turno en un día específico
            day_shift_key = (shift.day, shift.name)
            
            # Determinar cuántos empleados se necesitan asignar para este turno
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue  # Este turno ya tiene los empleados requeridos
            
            # Encontrar empleados calificados para este turno
            qualified_employees = self._find_qualified_employees(
                employees, shift, day_shift_employees, employee_hours, employee_days
            )
            
            # Cuando hay pocos empleados, el costo no debe ser el único criterio
            # Consideramos la carga de trabajo actual para balancear mejor
            def score_employee(e):
                # Un puntaje menor es mejor para la selección
                workload_factor = len(employee_days[e.id]) / 7  # 0 a 1
                hours_factor = employee_hours[e.id] / avg_max_hours  # 0 a ~1
                cost_factor = e.hourly_cost / avg_hourly_cost  # relativo a promedio
                
                # Balance entre costo y distribución de carga
                balance_factor = 0.7  # Dar peso al costo
                
                return (balance_factor * cost_factor) + ((1 - balance_factor) * (workload_factor + hours_factor))
            
            # Ordenar empleados por el puntaje combinado
            sorted_employees = sorted(qualified_employees, key=score_employee)
            
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
                
                # Añadir asignación a la solución y actualizar estado
                self._add_assignment(solution, selected_employee, shift, 
                                 day_shift_employees, employee_hours, employee_days)
        
        return solution
    
    def _find_qualified_employees(self, employees: List[Employee], shift: Shift,
                               day_shift_employees: Dict, employee_hours: Dict,
                               employee_days: Dict) -> List[Employee]:
        """Encuentra empleados calificados para un turno específico."""
        # Clave para identificar un tipo de turno en un día específico
        day_shift_key = (shift.day, shift.name)
        
        qualified = []
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
            
            # Verificar días consecutivos
            max_days = min(7, employee.max_consecutive_days)
            if shift.day in employee_days[employee.id]:
                # Ya tiene un turno ese día, no es problema
                pass
            elif len(employee_days[employee.id]) >= max_days:
                continue
            
            # Este empleado cumple con todas las condiciones
            qualified.append(employee)
        
        return qualified
    
    def _add_assignment(self, solution: Solution, employee: Employee, shift: Shift,
                     day_shift_employees: Dict, employee_hours: Dict, 
                     employee_days: Dict) -> None:
        """Añade una asignación a la solución y actualiza el estado."""
        day_shift_key = (shift.day, shift.name)
        
        # Añadir asignación a la solución
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
    
    def _constructive_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Crear una solución usando un enfoque más determinístico y constructivo."""
        solution = Solution()
        
        # Inicializar estado de asignaciones
        day_shift_employees, employee_hours, employee_days = self._track_assignment_state(
            solution, employees, shifts
        )
        
        # Ordenar turnos por prioridad (descendente) y días
        days_order = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO', 'DOMINGO']
        sorted_shifts = sorted(shifts, 
                              key=lambda s: (
                                  -s.priority,
                                  days_order.index(s.day.name) if s.day.name in days_order else 999,
                                  s.name.name  # ShiftType como MAÑANA, TARDE, NOCHE
                              ))
        
        # Primer paso: asociar cada empleado con los turnos para los que está mejor calificado
        employee_best_shifts = defaultdict(list)
        
        for employee in employees:
            for shift in sorted_shifts:
                # Verificar calificación básica
                if (employee.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(employee.skills)):
                    # Calcular puntuación para este turno
                    preference = employee.get_preference_score(shift.day, shift.name)
                    cost_efficiency = 1.0 / (employee.hourly_cost + 0.1)  # Inverso del costo
                    score = preference * cost_efficiency
                    
                    employee_best_shifts[employee.id].append((shift, score))
            
            # Ordenar turnos para cada empleado por puntuación (mayor primero)
            employee_best_shifts[employee.id].sort(key=lambda x: x[1], reverse=True)
        
        # Segundo paso: Para cada turno, asignar empleados priorizando mejores calificaciones
        for shift in sorted_shifts:
            day_shift_key = (shift.day, shift.name)
            required = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if required <= 0:
                continue  # Este turno ya está cubierto
            
            # Encontrar empleados disponibles y calificados para este turno
            qualified_employees = self._find_qualified_employees(
                employees, shift, day_shift_employees, 
                employee_hours, employee_days
            )
            
            # Ordenar candidatos por preferencia (mayor primero) y costo (menor primero)
            candidates = []
            for employee in qualified_employees:
                preference = employee.get_preference_score(shift.day, shift.name)
                cost = employee.hourly_cost
                candidates.append((employee, preference, cost))
                
            candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
            # Asignar los mejores candidatos
            for i in range(min(required, len(candidates))):
                employee = candidates[i][0]
                self._add_assignment(
                    solution, employee, shift, 
                    day_shift_employees, employee_hours, employee_days
                )
        
        return solution
    
    def _hybrid_solution(self, employees: List[Employee], shifts: List[Shift], alpha: float = 0.5) -> Solution:
        """Crear una solución combinando estrategias greedy y aleatorias."""
        solution = Solution()
        
        # Clasificar turnos por prioridad
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for shift in shifts:
            if shift.priority > 8:  # Alta prioridad
                high_priority.append(shift)
            elif shift.priority > 5:  # Media prioridad
                medium_priority.append(shift)
            else:  # Baja prioridad
                low_priority.append(shift)
        
        # Inicializar estado de asignaciones
        day_shift_employees, employee_hours, employee_days = self._track_assignment_state(
            solution, employees, shifts
        )
        
        # Procesar turnos de alta prioridad de forma determinista
        for shift in high_priority:
            day_shift_key = (shift.day, shift.name)
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue
            
            # Encontrar empleados calificados
            qualified_employees = self._find_qualified_employees(
                employees, shift, day_shift_employees, 
                employee_hours, employee_days
            )
            
            # Calcular scores para cada empleado
            candidates = []
            for employee in qualified_employees:
                # Calcular score: preferencia alta, horas pocas, costo bajo = mejor
                preference = employee.get_preference_score(shift.day, shift.name)
                cost_score = 1.0 / (employee.hourly_cost + 0.1)
                hours_score = 1.0 / (employee_hours[employee.id] + 1.0)
                score = preference * 0.5 + cost_score * 0.3 + hours_score * 0.2
                candidates.append((employee, score))
            
            # Ordenar por score (mayor primero)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Asignar los mejores
            for i in range(min(needed, len(candidates))):
                employee = candidates[i][0]
                self._add_assignment(
                    solution, employee, shift, 
                    day_shift_employees, employee_hours, employee_days
                )
        
        # Proceso aleatorizado para turnos de prioridad media y baja
        for shift in medium_priority + low_priority:
            day_shift_key = (shift.day, shift.name)
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue
            
            # Encontrar empleados disponibles
            qualified_employees = self._find_qualified_employees(
                employees, shift, day_shift_employees, 
                employee_hours, employee_days
            )
            
            # Selección semi-aleatoria usando el parámetro alpha
            if qualified_employees:
                # Determinar el tamaño de la RCL
                rcl_size = max(1, int(alpha * len(qualified_employees)))
                # Tomar los primeros elementos (después de aleatorizar)
                random.shuffle(qualified_employees)
                rcl = qualified_employees[:rcl_size]
                
                # Asignar empleados del RCL
                for i in range(min(needed, len(rcl))):
                    employee = rcl[i]
                    self._add_assignment(
                        solution, employee, shift, 
                        day_shift_employees, employee_hours, employee_days
                    )
        
        return solution
    
    def _minimal_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Crear una solución minimalista enfocada en cubrir los turnos más importantes."""
        solution = Solution()
        
        # Ordenar turnos por prioridad (mayor primero)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        # Tomar solo un subconjunto de los turnos más prioritarios
        critical_shifts = sorted_shifts[:len(sorted_shifts) // 2]  # Solo la mitad superior
        
        # Inicializar estado
        day_shift_employees, employee_hours, employee_days = self._track_assignment_state(
            solution, employees, shifts
        )
        
        # Para cada empleado, asignar un turno crítico para el que esté calificado
        employee_shifts = defaultdict(list)
        for employee in employees:
            for shift in critical_shifts:
                if (employee.is_available(shift.day, shift.name) and 
                    shift.required_skills.issubset(employee.skills)):
                    employee_shifts[employee.id].append(shift)
        
        # Asignar empleados a sus mejores turnos
        for emp_id, possible_shifts in employee_shifts.items():
            # Ordenar aleatoriamente para diversificar
            random.shuffle(possible_shifts)
            
            employee = next((e for e in employees if e.id == emp_id), None)
            if not employee:
                continue
                
            for shift in possible_shifts:
                day_shift_key = (shift.day, shift.name)
                
                # Solo asignar si aún se necesitan empleados para este turno
                if len(day_shift_employees[day_shift_key]) < shift.required_employees:
                    # Verificar que no esté ya asignado a este turno
                    if emp_id in day_shift_employees[day_shift_key]:
                        continue
                        
                    # Verificar límites de horas
                    if employee_hours[emp_id] + shift.duration_hours > employee.max_hours_per_week:
                        continue
                        
                    # Verificar días consecutivos
                    if (shift.day not in employee_days[emp_id] and 
                        len(employee_days[emp_id]) >= employee.max_consecutive_days):
                        continue
                    
                    # Añadir asignación
                    self._add_assignment(
                        solution, employee, shift, 
                        day_shift_employees, employee_hours, employee_days
                    )
                    
                    # Limitar a máximo un par de turnos por empleado en este enfoque minimalista
                    if len(employee_days[emp_id]) >= 2:
                        break
        
        return solution

    def _local_search(self, solution: Solution, employees: List[Employee], shifts: List[Shift], 
                     max_iterations: int, validator: SolutionValidator) -> Optional[Solution]:
        """Mejora la solución mediante búsqueda local, manteniendo factibilidad."""
        current_solution = solution.clone()
        current_fitness = self._calculate_fitness(current_solution, employees, shifts)
        
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        # Obtener estado actual de asignaciones
        day_shift_employees, employee_hours, employee_days = self._track_assignment_state(
            current_solution, employees, shifts
        )
        
        # Estrategias de búsqueda local
        local_search_methods = [
            self._try_employee_replacement,
            self._try_load_rebalancing, 
            self._try_coverage_improvement
        ]
        
        for iteration in range(max_iterations):
            improved = False
            
            # Seleccionar un método de búsqueda local al azar
            search_method = random.choice(local_search_methods)
            
            # Aplicar el método seleccionado
            new_solution, success = search_method(
                current_solution, employees, shifts, 
                employee_dict, shift_dict,
                day_shift_employees, employee_hours, employee_days,
                validator, current_fitness
            )
            
            if success:
                current_solution = new_solution
                current_fitness = self._calculate_fitness(current_solution, employees, shifts)
                
                # Actualizar estado de asignaciones
                day_shift_employees, employee_hours, employee_days = self._track_assignment_state(
                    current_solution, employees, shifts
                )
                
                improved = True
            
            # Si no se encontró ninguna mejora, detener la búsqueda local con baja probabilidad
            if not improved and random.random() < 0.1:  # 10% de probabilidad de terminar temprano
                break
        
        # Validar la solución final (por precaución)
        validation_result = validator.validate(current_solution, employees, shifts)
        
        if validation_result.is_valid:
            return current_solution
        else:
            # Si de algún modo la solución final es inválida, devolver None
            logger.warning("La búsqueda local generó una solución inválida. Usando solución original.")
            return None
    
    def _try_employee_replacement(self, current_solution, employees, shifts, 
                               employee_dict, shift_dict, 
                               day_shift_employees, employee_hours, employee_days,
                               validator, current_fitness):
        """Intenta mejorar la solución reemplazando un empleado por otro más económico."""
        # Seleccionar una asignación aleatoria para intentar reemplazar
        if not current_solution.assignments:
            return current_solution, False
            
        assignment = random.choice(current_solution.assignments)
        shift = shift_dict.get(assignment.shift_id)
        current_employee_id = assignment.employee_id
        
        if not shift:
            return current_solution, False
            
        day_shift_key = (shift.day, shift.name)
        
        # Encontrar empleados potenciales para reemplazo
        potential_replacements = [
            e for e in employees 
            if e.id != current_employee_id and
            e.is_available(shift.day, shift.name) and
            shift.required_skills.issubset(e.skills) and
            e.id not in day_shift_employees[day_shift_key]
        ]
        
        if not potential_replacements:
            return current_solution, False
            
        # Seleccionar un empleado al azar
        new_employee = random.choice(potential_replacements)
        
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
        
        if validation_result.is_valid:
            # Calcular fitness de la nueva solución
            new_fitness = self._calculate_fitness(new_solution, employees, shifts)
            
            # Aceptar si mejora
            if new_fitness > current_fitness:
                return new_solution, True
        
        return current_solution, False
    
    def _try_load_rebalancing(self, current_solution, employees, shifts, 
                          employee_dict, shift_dict, 
                          day_shift_employees, employee_hours, employee_days,
                          validator, current_fitness):
        """Intenta rebalancear la carga de trabajo entre empleados."""
        # Detectar empleados con mucha y poca carga
        employee_load = defaultdict(int)
        for a in current_solution.assignments:
            employee_load[a.employee_id] += 1
        
        if not employee_load:
            return current_solution, False
            
        # Identificar empleados con mayor y menor carga
        max_load = max(employee_load.values())
        min_load = min(employee_load.values())
        
        if max_load - min_load < 2:  # No hay diferencia significativa de carga
            return current_solution, False
            
        # Buscar un empleado sobrecargado
        overloaded_employees = [e_id for e_id, load in employee_load.items() if load == max_load]
        
        if not overloaded_employees:
            return current_solution, False
            
        overloaded_id = random.choice(overloaded_employees)
        
        # Buscar empleados con poca carga
        underloaded_employees = [
            e.id for e in employees 
            if e.id in employee_load and employee_load[e.id] <= min_load + 1
        ]
        
        # Intentar transferir un turno
        for a in current_solution.assignments:
            if a.employee_id != overloaded_id:
                continue
                
            shift = shift_dict.get(a.shift_id)
            if not shift:
                continue
                
            # Probar transferir este turno
            for underloaded_id in underloaded_employees:
                underloaded_employee = employee_dict.get(underloaded_id)
                if not underloaded_employee:
                    continue
                    
                # Verificar si el empleado con poca carga puede tomar este turno
                if (underloaded_employee.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(underloaded_employee.skills)):
                    
                    # Crear solución con transferencia
                    new_solution = current_solution.clone()
                    new_assignments = []
                    
                    for current_a in new_solution.assignments:
                        if current_a.shift_id == a.shift_id and current_a.employee_id == overloaded_id:
                            # Transferir asignación
                            new_assignments.append(Assignment(
                                employee_id=underloaded_id,
                                shift_id=shift.id,
                                cost=underloaded_employee.hourly_cost * shift.duration_hours
                            ))
                        else:
                            new_assignments.append(current_a)
                    
                    new_solution.assignments = new_assignments
                    
                    # Validar nueva solución
                    validation_result = validator.validate(new_solution, employees, shifts)
                    
                    if validation_result.is_valid:
                        new_fitness = self._calculate_fitness(new_solution, employees, shifts)
                        
                        # Para rebalanceo, aceptamos incluso con fitness ligeramente inferior
                        if new_fitness >= current_fitness * 0.98:
                            return new_solution, True
        
        return current_solution, False
    
    def _try_coverage_improvement(self, current_solution, employees, shifts, 
                               employee_dict, shift_dict, 
                               day_shift_employees, employee_hours, employee_days,
                               validator, current_fitness):
        """Intenta mejorar la cobertura de turnos con cobertura insuficiente."""
        # Encontrar turnos con cobertura insuficiente
        for shift in shifts:
            day_shift_key = (shift.day, shift.name)
            assigned_count = len(day_shift_employees[day_shift_key])
            
            if assigned_count >= shift.required_employees:
                continue  # Este turno ya tiene suficiente cobertura
                
            # Encontrar empleados disponibles para este turno
            available_employees = [
                e for e in employees 
                if e.id not in day_shift_employees[day_shift_key] and
                e.is_available(shift.day, shift.name) and
                shift.required_skills.issubset(e.skills)
            ]
            
            if not available_employees:
                continue
                
            # Seleccionar un empleado al azar
            selected_employee = random.choice(available_employees)
            
            # Crear nueva solución con esta asignación adicional
            new_solution = current_solution.clone()
            new_assignment = Assignment(
                employee_id=selected_employee.id,
                shift_id=shift.id,
                cost=selected_employee.hourly_cost * shift.duration_hours
            )
            new_solution.add_assignment(new_assignment)
            
            # Validar la nueva solución
            validation_result = validator.validate(new_solution, employees, shifts)
            
            if validation_result.is_valid:
                new_fitness = self._calculate_fitness(new_solution, employees, shifts)
                
                # Aceptamos mejoras o incluso soluciones con fitness similar
                # para completar la cobertura de turnos
                if new_fitness >= current_fitness * 0.95:
                    return new_solution, True
        
        return current_solution, False
    
    def _calculate_fitness(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> float:
        """Calcula la puntuación de fitness para una solución factible (solo considera costo, no violaciones)."""
        if not solution.assignments:
            return 0.0
            
        # Mapeos para acceso rápido
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        # Calcular el costo total y datos de cobertura
        total_cost = 0.0
        employee_load = defaultdict(int)
        shift_coverage = defaultdict(set)
        shift_requirements = {s.id: s.required_employees for s in shifts}
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Cálculo del costo de esta asignación
                assignment_cost = employee.hourly_cost * shift.duration_hours
                assignment.cost = assignment_cost
                total_cost += assignment_cost
                
                # Actualizar carga del empleado
                employee_load[emp_id] += 1
                
                # Actualizar cobertura del turno
                shift_coverage[shift_id].add(emp_id)
        
        # Asegurar un costo mínimo
        total_cost = max(10.0, total_cost)
        solution.total_cost = total_cost
        
        # Con restricciones duras, las violaciones siempre son 0
        solution.constraint_violations = 0
        
        # El fitness es inversamente proporcional al costo
        # Usamos esta fórmula para evitar valores extremos
        fitness_score = 1000.0 / (1.0 + total_cost / 100.0)
        
        # --- Bonificaciones ---
        
        # 1. Bonificación por preferencias de empleados
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
        
        # 2. Bonificación por cobertura de turnos
        total_shifts = len(shifts)
        covered_shifts = sum(1 for s_id, employees in shift_coverage.items() 
                          if len(employees) >= shift_requirements.get(s_id, 0))
        
        coverage_rate = covered_shifts / total_shifts if total_shifts > 0 else 0
        coverage_bonus = coverage_rate * 0.8 * fitness_score
        
        # 3. Bonificación por balanceo de carga
        if employee_load:
            load_values = list(employee_load.values())
            avg_load = sum(load_values) / len(load_values)
            variance = sum((x - avg_load) ** 2 for x in load_values) / len(load_values)
            std_dev = variance ** 0.5
            
            # Normalizar la desviación estándar
            max_possible_std = ((len(shifts) / len(employees)) ** 0.5) if employees else 1.0
            normalized_std = std_dev / max_possible_std if max_possible_std > 0 else 1.0
            
            # Bonificación por distribución equilibrada (menor desviación es mejor)
            balance_bonus = (1.0 - normalized_std) * 0.3 * fitness_score
        else:
            balance_bonus = 0
        
        # Fitness final incluye preferencias, cobertura y balance
        final_fitness = fitness_score + preference_bonus + coverage_bonus + balance_bonus
        
        # Guardar el fitness en la solución para referencias futuras
        solution.fitness_score = max(0.1, final_fitness)
        
        return solution.fitness_score