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
            "alpha": 0.5,  # Aumentado a 0.5 para ser más aleatorio con pocos empleados
            "local_search_iterations": 50,
            "max_construction_attempts": 500,  # Aumentado a 500 para dar más oportunidades
            "max_initial_solution_attempts": 1000,  # Nuevo: intentos específicos para solución inicial
            "validation_timeout": 30,  # Aumentado a 30 segundos para permitir más intentos
            "use_constructive_approach": True,  # Nuevo: usar enfoque constructivo si aleatorio falla
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
        alpha = config.get("alpha", 0.5)  # Alpha aumentado
        local_search_iterations = config.get("local_search_iterations", 50)
        max_construction_attempts = config.get("max_construction_attempts", 500)  # Aumentado
        max_initial_solution_attempts = config.get("max_initial_solution_attempts", 1000)  # Nuevo
        validation_timeout = config.get("validation_timeout", 30)  # Aumentado
        use_constructive_approach = config.get("use_constructive_approach", True)  # Nuevo
        
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
        logger.info(f"Optimizado para conjunto pequeño de {len(employees)} empleados")
        
        # Intentar generar solución inicial con diferentes estrategias
        best_solution = self._create_initial_solution(
            employees, shifts, validator, 
            max_initial_solution_attempts, validation_timeout,
            alpha, use_constructive_approach
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
                    candidate = self._greedy_randomized_construction(
                        employees, shifts, current_alpha
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
        
        # Registrar métricas finales
        logger.info(f"Optimización GRASP completada. Fitness de la mejor solución: {best_fitness:.4f}")
        logger.info(f"Métricas: {metrics['valid_solutions_found']}/{metrics['construction_attempts']} "
                  f"soluciones válidas construidas. {metrics['local_search_improvements']} mejoras por búsqueda local.")
        
        return best_solution
    
    def _create_initial_solution(self,
                              employees: List[Employee],
                              shifts: List[Shift],
                              validator: SolutionValidator,
                              max_attempts: int,
                              timeout: float,
                              alpha: float,
                              use_constructive_approach: bool) -> Optional[Solution]:
        """Crear una solución inicial válida usando varias estrategias."""
        logger.info("Generando solución inicial para GRASP...")
        start_time = time.time()
        
        # Estrategia 1: Intentar construir con GRASP estándar
        for attempt in range(max_attempts // 3):
            if time.time() - start_time > timeout / 3:
                logger.info(f"Timeout en estrategia 1 después de {attempt} intentos")
                break
                
            candidate = self._greedy_randomized_construction(employees, shifts, alpha)
            validation_result = validator.validate(candidate, employees, shifts)
            
            if validation_result.is_valid:
                logger.info(f"Solución inicial válida generada con GRASP en {attempt + 1} intentos")
                return candidate
                
            if (attempt + 1) % 100 == 0:
                logger.info(f"Estrategia 1: {attempt + 1} intentos realizados")
        
        # Estrategia 2: Usar construcción determinista si está habilitado
        if use_constructive_approach:
            logger.info("Intentando estrategia 2: Enfoque constructivo determinista")
            constructive = self._create_constructive_solution(employees, shifts)
            
            validation_result = validator.validate(constructive, employees, shifts)
            if validation_result.is_valid:
                logger.info("Solución inicial válida generada usando enfoque constructivo")
                return constructive
        
        # Estrategia 3: Enfoque híbrido - construir solución gradualmente para turnos prioritarios
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time > 0:
            logger.info("Intentando estrategia 3: Enfoque híbrido por prioridades")
            for attempt in range(max_attempts // 3):
                if time.time() - start_time > timeout * 2 / 3:
                    logger.info(f"Timeout en estrategia 3 después de {attempt} intentos")
                    break
                    
                hybrid = self._create_hybrid_solution(employees, shifts)
                validation_result = validator.validate(hybrid, employees, shifts)
                
                if validation_result.is_valid:
                    logger.info(f"Solución inicial válida generada con enfoque híbrido en {attempt + 1} intentos")
                    return hybrid
                
                if (attempt + 1) % 100 == 0:
                    logger.info(f"Estrategia 3: {attempt + 1} intentos realizados")
        
        # Estrategia 4: Último recurso - Solución con asignaciones mínimas necesarias
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time > 0:
            logger.info("Intentando estrategia 4: Enfoque minimalista (asignaciones mínimas)")
            for attempt in range(max_attempts // 3):
                if time.time() - start_time > timeout:
                    logger.info(f"Timeout en estrategia 4 después de {attempt} intentos")
                    break
                    
                minimal = self._create_minimal_solution(employees, shifts)
                validation_result = validator.validate(minimal, employees, shifts)
                
                if validation_result.is_valid:
                    logger.info(f"Solución inicial válida generada con enfoque minimalista en {attempt + 1} intentos")
                    return minimal
                    
                if (attempt + 1) % 50 == 0:
                    logger.info(f"Estrategia 4: {attempt + 1} intentos realizados")
        
        logger.error("No se pudo generar una solución inicial válida con ninguna estrategia")
        return None
    
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
        
        # Rastrear qué empleados ya han sido asignados a cada tipo de turno en cada día
        day_shift_employees = defaultdict(set)
        
        # Rastrear horas y días asignados por empleado
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
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
                
                # Verificar días consecutivos
                max_days = min(7, employee.max_consecutive_days)
                if shift.day in employee_days[employee.id]:
                    # Ya tiene un turno ese día, no es problema
                    pass
                elif len(employee_days[employee.id]) >= max_days:
                    continue
                
                # Este empleado cumple con todas las condiciones
                qualified_employees.append(employee)
            
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
            # Con pocos empleados, usar una RCL más grande proporcional al alpha
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
    
    def _create_constructive_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Crear una solución usando un enfoque más determinístico y constructivo."""
        solution = Solution()
        
        # Crear mapeos para acceso rápido
        shift_dict = {s.id: s for s in shifts}
        employee_dict = {e.id: e for e in employees}
        
        # Rastrear asignaciones
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        day_shift_employees = defaultdict(set)
        
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
            candidates = []
            for employee in employees:
                emp_id = employee.id
                
                # Verificar si ya está asignado a este turno
                if emp_id in day_shift_employees[day_shift_key]:
                    continue
                
                # Verificar disponibilidad básica
                if not employee.is_available(shift.day, shift.name):
                    continue
                
                # Verificar habilidades
                if not shift.required_skills.issubset(employee.skills):
                    continue
                
                # Verificar límite de horas
                if employee_hours[emp_id] + shift.duration_hours > employee.max_hours_per_week:
                    continue
                
                # Verificar días consecutivos (simplificado)
                if shift.day in employee_days[emp_id] or len(employee_days[emp_id]) < employee.max_consecutive_days:
                    # Este empleado cumple con todas las condiciones
                    preference = employee.get_preference_score(shift.day, shift.name)
                    cost = employee.hourly_cost
                    candidates.append((employee, preference, cost))
            
            # Ordenar candidatos por preferencia (mayor primero) y costo (menor primero)
            candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
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
    
    def _create_hybrid_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
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
        
        # Mapeos para seguimiento
        day_shift_employees = defaultdict(set)
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
        # Procesar turnos de alta prioridad de forma determinista
        for shift in high_priority:
            day_shift_key = (shift.day, shift.name)
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue
            
            # Encontrar mejores empleados para este turno
            candidates = []
            for employee in employees:
                if (employee.id not in day_shift_employees[day_shift_key] and
                    employee.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(employee.skills) and
                    employee_hours[employee.id] + shift.duration_hours <= employee.max_hours_per_week):
                    
                    # Verificar días consecutivos
                    if (shift.day in employee_days[employee.id] or 
                        len(employee_days[employee.id]) < employee.max_consecutive_days):
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
        
        # Proceso aleatorizado para turnos de prioridad media y baja
        for shift in medium_priority + low_priority:
            day_shift_key = (shift.day, shift.name)
            needed = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if needed <= 0:
                continue
            
            # Encontrar empleados disponibles
            available = []
            for employee in employees:
                if (employee.id not in day_shift_employees[day_shift_key] and
                    employee.is_available(shift.day, shift.name) and
                    shift.required_skills.issubset(employee.skills) and
                    employee_hours[employee.id] + shift.duration_hours <= employee.max_hours_per_week):
                    
                    # Verificar días consecutivos
                    if (shift.day in employee_days[employee.id] or 
                        len(employee_days[employee.id]) < employee.max_consecutive_days):
                        available.append(employee)
            
            # Selección semi-aleatoria
            if available:
                random.shuffle(available)  # Aleatorizar
                for i in range(min(needed, len(available))):
                    employee = available[i]
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
    
    def _create_minimal_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Crear una solución minimalista enfocada en cubrir los turnos más importantes."""
        solution = Solution()
        
        # Ordenar turnos por prioridad (mayor primero)
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        # Tomar solo un subconjunto de los turnos más prioritarios
        critical_shifts = sorted_shifts[:len(sorted_shifts) // 2]  # Solo la mitad superior
        
        # Para cada empleado, asignar un turno crítico para el que esté calificado
        employee_shifts = defaultdict(list)
        for employee in employees:
            for shift in critical_shifts:
                if (employee.is_available(shift.day, shift.name) and 
                    shift.required_skills.issubset(employee.skills)):
                    employee_shifts[employee.id].append(shift)
        
        # Seguimiento de asignaciones
        day_shift_employees = defaultdict(set)
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
        # Asignar empleados a sus mejores turnos
        for emp_id, possible_shifts in employee_shifts.items():
            # Ordenar aleatoriamente para diversificar
            random.shuffle(possible_shifts)
            
            for shift in possible_shifts:
                day_shift_key = (shift.day, shift.name)
                
                # Solo asignar si aún se necesitan empleados para este turno
                if len(day_shift_employees[day_shift_key]) < shift.required_employees:
                    # Verificar que no esté ya asignado a este turno
                    if emp_id in day_shift_employees[day_shift_key]:
                        continue
                        
                    # Verificar límites de horas
                    employee = next((e for e in employees if e.id == emp_id), None)
                    if not employee or employee_hours[emp_id] + shift.duration_hours > employee.max_hours_per_week:
                        continue
                        
                    # Verificar días consecutivos
                    if (shift.day not in employee_days[emp_id] and 
                        len(employee_days[emp_id]) >= employee.max_consecutive_days):
                        continue
                    
                    # Crear asignación
                    assignment = Assignment(
                        employee_id=emp_id,
                        shift_id=shift.id,
                        cost=employee.hourly_cost * shift.duration_hours
                    )
                    solution.add_assignment(assignment)
                    
                    # Actualizar registros
                    day_shift_employees[day_shift_key].add(emp_id)
                    employee_hours[emp_id] += shift.duration_hours
                    employee_days[emp_id].add(shift.day)
                    
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
        
        # Rastrear asignaciones actuales por día y tipo de turno
        day_shift_employees = defaultdict(set)
        for assignment in current_solution.assignments:
            shift = shift_dict.get(assignment.shift_id)
            if shift:
                day_shift_key = (shift.day, shift.name)
                day_shift_employees[day_shift_key].add(assignment.employee_id)
        
        # Enfoques de búsqueda local para pocos empleados:
        # 1. Reemplazos para mejorar costo
        # 2. Rebalanceo de la carga de trabajo
        # 3. Completar turnos con cobertura insuficiente
        
        for iteration in range(max_iterations):
            improved = False
            
            # 1. Intento de mejora por reemplazo (más restrictivo con pocos empleados)
            if random.random() < 0.7:  # 70% de probabilidad
                # Seleccionar una asignación aleatoria para intentar reemplazar
                if current_solution.assignments:
                    assignment = random.choice(current_solution.assignments)
                    shift = shift_dict.get(assignment.shift_id)
                    current_employee_id = assignment.employee_id
                    
                    if shift:
                        day_shift_key = (shift.day, shift.name)
                        
                        # Encontrar empleados potenciales para reemplazo
                        potential_replacements = [
                            e for e in employees 
                            if e.id != current_employee_id and
                            e.is_available(shift.day, shift.name) and
                            shift.required_skills.issubset(e.skills) and
                            e.id not in day_shift_employees[day_shift_key]
                        ]
                        
                        # Ordenar primero por costo y luego probar
                        if potential_replacements:
                            # Usando una búsqueda más rápida: probar solo un candidato aleatorio
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
                                    current_solution = new_solution
                                    current_fitness = new_fitness
                                    
                                    # Actualizar registro de asignaciones
                                    day_shift_employees[day_shift_key].remove(current_employee_id)
                                    day_shift_employees[day_shift_key].add(new_employee.id)
                                    
                                    improved = True
            
            # 2. Intento de rebalanceo de carga (específico para pocos empleados)
            if not improved and random.random() < 0.5:
                # Detectar empleados con mucha y poca carga
                employee_load = defaultdict(int)
                for a in current_solution.assignments:
                    shift = shift_dict.get(a.shift_id)
                    if shift:
                        employee_load[a.employee_id] += 1
                
                if employee_load:
                    # Identificar empleados con mayor y menor carga
                    max_load = max(employee_load.values())
                    min_load = min(employee_load.values())
                    
                    if max_load - min_load >= 2:  # Diferencia significativa de carga
                        # Buscar un empleado sobrecargado
                        overloaded_employees = [e_id for e_id, load in employee_load.items() if load == max_load]
                        
                        if overloaded_employees:
                            overloaded_id = random.choice(overloaded_employees)
                            
                            # Buscar empleados con poca carga
                            underloaded_employees = [
                                e.id for e in employees 
                                if e.id in employee_load and employee_load[e.id] <= min_load + 1
                            ]
                            
                            # Intentar transferir un turno
                            for a in current_solution.assignments:
                                if a.employee_id == overloaded_id:
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
                                                    current_solution = new_solution
                                                    current_fitness = new_fitness
                                                    improved = True
                                                    break
                                    
                                    if improved:
                                        break
            
            # 3. Completar turnos con cobertura insuficiente (relevante con pocos empleados)
            if not improved:
                # Encontrar turnos con cobertura insuficiente
                for shift in shifts:
                    day_shift_key = (shift.day, shift.name)
                    assigned_count = len(day_shift_employees[day_shift_key])
                    
                    if assigned_count < shift.required_employees:
                        # Encontrar empleados disponibles para este turno
                        available_employees = [
                            e for e in employees 
                            if e.id not in day_shift_employees[day_shift_key] and
                            e.is_available(shift.day, shift.name) and
                            shift.required_skills.issubset(e.skills)
                        ]
                        
                        if available_employees:
                            # Seleccionar un empleado al azar o con menor carga
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
                            
                            if validation_result.is_valid:  # Corregido aquí
                                new_fitness = self._calculate_fitness(new_solution, employees, shifts)
                                
                                # Aceptamos mejoras o incluso soluciones con fitness similar
                                # para completar la cobertura de turnos
                                if new_fitness >= current_fitness * 0.95:
                                    current_solution = new_solution
                                    current_fitness = new_fitness
                                    
                                    day_shift_employees[day_shift_key].add(selected_employee.id)
                                    improved = True
                                    break
            
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
        
        # Con pocos empleados, dar mayor peso a la cobertura de turnos
        coverage_bonus = coverage_rate * 0.8 * fitness_score  # Aumento de 0.5 a 0.8
        
        # Evaluar el balanceo de carga entre empleados
        employee_load = defaultdict(int)
        for a in solution.assignments:
            employee_load[a.employee_id] += 1
        
        # Calcular la desviación estándar de la carga (menor es mejor)
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
        final_fitness += coverage_bonus + balance_bonus
        
        # Guardar el fitness en la solución para referencias futuras
        solution.fitness_score = max(0.1, final_fitness)
        
        return solution.fitness_score