import random
import logging
from typing import List, Dict, Any, Tuple, Optional
import time
import collections
from mh_optimizacion_turnos.domain.services.optimizer_strategy import OptimizerStrategy
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.models.assignment import Assignment

logger = logging.getLogger(__name__)

class GeneticAlgorithmOptimizer(OptimizerStrategy):
    """Implementación de Algoritmo Genético para la optimización de turnos con restricciones duras."""
    
    def get_name(self) -> str:
        return "Genetic Algorithm Optimizer"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "population_size": 30,            # Reducido para enfocarse en calidad
            "generations": 50,                # Reducido para que las pruebas sean más rápidas
            "mutation_rate": 0.15,            # Aumentado para más diversidad
            "crossover_rate": 0.85,           # Aumentado ligeramente
            "elitism_count": 3,               # Reducido para dar más espacio a soluciones nuevas
            "tournament_size": 3,
            "max_initialization_attempts": 1500, # Aumentado para intentar más soluciones iniciales
            "relaxation_factor": 0.75,        # Nuevo: factor para relajar restricciones temporalmente
            "max_repair_attempts": 50,        # Intentos para reparar soluciones inválidas
            "validation_timeout": 15,         # Aumentado: tiempo para generar soluciones válidas
            "use_constructive_approach": True, # Nuevo: usar enfoque constructivo si aleatorio falla
            "metrics": {
                "enabled": True,
                "track_evaluations": True,
                "track_validation_time": True
            }
        }
    
    def _init_metrics(self) -> Dict[str, Any]:
        """Inicializa y devuelve un diccionario de métricas para seguimiento de rendimiento."""
        return {
            "objective_evaluations": 0,
            "validation_time": 0,
            "repair_attempts": 0,
            "repair_success_rate": 0
        }
    
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimiza la asignación de turnos usando Algoritmo Genético con restricciones duras."""
        # Usar configuración predeterminada si no se proporciona
        if config is None:
            config = self.get_default_config()
        
        # Extraer configuración con valores por defecto
        population_size = config.get("population_size", 30)
        generations = config.get("generations", 50)
        mutation_rate = config.get("mutation_rate", 0.15)
        crossover_rate = config.get("crossover_rate", 0.85)
        elitism_count = config.get("elitism_count", 3)
        tournament_size = config.get("tournament_size", 3)
        max_initialization_attempts = config.get("max_initialization_attempts", 1500)
        max_repair_attempts = config.get("max_repair_attempts", 50)
        validation_timeout = config.get("validation_timeout", 15)
        use_constructive_approach = config.get("use_constructive_approach", True)
        relaxation_factor = config.get("relaxation_factor", 0.75)
        
        # Inicialización de métricas
        metrics = self._init_metrics()
        
        # Crear validador
        validator = SolutionValidator()
        
        logger.info(f"Iniciando optimización con algoritmo genético con {population_size} individuos para {generations} generaciones")
        logger.info(f"Usando enfoque de restricciones duras (solo soluciones factibles)")
        
        # Inicializar población con soluciones válidas
        start_time = time.time()
        population = self._initialize_valid_population(
            employees=employees, 
            shifts=shifts, 
            population_size=population_size, 
            validator=validator, 
            max_attempts=max_initialization_attempts, 
            timeout=validation_timeout,
            use_constructive_approach=use_constructive_approach, 
            relaxation_factor=relaxation_factor
        )
        initialization_time = time.time() - start_time
        logger.info(f"Población inicial de {len(population)} soluciones generada en {initialization_time:.2f} segundos")
        
        if not population:
            raise ValueError("No se pudo generar ninguna solución válida. Las restricciones son demasiado estrictas o inconsistentes.")
        
        # Evaluar fitness para la población inicial
        fitness_scores = self._evaluate_population_fitness(population, employees, shifts, metrics)
        
        # Bucle principal del algoritmo genético
        for generation in range(generations):
            start_gen_time = time.time()
            
            # Aplicar elitismo - mantener las mejores soluciones
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elitism_count]
            elite_solutions = [population[i].clone() for i in elite_indices]
            
            new_population = []
            repair_attempts = 0
            successful_repairs = 0
            
            # Generar nueva población
            while len(new_population) < population_size - elitism_count:
                # Selección
                parent1 = self._tournament_selection(population, fitness_scores, tournament_size)
                parent2 = self._tournament_selection(population, fitness_scores, tournament_size)
                
                # Cruce
                if random.random() < crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.clone(), parent2.clone()
                
                # Mutación y reparación si es necesario
                offspring1 = self._mutate_and_repair(
                    offspring1, employees, shifts, mutation_rate, 
                    validator, max_repair_attempts
                )
                
                if offspring1:  # Si la reparación fue exitosa
                    new_population.append(offspring1)
                    successful_repairs += 1
                
                repair_attempts += 1
                
                # Solo añadir el segundo hijo si hay espacio
                if len(new_population) < population_size - elitism_count:
                    offspring2 = self._mutate_and_repair(
                        offspring2, employees, shifts, mutation_rate, 
                        validator, max_repair_attempts
                    )
                    
                    if offspring2:  # Si la reparación fue exitosa
                        new_population.append(offspring2)
                        successful_repairs += 1
                    
                    repair_attempts += 1
                
                # Verificar si estamos teniendo dificultades para generar soluciones válidas
                if repair_attempts >= max_repair_attempts * 2 and len(new_population) < (population_size - elitism_count) / 2:
                    logger.warning(f"Dificultad para generar soluciones válidas en la generación {generation}. "
                                  f"Completando población con clones de soluciones elite.")
                    
                    # Completar con clones de las soluciones elite si es necesario
                    needed = population_size - elitism_count - len(new_population)
                    for i in range(min(needed, len(elite_solutions))):
                        new_population.append(elite_solutions[i % len(elite_solutions)].clone())
                    
                    break  # Salir del bucle de generación
            
            # Actualizar métricas de reparación
            self._update_repair_metrics(metrics, repair_attempts, successful_repairs)
            
            # Añadir las soluciones elite a la nueva población
            population = new_population + elite_solutions
            
            # Recalcular puntuaciones de fitness
            fitness_scores = self._evaluate_population_fitness(population, employees, shifts, metrics)
            
            # Registrar progreso
            self._log_generation_progress(generation, generations, fitness_scores, time.time() - start_gen_time, metrics)
        
        # Devolver la mejor solución
        best_idx = fitness_scores.index(max(fitness_scores))
        best_solution = population[best_idx]
        
        # Calcular el costo real antes de devolver
        self._calculate_solution_cost(best_solution, employees, shifts)
        
        logger.info(f"Optimización completa. Fitness de la mejor solución: {fitness_scores[best_idx]}")
        logger.info(f"Métricas: {metrics['objective_evaluations']} evaluaciones de función objetivo, "
                  f"Tasa de reparación: {metrics['repair_success_rate']:.2f}")
        
        return best_solution
    
    def _update_repair_metrics(self, metrics: Dict[str, Any], repair_attempts: int, successful_repairs: int) -> None:
        """Actualiza las métricas de reparación basadas en los intentos recientes."""
        metrics["repair_attempts"] += repair_attempts
        if repair_attempts > 0:
            current_repair_rate = successful_repairs / repair_attempts
            # Promedio ponderado con métricas anteriores
            if metrics["repair_success_rate"] == 0:
                metrics["repair_success_rate"] = current_repair_rate
            else:
                metrics["repair_success_rate"] = (
                    metrics["repair_success_rate"] * 0.7 + current_repair_rate * 0.3
                )
    
    def _evaluate_population_fitness(self, population: List[Solution], employees: List[Employee], 
                                   shifts: List[Shift], metrics: Dict[str, Any]) -> List[float]:
        """Evalúa la aptitud de todas las soluciones en la población."""
        fitness_scores = []
        for solution in population:
            fitness = self._calculate_fitness(solution, employees, shifts)
            fitness_scores.append(fitness)
            metrics["objective_evaluations"] += 1
        return fitness_scores
    
    def _log_generation_progress(self, generation: int, total_generations: int, 
                              fitness_scores: List[float], gen_time: float, metrics: Dict[str, Any]) -> None:
        """Registra el progreso de la generación actual."""
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        if generation % 10 == 0 or generation == total_generations - 1:
            logger.info(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, "
                        f"Fitness promedio = {avg_fitness:.4f}, "
                        f"Tiempo: {gen_time:.2f}s, "
                        f"Tasa de reparación: {metrics['repair_success_rate']:.2f}")

    def _initialize_valid_population(
        self, 
        employees: List[Employee], 
        shifts: List[Shift], 
        population_size: int,
        validator: SolutionValidator,
        max_attempts: int,
        timeout: float,
        use_constructive_approach: bool = True,
        relaxation_factor: float = 0.75
    ) -> List[Solution]:
        """Inicializa una población de soluciones válidas."""
        population = []
        attempts = 0
        start_time = time.time()
        
        max_attempts_per_solution = max(1000, max_attempts // population_size)
        
        # Intentar generar soluciones aleatorias válidas
        while len(population) < population_size and attempts < max_attempts and (time.time() - start_time <= timeout * population_size):
            # Generar una solución candidata aleatoria
            solution = self._generate_candidate_solution(employees, shifts)
            attempts += 1
            
            # Validar la solución
            validation_result = validator.validate(solution, employees, shifts)
            
            if validation_result.is_valid:
                # Calcular costo real
                self._calculate_solution_cost(solution, employees, shifts)
                population.append(solution)
                
            # Informar progreso periódicamente
            if attempts % 500 == 0:
                success_rate = len(population) / attempts * 100 if attempts > 0 else 0
                logger.info(f"Generación de población inicial: {len(population)}/{population_size} soluciones válidas "
                          f"en {attempts} intentos ({(time.time() - start_time):.2f}s), tasa de éxito: {success_rate:.2f}%")
        
        # Si no se ha generado suficientes soluciones, intentar enfoque constructivo
        if len(population) < population_size:
            population = self._try_alternative_population_creation(
                employees, shifts, population, population_size, validator, 
                use_constructive_approach, relaxation_factor
            )
        
        return population
    
    def _try_alternative_population_creation(
        self,
        employees: List[Employee],
        shifts: List[Shift],
        existing_population: List[Solution],
        target_size: int,
        validator: SolutionValidator,
        use_constructive_approach: bool = True,
        relaxation_factor: float = 0.75
    ) -> List[Solution]:
        """Intenta estrategias alternativas para crear una población inicial cuando el enfoque aleatorio falla."""
        population = list(existing_population)  # Copiar la población existente
        
        # 1. Intentar enfoque constructivo si está habilitado y no hay soluciones
        if use_constructive_approach and len(population) == 0:
            logger.info("Intentando enfoque constructivo para generar la población inicial")
            constructive_solution = self._generate_constructive_solution(employees, shifts)
            validation_result = validator.validate(constructive_solution, employees, shifts)
            
            if validation_result.is_valid:
                self._calculate_solution_cost(constructive_solution, employees, shifts)
                population.append(constructive_solution)
                
                # Generar más soluciones como variaciones de esta válida
                for _ in range(min(5, target_size - len(population))):
                    variant = constructive_solution.clone()
                    self._mutate(variant, employees, shifts, 0.1)  # Mutación ligera
                    if validator.validate(variant, employees, shifts).is_valid:
                        self._calculate_solution_cost(variant, employees, shifts)
                        population.append(variant)
        
        # 2. Si aún necesitamos más soluciones, intentar con restricciones relajadas
        if len(population) < target_size:
            logger.info("Intentando generar población con restricciones relajadas temporalmente")
            relaxed_count = target_size - len(population)
            relaxed_solutions = self._generate_relaxed_solutions(
                employees, shifts, validator, relaxed_count, relaxation_factor
            )
            population.extend(relaxed_solutions)
            logger.info(f"Añadidas {len(relaxed_solutions)} soluciones con restricciones relajadas")
        
        return population
    
    def _generate_candidate_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Genera una solución candidata intentando cumplir con las restricciones desde el inicio."""
        solution = Solution()
        
        # Tracking de horas y días asignados por empleado
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
        # Tracking de empleados por turno y día
        day_shift_employees = collections.defaultdict(set)
        
        # Dar preferencia a los turnos de mayor prioridad y a los que requieren más empleados
        sorted_shifts = sorted(shifts, key=lambda s: (s.priority, s.required_employees), reverse=True)
        
        for shift in sorted_shifts:
            day_shift_key = (shift.day, shift.name)
            
            # Identificar empleados calificados que aún no han sido asignados a este turno
            qualified_employees = [
                e for e in employees
                if e.id not in day_shift_employees[day_shift_key] and
                e.is_available(shift.day, shift.name) and
                shift.required_skills.issubset(e.skills) and
                employee_hours[e.id] + shift.duration_hours <= e.max_hours_per_week
            ]
            
            # Ordenar por costo y preferencia
            qualified_employees.sort(
                key=lambda e: (employee_hours[e.id], 
                              -e.get_preference_score(shift.day, shift.name),
                              e.hourly_cost)
            )
            
            # Asignar hasta el número requerido de empleados o los disponibles
            num_to_assign = min(shift.required_employees, len(qualified_employees))
            
            for i in range(num_to_assign):
                employee = qualified_employees[i]
                
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
    
    def _generate_constructive_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Genera una solución usando un enfoque más determinístico y constructivo."""
        solution = Solution()
        
        # Crear mapeos para acceso rápido
        employee_dict = {e.id: e for e in employees}
        
        # Rastrear asignaciones
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        day_shift_employees = collections.defaultdict(set)
        
        # Ordenar turnos por prioridad (descendente) y días
        days_order = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO', 'DOMINGO']
        sorted_shifts = sorted(shifts, 
                               key=lambda s: (
                                   -s.priority,
                                   days_order.index(s.day.name) if s.day.name in days_order else 999,
                                   s.name.name  # ShiftType como MAÑANA, TARDE, NOCHE
                               ))
        
        # Asociar cada empleado con sus turnos más adecuados
        employee_best_shifts = self._map_employee_best_shifts(employees, sorted_shifts)
        
        # Para cada turno, asignar empleados priorizando mejores calificaciones
        for shift in sorted_shifts:
            day_shift_key = (shift.day, shift.name)
            required = shift.required_employees - len(day_shift_employees[day_shift_key])
            
            if required <= 0:
                continue  # Este turno ya está cubierto
            
            # Encontrar empleados disponibles y calificados para este turno
            candidates = self._find_candidates_for_shift(
                shift, employees, employee_hours, employee_days, day_shift_employees
            )
            
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
    
    def _map_employee_best_shifts(self, employees: List[Employee], shifts: List[Shift]) -> Dict[str, List[Tuple[Shift, float]]]:
        """Mapea cada empleado a sus mejores turnos por calificación y preferencia."""
        employee_best_shifts = collections.defaultdict(list)
        
        for employee in employees:
            for shift in shifts:
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
        
        return employee_best_shifts
    
    def _find_candidates_for_shift(
        self,
        shift: Shift,
        employees: List[Employee],
        employee_hours: Dict[str, float],
        employee_days: Dict[str, set],
        day_shift_employees: Dict[Tuple, set]
    ) -> List[Tuple[Employee, float, float]]:
        """Encuentra empleados candidatos para un turno específico."""
        day_shift_key = (shift.day, shift.name)
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
        
        return candidates
    
    def _generate_relaxed_solutions(
        self, 
        employees: List[Employee], 
        shifts: List[Shift], 
        validator: SolutionValidator,
        count: int,
        relaxation_factor: float
    ) -> List[Solution]:
        """Genera soluciones con restricciones relajadas temporalmente."""
        solutions = []
        
        # Generar una solución constructiva relajada para cada requerimiento
        for _ in range(count):
            solution = Solution()
            employees_per_shift = collections.defaultdict(list)  # Para rastrear asignaciones
            
            # Dar prioridad a la cobertura básica, relajando algunas restricciones
            for shift in shifts:
                # Encontrar empleados que al menos están disponibles en ese día
                available_employees = [
                    e for e in employees
                    if e.is_available(shift.day, shift.name) or random.random() < relaxation_factor
                ]
                
                # Si no hay suficientes empleados incluso con relajación, seguir intentando
                if len(available_employees) < shift.required_employees:
                    available_employees = list(employees)
                
                # Seleccionar empleados aleatoriamente
                if available_employees:
                    # Número requerido o disponible, lo que sea menor
                    num_to_assign = min(shift.required_employees, len(available_employees))
                    selected = random.sample(available_employees, num_to_assign)
                    
                    for employee in selected:
                        # Crear asignación
                        assignment = Assignment(
                            employee_id=employee.id,
                            shift_id=shift.id,
                            cost=employee.hourly_cost * shift.duration_hours
                        )
                        solution.add_assignment(assignment)
                        employees_per_shift[shift.id].append(employee.id)
            
            solutions.append(solution)
        
        return solutions
    
    def _calculate_fitness(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> float:
        """Calcula la puntuación de fitness para una solución considerando solo el costo (sin penalizaciones)."""
        # Calcular el costo total de la solución
        self._calculate_solution_cost(solution, employees, shifts)
        
        # El fitness es inversamente proporcional al costo
        # Usamos esta fórmula para evitar valores extremos
        fitness_score = 1000.0 / (1.0 + solution.total_cost / 100.0)
        
        # Añadir bonificaciones por preferencias de empleados
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
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
        
        # Fitness final incluye las preferencias
        final_fitness = fitness_score + preference_bonus
        
        # Guardar el fitness en la solución para referencias futuras
        solution.fitness_score = max(0.1, final_fitness)
        
        return solution.fitness_score
    
    def _calculate_solution_cost(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> float:
        """Calcula el costo real de la solución basado en las asignaciones."""
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
        
        return total_cost
    
    def _tournament_selection(self, population: List[Solution], fitness_scores: List[float], tournament_size: int) -> Solution:
        """Selecciona una solución utilizando selección por torneo."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index].clone()
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Aplica cruce para crear dos descendientes, utilizando operaciones de conjunto más eficientes."""
        # Convertir asignaciones a conjuntos para manipulación más eficiente
        assignments1 = set(parent1.assignments)
        assignments2 = set(parent2.assignments)
        
        # Obtener elementos comunes y únicos
        common = assignments1 & assignments2
        unique_to_parent1 = assignments1 - common
        unique_to_parent2 = assignments2 - common
        
        # Si no hay elementos únicos en algún padre, simplemente clonar
        if not unique_to_parent1 or not unique_to_parent2:
            return parent1.clone(), parent2.clone()
        
        # Crear descendientes usando punto de cruce único con las asignaciones únicas
        unique_to_parent1_list = list(unique_to_parent1)
        unique_to_parent2_list = list(unique_to_parent2)
        
        # Determinar punto de cruce
        crossover_point = random.randint(1, min(len(unique_to_parent1_list), len(unique_to_parent2_list)))
        
        # Crear nuevos conjuntos de asignaciones usando comprehensiones más eficientes
        offspring1_assignments = list(common) + unique_to_parent1_list[:crossover_point] + unique_to_parent2_list[crossover_point:]
        offspring2_assignments = list(common) + unique_to_parent2_list[:crossover_point] + unique_to_parent1_list[crossover_point:]
        
        # Crear soluciones descendientes
        offspring1 = Solution()
        offspring1.assignments = offspring1_assignments
        
        offspring2 = Solution()
        offspring2.assignments = offspring2_assignments
        
        return offspring1, offspring2
    
    def _mutate_and_repair(self, 
                          solution: Solution, 
                          employees: List[Employee], 
                          shifts: List[Shift], 
                          mutation_rate: float,
                          validator: SolutionValidator,
                          max_repair_attempts: int) -> Optional[Solution]:
        """Aplica mutación a una solución y repara si es necesario para mantener factibilidad."""
        if not solution or not solution.assignments:
            return None
            
        # Clonar para no modificar la original
        mutated = solution.clone()
        original_assignments = list(mutated.assignments)
        
        # Aplicar mutación
        self._mutate(mutated, employees, shifts, mutation_rate)
        
        # Validar resultado
        validation_result = validator.validate(mutated, employees, shifts)
        
        # Si es válida, retornar; si no, intentar reparar
        if validation_result.is_valid:
            return mutated
            
        # Intentos de reparación
        for attempt in range(max_repair_attempts):
            # Restaurar a un estado conocido válido (el original antes de la mutación)
            repaired = Solution()
            repaired.assignments = list(original_assignments)
            
            # Intentar una mutación más conservadora con tasa reducida
            reduced_rate = mutation_rate * (0.9 ** attempt)  # Reducir gradualmente
            self._mutate(repaired, employees, shifts, reduced_rate)
            
            # Validar la solución reparada
            validation_result = validator.validate(repaired, employees, shifts)
            if validation_result.is_valid:
                return repaired
        
        # Si no se pudo reparar, devolver None para indicar fallo
        return None
    
    def _mutate(self, solution: Solution, employees: List[Employee], shifts: List[Shift], mutation_rate: float) -> None:
        """Aplica mutación a una solución con una tasa reducida para preservar la factibilidad."""
        employee_dict = {e.id: e for e in employees}
        shift_dict = {s.id: s for s in shifts}
        
        # Para cada asignación, decidir si mutar
        for i, assignment in enumerate(solution.assignments[:]):
            if random.random() < mutation_rate:
                # Reemplazar esta asignación con una nueva
                shift_id = assignment.shift_id
                if shift_id in shift_dict:
                    shift = shift_dict[shift_id]
                    
                    # Encontrar empleados calificados y disponibles para este turno
                    qualified_employees = [
                        e for e in employees 
                        if e.id != assignment.employee_id and 
                        e.is_available(shift.day, shift.name) and
                        shift.required_skills.issubset(e.skills)
                    ]
                    
                    if qualified_employees:
                        # Eliminar la asignación anterior
                        solution.assignments.remove(assignment)
                        
                        # Crear nueva asignación con un empleado calificado aleatorio
                        new_employee = random.choice(qualified_employees)
                        new_assignment = Assignment(
                            employee_id=new_employee.id,
                            shift_id=shift_id,
                            cost=new_employee.hourly_cost * shift.duration_hours
                        )
                        solution.add_assignment(new_assignment)
        
        # Considerar añadir una nueva asignación para turnos sin cobertura completa
        if random.random() < mutation_rate:
            # Encontrar turnos con cobertura insuficiente
            undercover_shifts = []
            for shift in shifts:
                assigned = len(solution.get_shift_employees(shift.id))
                if assigned < shift.required_employees:
                    # Este turno necesita más empleados
                    undercover_shifts.append((shift, shift.required_employees - assigned))
            
            if undercover_shifts:
                # Seleccionar un turno aleatorio para mejorar
                selected_shift, needed = random.choice(undercover_shifts)
                
                # Encontrar empleados disponibles que no estén ya asignados
                current_assigned_ids = set(solution.get_shift_employees(selected_shift.id))
                available_employees = [
                    e for e in employees 
                    if e.id not in current_assigned_ids and 
                    e.is_available(selected_shift.day, selected_shift.name) and
                    selected_shift.required_skills.issubset(e.skills)
                ]
                
                if available_employees:
                    # Asignar hasta el número necesario de empleados
                    to_assign = min(needed, len(available_employees))
                    for i in range(to_assign):
                        emp = random.choice(available_employees)
                        available_employees.remove(emp)  # No seleccionar dos veces
                        
                        new_assignment = Assignment(
                            employee_id=emp.id,
                            shift_id=selected_shift.id,
                            cost=emp.hourly_cost * selected_shift.duration_hours
                        )
                        solution.add_assignment(new_assignment)