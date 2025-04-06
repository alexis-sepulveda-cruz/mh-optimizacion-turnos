import random
import logging
from typing import List, Dict, Any, Tuple
import time

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
            "population_size": 500,
            "generations": 1000,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_count": 5,
            "tournament_size": 3,
            "max_repair_attempts": 100,  # Número máximo de intentos para reparar soluciones inválidas
            "validation_timeout": 100,  # Tiempo máximo (segundos) para generar una solución válida
            "metrics": {
                "enabled": True,  # Habilitar recolección de métricas
                "track_evaluations": True,  # Contar número de evaluaciones de función objetivo
                "track_validation_time": True  # Medir tiempo de validación
            }
        }
    
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimiza la asignación de turnos usando Algoritmo Genético con restricciones duras."""
        if config is None:
            config = self.get_default_config()
        
        # Extraer parámetros de configuración
        population_size = config.get("population_size", 500)
        generations = config.get("generations", 1000)
        mutation_rate = config.get("mutation_rate", 0.1)
        crossover_rate = config.get("crossover_rate", 0.8)
        elitism_count = config.get("elitism_count", 5)
        tournament_size = config.get("tournament_size", 3)
        
        # Parámetros de restricciones duras
        max_repair_attempts = config.get("max_repair_attempts", 100)
        validation_timeout = config.get("validation_timeout", 100)
        
        # Inicialización de métricas
        metrics = {
            "objective_evaluations": 0,
            "validation_time": 0,
            "repair_attempts": 0,
            "repair_success_rate": 0
        }
        
        # Crear validador
        validator = SolutionValidator()
        
        logger.info(f"Iniciando optimización con algoritmo genético con {population_size} individuos para {generations} generaciones")
        logger.info(f"Usando enfoque de restricciones duras (solo soluciones factibles)")
        
        # Inicializar población con soluciones válidas
        start_time = time.time()
        population = self._initialize_valid_population(
            employees, shifts, population_size, validator, max_repair_attempts, validation_timeout
        )
        initialization_time = time.time() - start_time
        logger.info(f"Población inicial de {len(population)} soluciones válidas generada en {initialization_time:.2f} segundos")
        
        if not population:
            raise ValueError("No se pudo generar una población inicial con soluciones válidas. Revise las restricciones.")
        
        # Evaluar fitness para la población inicial (ahora solo considera costo, no violaciones)
        fitness_scores = []
        for solution in population:
            fitness = self._calculate_fitness(solution, employees, shifts)
            fitness_scores.append(fitness)
            metrics["objective_evaluations"] += 1
        
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
            
            # Añadir las soluciones elite a la nueva población
            population = new_population + elite_solutions
            
            # Recalcular puntuaciones de fitness
            fitness_scores = []
            for solution in population:
                fitness = self._calculate_fitness(solution, employees, shifts)
                fitness_scores.append(fitness)
                metrics["objective_evaluations"] += 1
            
            # Registrar progreso
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            gen_time = time.time() - start_gen_time
            
            if generation % 10 == 0 or generation == generations - 1:
                logger.info(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, "
                          f"Fitness promedio = {avg_fitness:.4f}, "
                          f"Tiempo: {gen_time:.2f}s, "
                          f"Tasa de reparación: {metrics['repair_success_rate']:.2f}")
        
        # Devolver la mejor solución
        best_idx = fitness_scores.index(max(fitness_scores))
        best_solution = population[best_idx]
        
        # Calcular el costo real antes de devolver
        self._calculate_solution_cost(best_solution, employees, shifts)
        
        logger.info(f"Optimización completa. Fitness de la mejor solución: {fitness_scores[best_idx]}")
        logger.info(f"Métricas: {metrics['objective_evaluations']} evaluaciones de función objetivo, "
                  f"Tasa de reparación: {metrics['repair_success_rate']:.2f}")
        
        return best_solution
    
    def _initialize_valid_population(
        self, 
        employees: List[Employee], 
        shifts: List[Shift], 
        population_size: int,
        validator: SolutionValidator,
        max_attempts: int,
        timeout: float
    ) -> List[Solution]:
        """Inicializa una población de soluciones válidas."""
        population = []
        attempts = 0
        start_time = time.time()
        
        while len(population) < population_size and attempts < max_attempts * population_size:
            if time.time() - start_time > timeout * population_size:
                logger.warning(f"Tiempo de inicialización excedido. Se generaron {len(population)} soluciones válidas "
                             f"de {population_size} requeridas.")
                break
                
            # Generar una solución candidata
            solution = self._generate_candidate_solution(employees, shifts)
            attempts += 1
            
            # Validar la solución
            validation_result = validator.validate(solution, employees, shifts)
            
            if validation_result.is_valid:
                # Calcular costo real
                self._calculate_solution_cost(solution, employees, shifts)
                population.append(solution)
                
            # Informar progreso
            if attempts % 100 == 0:
                logger.info(f"Generación de población inicial: {len(population)}/{population_size} soluciones válidas "
                          f"en {attempts} intentos ({(time.time() - start_time):.2f}s)")
        
        if not population:
            # Si no pudimos generar ninguna solución válida, algo está muy mal con las restricciones
            logger.error("No se pudo generar ninguna solución válida. Las restricciones pueden ser demasiado estrictas.")
        
        return population
    
    def _generate_candidate_solution(self, employees: List[Employee], shifts: List[Shift]) -> Solution:
        """Genera una solución candidata intentando cumplir con las restricciones desde el inicio."""
        solution = Solution()
        
        # Dar preferencia a los turnos de mayor prioridad
        sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
        
        # Para cada turno, asignar empleados que:
        # 1. Estén disponibles para ese turno
        # 2. Tengan las habilidades requeridas
        # 3. No excedan sus horas máximas
        # 4. No excedan sus días consecutivos
        
        # Tracking de horas y días asignados por empleado
        employee_hours = {e.id: 0.0 for e in employees}
        employee_days = {e.id: set() for e in employees}
        
        for shift in sorted_shifts:
            # Identificar empleados que cumplen con todas las restricciones para este turno
            qualified_employees = []
            
            for employee in employees:
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
                    # Ya alcanzó el máximo de días, pero podría ser compatible si este día está en secuencia
                    # Simplificación: permitimos siempre que no exceda el número total de días
                    if len(employee_days[employee.id]) < 7:  # Asumiendo una semana de 7 días
                        pass
                    else:
                        continue
                
                # Este empleado cumple con todas las condiciones
                qualified_employees.append(employee)
            
            # Si no hay suficientes empleados calificados, podríamos tener problemas
            if len(qualified_employees) < shift.required_employees:
                # En un enfoque más avanzado, podríamos aplicar una técnica de reparación aquí
                # Por ahora, simplemente asignamos los que haya disponibles
                pass
            
            # Asignar empleados aleatoriamente entre los calificados
            # (hasta el número requerido o los disponibles)
            num_to_assign = min(shift.required_employees, len(qualified_employees))
            if num_to_assign > 0:
                selected_employees = random.sample(qualified_employees, num_to_assign)
                
                for employee in selected_employees:
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
        
        return solution
    
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
        """Aplica cruce para crear dos descendientes, manteniendo la factibilidad."""
        # Convertir asignaciones a conjuntos para una manipulación más sencilla
        assignments1 = set(parent1.assignments)
        assignments2 = set(parent2.assignments)
        
        # Obtener elementos únicos de cada padre
        unique_to_parent1 = assignments1 - assignments2
        unique_to_parent2 = assignments2 - assignments1
        
        # Obtener elementos comunes
        common = assignments1.intersection(assignments2)
        
        # Crear descendientes mezclando elementos únicos
        if unique_to_parent1 and unique_to_parent2:
            crossover_point = random.randint(1, min(len(unique_to_parent1), len(unique_to_parent2)))
            
            # Convertir a listas para indexación
            unique_to_parent1_list = list(unique_to_parent1)
            unique_to_parent2_list = list(unique_to_parent2)
            
            # Crear nuevos conjuntos de asignaciones
            offspring1_assignments = common.union(
                set(unique_to_parent1_list[:crossover_point]),
                set(unique_to_parent2_list[crossover_point:])
            )
            
            offspring2_assignments = common.union(
                set(unique_to_parent2_list[:crossover_point]),
                set(unique_to_parent1_list[crossover_point:])
            )
        else:
            # Si un padre no tiene elementos únicos, simplemente clonar los padres
            return parent1.clone(), parent2.clone()
        
        # Crear soluciones descendientes
        offspring1 = Solution()
        offspring1.assignments = list(offspring1_assignments)
        
        offspring2 = Solution()
        offspring2.assignments = list(offspring2_assignments)
        
        return offspring1, offspring2
    
    def _mutate_and_repair(self, 
                          solution: Solution, 
                          employees: List[Employee], 
                          shifts: List[Shift], 
                          mutation_rate: float,
                          validator: SolutionValidator,
                          max_repair_attempts: int) -> Solution:
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