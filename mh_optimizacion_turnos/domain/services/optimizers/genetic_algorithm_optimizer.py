import random
import logging
from typing import List, Dict, Any, Tuple

from ..optimizer_strategy import OptimizerStrategy
from ...models.solution import Solution
from ...models.employee import Employee
from ...models.shift import Shift
from ...models.assignment import Assignment


logger = logging.getLogger(__name__)


class GeneticAlgorithmOptimizer(OptimizerStrategy):
    """Implementación de Algoritmo Genético para la optimización de turnos."""
    
    def get_name(self) -> str:
        return "Genetic Algorithm Optimizer"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_count": 5,
            "tournament_size": 3
        }
    
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimiza la asignación de turnos usando Algoritmo Genético."""
        if config is None:
            config = self.get_default_config()
        
        # Extraer parámetros de configuración
        population_size = config.get("population_size", 50)
        generations = config.get("generations", 100)
        mutation_rate = config.get("mutation_rate", 0.1)
        crossover_rate = config.get("crossover_rate", 0.8)
        elitism_count = config.get("elitism_count", 5)
        tournament_size = config.get("tournament_size", 3)
        
        logger.info(f"Iniciando optimización con algoritmo genético con {population_size} individuos para {generations} generaciones")
        
        # Inicializar población
        population = self._initialize_population(employees, shifts, population_size)
        
        # Evaluar fitness para la población inicial
        fitness_scores = [self._calculate_fitness(solution, employees, shifts) for solution in population]
        
        # Bucle principal del algoritmo genético
        for generation in range(generations):
            # Aplicar elitismo - mantener las mejores soluciones
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elitism_count]
            elite_solutions = [population[i].clone() for i in elite_indices]
            
            new_population = []
            
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
                
                # Mutación
                self._mutate(offspring1, employees, shifts, mutation_rate)
                self._mutate(offspring2, employees, shifts, mutation_rate)
                
                new_population.append(offspring1)
                if len(new_population) < population_size - elitism_count:
                    new_population.append(offspring2)
            
            # Añadir las soluciones elite
            population = new_population + elite_solutions
            
            # Recalcular puntuaciones de fitness
            fitness_scores = [self._calculate_fitness(solution, employees, shifts) for solution in population]
            
            # Registrar progreso
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            logger.info(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, Fitness promedio = {avg_fitness:.4f}")
        
        # Devolver la mejor solución
        best_idx = fitness_scores.index(max(fitness_scores))
        best_solution = population[best_idx]
        
        logger.info(f"Optimización completa. Fitness de la mejor solución: {fitness_scores[best_idx]}")
        
        return best_solution
    
    def _initialize_population(self, employees: List[Employee], shifts: List[Shift], population_size: int) -> List[Solution]:
        """Inicializa una población aleatoria de soluciones."""
        population = []
        
        for _ in range(population_size):
            solution = Solution()
            
            # Dar preferencia a los turnos de mayor prioridad
            sorted_shifts = sorted(shifts, key=lambda s: s.priority, reverse=True)
            
            for shift in sorted_shifts:
                # Encontrar empleados calificados para este turno
                qualified_employees = [
                    e for e in employees 
                    if e.is_available(shift.day, shift.name) and shift.required_skills.issubset(e.skills)
                ]
                
                # Si no hay empleados calificados, elegir empleados aleatorios
                if not qualified_employees and employees:
                    qualified_employees = random.sample(employees, min(5, len(employees)))
                
                # Intentar asignar el número requerido de empleados
                if qualified_employees:
                    assigned_employees = random.sample(
                        qualified_employees, 
                        min(shift.required_employees, len(qualified_employees))
                    )
                    
                    for employee in assigned_employees:
                        assignment = Assignment(
                            employee_id=employee.id,
                            shift_id=shift.id,
                            cost=employee.hourly_cost * shift.duration_hours
                        )
                        solution.add_assignment(assignment)
            
            solution.calculate_total_cost()
            population.append(solution)
            
        return population
    
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
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
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
    
    def _tournament_selection(self, population: List[Solution], fitness_scores: List[float], tournament_size: int) -> Solution:
        """Selecciona una solución utilizando selección por torneo."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index].clone()
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Aplica cruce para crear dos descendientes."""
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
        offspring1.calculate_total_cost()
        
        offspring2 = Solution()
        offspring2.assignments = list(offspring2_assignments)
        offspring2.calculate_total_cost()
        
        return offspring1, offspring2
    
    def _mutate(self, solution: Solution, employees: List[Employee], shifts: List[Shift], mutation_rate: float) -> None:
        """Aplica mutación a una solución."""
        # Para cada asignación, decidir si mutar
        for i, assignment in enumerate(solution.assignments[:]):  # Copiar la lista para evitar problemas de modificación durante la iteración
            if random.random() < mutation_rate:
                # Reemplazar esta asignación con una nueva
                
                # Primero, eliminar la asignación anterior
                solution.assignments.remove(assignment)
                
                # Luego crear una nueva asignación con un empleado diferente
                shift_id = assignment.shift_id
                if shifts:
                    # Encontrar el objeto shift
                    shift = next((s for s in shifts if s.id == shift_id), None)
                    
                    if shift:
                        # Seleccionar un empleado diferente
                        available_employees = [
                            e for e in employees 
                            if e.id != assignment.employee_id and 
                            e.is_available(shift.day, shift.name) and
                            shift.required_skills.issubset(e.skills)
                        ]
                        
                        if available_employees:
                            new_employee = random.choice(available_employees)
                            new_assignment = Assignment(
                                employee_id=new_employee.id,
                                shift_id=shift_id,
                                cost=new_employee.hourly_cost * shift.duration_hours
                            )
                            solution.add_assignment(new_assignment)
        
        # Mutación adicional: añadir una asignación completamente nueva con cierta probabilidad
        if random.random() < mutation_rate and shifts:
            random_shift = random.choice(shifts)
            
            # Comprobar si este turno necesita más empleados
            assigned_employees = len(solution.get_shift_employees(random_shift.id))
            
            if assigned_employees < random_shift.required_employees:
                # Encontrar empleados disponibles que no estén ya asignados a este turno
                current_assigned_ids = set(solution.get_shift_employees(random_shift.id))
                available_employees = [
                    e for e in employees 
                    if e.id not in current_assigned_ids and 
                    e.is_available(random_shift.day, random_shift.name) and
                    random_shift.required_skills.issubset(e.skills)
                ]
                
                if available_employees:
                    new_employee = random.choice(available_employees)
                    new_assignment = Assignment(
                        employee_id=new_employee.id,
                        shift_id=random_shift.id,
                        cost=new_employee.hourly_cost * random_shift.duration_hours
                    )
                    solution.add_assignment(new_assignment)
        
        # Recalcular costo después de las mutaciones
        solution.calculate_total_cost()