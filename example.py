#!/usr/bin/env python
"""
Ejemplo de uso del Sistema de Asignación Óptima de Turnos de Trabajo.

Este script demuestra la implementación completa del sistema utilizando
diferentes algoritmos metaheurísticos (Algoritmo Genético, Búsqueda Tabú y GRASP)
para optimizar la asignación de turnos a empleados.
"""

import logging
import sys
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import os
import statistics

from mh_optimizacion_turnos.domain.value_objects.day import Day
from mh_optimizacion_turnos.domain.value_objects.shift_type import ShiftType
from mh_optimizacion_turnos.domain.value_objects.skill import Skill
from mh_optimizacion_turnos.domain.value_objects.algorithm_type import AlgorithmType
from mh_optimizacion_turnos.domain.value_objects.export_format import ExportFormat
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.services.shift_optimizer_service import ShiftOptimizerService

from mh_optimizacion_turnos.infrastructure.repositories.in_memory_employee_repository import InMemoryEmployeeRepository
from mh_optimizacion_turnos.infrastructure.repositories.in_memory_shift_repository import InMemoryShiftRepository

from mh_optimizacion_turnos.infrastructure.adapters.input.shift_assignment_service_adapter import ShiftAssignmentServiceAdapter
from mh_optimizacion_turnos.infrastructure.adapters.output.schedule_export_adapter import ScheduleExportAdapter

# Constantes para la configuración de datos de prueba
NUM_EMPLOYEES = 20          # Aumentado de 10 a 20 para tener más empleados disponibles
MIN_EMPLOYEE_ID = 1         # ID inicial para empleados
MAX_HOURS_PER_WEEK = 40     # Máximo de horas por semana por empleado
MAX_CONSECUTIVE_DAYS = 5    # Máximo de días consecutivos por empleado
MIN_HOURLY_COST = 10.0      # Costo mínimo por hora
MAX_HOURLY_COST = 20.0      # Costo máximo por hora
EMPLOYEES_PER_SHIFT = 2     # Reducido de 3 a 2 empleados requeridos por turno
MIN_EMPLOYEE_SKILLS = 2     # Aumentado de 1 a 2 habilidades mínimas por empleado
MAX_EMPLOYEE_SKILLS = 4     # Aumentado de 3 a 4 habilidades máximas por empleado

# Constantes para turnos
MORNING_SHIFT_START = 8     # Hora de inicio turno mañana
MORNING_SHIFT_END = 16      # Hora de fin turno mañana
AFTERNOON_SHIFT_START = 16  # Hora de inicio turno tarde
AFTERNOON_SHIFT_END = 0     # Hora de fin turno tarde (medianoche)
NIGHT_SHIFT_START = 0       # Hora de inicio turno noche
NIGHT_SHIFT_END = 8         # Hora de fin turno noche
HIGH_PRIORITY = 2           # Prioridad alta para turnos
NORMAL_PRIORITY = 1         # Prioridad normal para turnos

# Constantes para preferencias
MIN_REGULAR_PREFERENCE = 1  # Preferencia mínima para turnos regulares
MAX_REGULAR_PREFERENCE = 4  # Preferencia máxima para turnos regulares
MIN_MORNING_PREFERENCE = 3  # Preferencia mínima para turno mañana
MAX_MORNING_PREFERENCE = 6  # Preferencia máxima para turno mañana

# Constantes para algoritmos
ALGORITHM_RUNS = 5          # Número de ejecuciones para medir consistencia
GENETIC_POPULATION_SIZE = 30  # Tamaño de población para algoritmo genético
GENETIC_GENERATIONS = 50      # Número de generaciones para algoritmo genético
TABU_MAX_ITERATIONS = 50      # Iteraciones para búsqueda tabú
TABU_TENURE = 10              # Tenencia tabú para búsqueda tabú
GRASP_MAX_ITERATIONS = 30     # Iteraciones para GRASP
GRASP_ALPHA = 0.3             # Factor alpha para GRASP

# Configuración de registro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./assets/logs/mh-optimizacion-turnos.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def ask_continue_iteration():
    """Pregunta al usuario si desea continuar con la siguiente iteración."""
    while True:
        response = input("¿Desea continuar con la iteración? (s/n): ").strip().lower()
        if response in ['s', 'si', 'sí', 'y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Respuesta no válida. Por favor, ingrese 's' para sí o 'n' para no.")


def setup_test_data():
    """Configura datos de ejemplo para pruebas."""
    # Crear repositorios
    employee_repo = InMemoryEmployeeRepository()
    shift_repo = InMemoryShiftRepository()
    
    # Crear turnos
    days = [Day.LUNES, Day.MARTES, Day.MIERCOLES, Day.JUEVES, Day.VIERNES, Day.SABADO, Day.DOMINGO]
    shift_types = [ShiftType.MAÑANA, ShiftType.TARDE, ShiftType.NOCHE]
    skills = [Skill.ATENCION_AL_CLIENTE, Skill.MANUFACTURA, Skill.CAJA, Skill.INVENTARIO, Skill.LIMPIEZA, Skill.SUPERVISOR]
    
    # Horas para cada tipo de turno
    shift_hours = {
        ShiftType.MAÑANA: (datetime(2025, 1, 1, MORNING_SHIFT_START, 0), 
                         datetime(2025, 1, 1, MORNING_SHIFT_END, 0)),
        ShiftType.TARDE: (datetime(2025, 1, 1, AFTERNOON_SHIFT_START, 0), 
                        datetime(2025, 1, 1, AFTERNOON_SHIFT_END, 0)),
        ShiftType.NOCHE: (datetime(2025, 1, 1, NIGHT_SHIFT_START, 0), 
                        datetime(2025, 1, 1, NIGHT_SHIFT_END, 0))
    }
    
    # Crear turnos para cada día y tipo
    for day in days:
        for shift_type in shift_types:
            start_time, end_time = shift_hours[shift_type]
            required_skills = set()
            
            # Diferentes habilidades requeridas según el turno
            if shift_type == ShiftType.MAÑANA:
                required_skills = {skills[0], skills[1],skills[2]}  # Atención al cliente, Caja
            elif shift_type == ShiftType.TARDE:
                required_skills = {skills[0], skills[3],skills[1]}  # Atención al cliente, Inventario
            elif shift_type == ShiftType.NOCHE:
                required_skills = {skills[4], skills[5]}  # Limpieza, Supervisor
            
            shift = Shift(
                name=shift_type,
                day=day,
                start_time=start_time,
                end_time=end_time,
                required_employees=EMPLOYEES_PER_SHIFT,
                required_skills=required_skills,
                priority=HIGH_PRIORITY if shift_type == ShiftType.MAÑANA else NORMAL_PRIORITY
            )
            shift_repo.save(shift)
    
    # Crear empleados
    for i in range(MIN_EMPLOYEE_ID, MIN_EMPLOYEE_ID + NUM_EMPLOYEES):
        # Seleccionar aleatoriamente entre MIN_EMPLOYEE_SKILLS y MAX_EMPLOYEE_SKILLS habilidades
        random_skill_count = np.random.randint(MIN_EMPLOYEE_SKILLS, MAX_EMPLOYEE_SKILLS + 1)
        # Convertir skills a una lista para poder seleccionar aleatoriamente
        skills_list = list(skills)
        # Seleccionar índices aleatorios
        random_indices = np.random.choice(
            range(len(skills_list)), 
            size=random_skill_count, 
            replace=False
        )
        # Crear conjunto de habilidades aleatorias
        random_skills = {skills_list[i] for i in random_indices}
        
        employee = Employee(
            name=f"Empleado {i}",
            max_hours_per_week=MAX_HOURS_PER_WEEK,
            max_consecutive_days=MAX_CONSECUTIVE_DAYS,
            skills=random_skills,
            hourly_cost=np.random.uniform(MIN_HOURLY_COST, MAX_HOURLY_COST)
        )
        
        # Definir disponibilidad aleatoria utilizando los enums
        availability = {}
        for day in days:
            # Seleccionar aleatoriamente entre 1 y el número total de tipos de turnos
            available_shifts_count = np.random.randint(1, len(shift_types) + 1)
            # Convertir a lista para facilitar la selección aleatoria
            shift_types_list = list(shift_types)
            # Seleccionar turnos aleatorios
            random_indices = np.random.choice(
                range(len(shift_types_list)), 
                size=available_shifts_count,
                replace=False
            )
            available_shifts = [shift_types_list[i] for i in random_indices]
            availability[day] = available_shifts
        
        employee.availability = availability
        
        # Definir preferencias aleatorias usando directamente los enums
        preferences = {}
        for day in days:
            # Inicializar diccionario para este día si no existe
            if day not in preferences:
                preferences[day] = {}
                
            for shift_type in shift_types:
                # Verificar si este turno está en la disponibilidad del empleado para este día
                if day in employee.availability and shift_type in employee.availability[day]:
                    # Mayor probabilidad de preferir mañana
                    if shift_type == ShiftType.MAÑANA:
                        preference = np.random.randint(MIN_MORNING_PREFERENCE, MAX_MORNING_PREFERENCE)
                    else:
                        preference = np.random.randint(MIN_REGULAR_PREFERENCE, MAX_REGULAR_PREFERENCE)
                    # Guardar preferencia usando directamente los enums
                    preferences[day][shift_type] = preference
        
        employee.preferences = preferences
        employee_repo.save(employee)
    
    return employee_repo, shift_repo


def run_algorithm(service, algorithm, config):
    """Ejecuta un algoritmo y captura métricas de rendimiento."""
    metrics = {
        "algorithm": algorithm.to_string(),
        "start_time": time.time(),
        "solution": None,
        "execution_time": None,
        "cost": None,
        "fitness": None,
        "violations": None,  # Siempre será 0 con restricciones duras
        "assignments_count": None,
        "coverage_percentage": None
    }
    
    # Establecer el algoritmo
    service.set_algorithm(algorithm)
    
    # Ejecutar el algoritmo
    solution = service.generate_schedule(algorithm_config=config)
    
    # Capturar métricas
    metrics["execution_time"] = time.time() - metrics["start_time"]
    metrics["solution"] = solution
    metrics["cost"] = solution.total_cost
    metrics["fitness"] = solution.fitness_score
    metrics["violations"] = solution.constraint_violations
    metrics["assignments_count"] = len(solution.assignments)
    
    # Calcular cobertura
    shift_repo = service.shift_optimizer_service.shift_repository
    shifts = shift_repo.get_all()
    total_required = sum(shift.required_employees for shift in shifts)
    
    # Agrupar por turnos para contar asignaciones
    shift_assignment_count = {}
    for assignment in solution.assignments:
        shift_id = assignment.shift_id
        if (shift_id not in shift_assignment_count):
            shift_assignment_count[shift_id] = 0
        shift_assignment_count[shift_id] += 1
    
    # Calcular porcentaje de cobertura
    covered_positions = 0
    for shift in shifts:
        assigned = shift_assignment_count.get(shift.id, 0)
        covered = min(assigned, shift.required_employees)
        covered_positions += covered
    
    metrics["coverage_percentage"] = (covered_positions / total_required * 100) if total_required > 0 else 0
    
    return metrics


def compare_algorithms(service, export_adapter, algorithms=None, runs=ALGORITHM_RUNS, interactive=False):
    """Compara los diferentes algoritmos metaheurísticos."""
    if algorithms is None:
        algorithms = service.get_available_algorithm_enums()
    
    all_results = []
    algorithm_results = {}
    
    for algorithm in algorithms:
        logger.info(f"Evaluando algoritmo: {algorithm.to_string()}")
        
        # Configuración específica para cada algoritmo
        if algorithm == AlgorithmType.GENETIC:
            config = {
                "population_size": GENETIC_POPULATION_SIZE, 
                "generations": GENETIC_GENERATIONS, 
                "interactive": interactive
            }
        elif algorithm == AlgorithmType.TABU:
            config = {
                "max_iterations": TABU_MAX_ITERATIONS, 
                "tabu_tenure": TABU_TENURE, 
                "interactive": interactive
            }
        elif algorithm == AlgorithmType.GRASP:
            config = {
                "max_iterations": GRASP_MAX_ITERATIONS, 
                "alpha": GRASP_ALPHA, 
                "interactive": interactive
            }
        else:
            config = {"interactive": interactive}
        
        # Ejecutar el algoritmo varias veces para medir consistencia
        run_results = []
        for run in range(runs):
            logger.info(f"Ejecutando {algorithm.to_string()} - Corrida {run+1}/{runs}")
            
            if interactive and run > 0:
                # Preguntar si continuar con las siguientes ejecuciones
                print(f"\nPrepárando corrida {run+1}/{runs} del algoritmo {algorithm.to_string()}")
                if not ask_continue_iteration():
                    logger.info(f"Usuario decidió detener las ejecuciones de {algorithm.to_string()} después de {run} corridas.")
                    break
            
            metrics = run_algorithm(service, algorithm, config)
            run_results.append(metrics)
            
            # Mostrar resultados de esta ejecución
            logger.info(f"Algoritmo: {metrics['algorithm']}")
            logger.info(f"Tiempo de ejecución: {metrics['execution_time']:.2f} segundos")
            logger.info(f"Costo total: {metrics['cost']:.2f}")
            logger.info(f"Fitness: {metrics['fitness']:.4f}")
            logger.info(f"Asignaciones: {metrics['assignments_count']}")
            logger.info(f"Cobertura de turnos: {metrics['coverage_percentage']:.1f}%")
            logger.info("-" * 50)
            
            # Exportar solución como texto
            solution_text = export_adapter.export_solution(metrics['solution'], ExportFormat.TEXT)
            logger.info(f"\nSolución #{run+1} con {metrics['algorithm']}:\n{solution_text}\n")
        
        # Guardar todos los resultados
        all_results.extend(run_results)
        
        # Calcular estadísticas para este algoritmo
        if run_results:
            costs = [r["cost"] for r in run_results]
            times = [r["execution_time"] for r in run_results]
            fitnesses = [r["fitness"] for r in run_results]
            coverages = [r["coverage_percentage"] for r in run_results]
            
            # Guardar mejores resultados y estadísticas
            algorithm_results[algorithm.to_string()] = {
                "runs": len(run_results),
                "best_solution": min(run_results, key=lambda r: r["cost"])["solution"],
                "avg_cost": statistics.mean(costs),
                "std_cost": statistics.stdev(costs) if len(costs) > 1 else 0,
                "avg_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "avg_fitness": statistics.mean(fitnesses),
                "std_fitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0,
                "avg_coverage": statistics.mean(coverages),
                "std_coverage": statistics.stdev(coverages) if len(coverages) > 1 else 0
            }
    
    return algorithm_results, all_results


def plot_comparison(results, output_dir="./assets/plots"):
    """Grafica la comparación de resultados entre algoritmos."""
    algorithms = list(results.keys())
    
    # Crear el directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Gráfico de barras comparando tiempo, costo, fitness y cobertura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Datos para los gráficos
    avg_times = [results[algo]["avg_time"] for algo in algorithms]
    std_times = [results[algo]["std_time"] for algo in algorithms]
    
    avg_costs = [results[algo]["avg_cost"] for algo in algorithms]
    std_costs = [results[algo]["std_cost"] for algo in algorithms]
    
    avg_fitness = [results[algo]["avg_fitness"] for algo in algorithms]
    std_fitness = [results[algo]["std_fitness"] for algo in algorithms]
    
    avg_coverage = [results[algo]["avg_coverage"] for algo in algorithms]
    std_coverage = [results[algo]["std_coverage"] for algo in algorithms]
    
    # Gráfico de tiempo de ejecución
    ax1.bar(algorithms, avg_times, yerr=std_times, capsize=5, color='blue', alpha=0.7)
    ax1.set_title('Tiempo de Ejecución')
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico de costo
    ax2.bar(algorithms, avg_costs, yerr=std_costs, capsize=5, color='green', alpha=0.7)
    ax2.set_title('Costo Total')
    ax2.set_ylabel('Costo')
    ax2.tick_params(axis='x', rotation=45)
    
    # Gráfico de fitness
    ax3.bar(algorithms, avg_fitness, yerr=std_fitness, capsize=5, color='purple', alpha=0.7)
    ax3.set_title('Fitness (mayor es mejor)')
    ax3.set_ylabel('Fitness')
    ax3.tick_params(axis='x', rotation=45)
    
    # Gráfico de cobertura
    ax4.bar(algorithms, avg_coverage, yerr=std_coverage, capsize=5, color='red', alpha=0.7)
    ax4.set_title('Cobertura de Turnos')
    ax4.set_ylabel('% Cobertura')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparacion_algoritmos.png')
    logger.info(f"Gráfico de comparación guardado como '{output_dir}/comparacion_algoritmos.png'")
    
    # 2. Gráfico radar para comparar los algoritmos en múltiples dimensiones
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Categorías para el gráfico radar
    categories = ['Tiempo\n(inverso)', 'Costo\n(inverso)', 'Fitness', 'Cobertura', 'Consistencia\n(inverso)']
    N = len(categories)
    
    # Ángulos del gráfico (dividimos el espacio por igual)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el polígono
    
    # Normalizar los datos para que estén entre 0 y 1
    max_time = max(avg_times)
    max_cost = max(avg_costs)
    max_fitness = max(avg_fitness)
    max_coverage = max(avg_coverage)
    max_std = max([results[algo]["std_cost"] / results[algo]["avg_cost"] for algo in algorithms])
    
    # Inicializar gráfico radar
    ax.set_theta_offset(np.pi / 2)  # Rotar para que comience desde arriba
    ax.set_theta_direction(-1)      # Dirección horaria
    
    # Establecer los límites del radar y las etiquetas
    ax.set_ylim(0, 1)
    plt.xticks(angles[:-1], categories)
    
    # Dibujar para cada algoritmo
    for i, algorithm in enumerate(algorithms):
        # Normalizar y convertir los valores (para tiempo, costo y std, menor es mejor, así que invertimos)
        values = [
            1 - (results[algorithm]["avg_time"] / max_time if max_time > 0 else 0),  # Tiempo (inverso)
            1 - (results[algorithm]["avg_cost"] / max_cost if max_cost > 0 else 0),  # Costo (inverso) 
            results[algorithm]["avg_fitness"] / max_fitness if max_fitness > 0 else 0,  # Fitness
            results[algorithm]["avg_coverage"] / 100.0,  # Cobertura (ya está en %)
            1 - ((results[algorithm]["std_cost"] / results[algorithm]["avg_cost"]) / max_std if max_std > 0 else 0)  # Consistencia (inverso)
        ]
        values += values[:1]  # Cerrar el polígono
        
        # Dibujar el polígono y agregar etiqueta
        ax.plot(angles, values, linewidth=2, label=algorithm)
        ax.fill(angles, values, alpha=0.25)
    
    # Añadir leyenda y título
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Comparación multidimensional de algoritmos', size=15, y=1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparacion_radar.png')
    logger.info(f"Gráfico radar guardado como '{output_dir}/comparacion_radar.png'")
    
    # 3. Exportar resultados a CSV
    results_df = pd.DataFrame({
        'Algoritmo': algorithms,
        'Tiempo_Ejecucion_Promedio': avg_times,
        'Tiempo_Ejecucion_Desviacion': std_times,
        'Costo_Promedio': avg_costs,
        'Costo_Desviacion': std_costs,
        'Fitness_Promedio': avg_fitness,
        'Fitness_Desviacion': std_fitness,
        'Cobertura_Promedio': avg_coverage,
        'Cobertura_Desviacion': std_coverage,
        'Corridas': [results[algo]["runs"] for algo in algorithms]
    })
    
    csv_path = f'{output_dir}/resultados_comparacion.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Resultados de comparación exportados a '{csv_path}'")
    
    # Exportar las mejores soluciones de cada algoritmo como JSON
    best_solutions = {}
    for algorithm in algorithms:
        # Serializar la mejor solución
        best_solution = results[algorithm]["best_solution"]
        serialized_assignments = []
        
        for assignment in best_solution.assignments:
            serialized_assignments.append({
                'employee_id': str(assignment.employee_id),
                'shift_id': str(assignment.shift_id),
                'cost': assignment.cost
            })
        
        best_solutions[algorithm] = {
            'cost': best_solution.total_cost,
            'fitness': best_solution.fitness_score,
            'assignments_count': len(best_solution.assignments),
            'assignments': serialized_assignments
        }
    
    json_path = f'{output_dir}/mejores_soluciones.json'
    with open(json_path, 'w') as f:
        json.dump(best_solutions, f, indent=2)
    logger.info(f"Mejores soluciones exportadas a '{json_path}'")
    
    # Mostrar los gráficos
    plt.show()


def main():
    """Función principal de ejemplo."""
    logger.info("Iniciando ejemplo del Sistema de Asignación Óptima de Turnos de Trabajo con restricciones duras")
    
    # Crear el directorio de salida si no existe
    os.makedirs('./assets/plots', exist_ok=True)
    
    # Configurar datos de prueba
    employee_repo, shift_repo = setup_test_data()
    
    # Crear servicios
    solution_validator = SolutionValidator()
    
    shift_optimizer_service = ShiftOptimizerService(
        employee_repository=employee_repo,
        shift_repository=shift_repo,
        solution_validator=solution_validator
    )
    
    # Crear adaptadores
    shift_assignment_service = ShiftAssignmentServiceAdapter(
        shift_optimizer_service=shift_optimizer_service
    )
    
    export_adapter = ScheduleExportAdapter(
        employee_repository=employee_repo,
        shift_repository=shift_repo
    )
    
    # Preguntar al usuario si desea modo interactivo
    interactive_mode = input("¿Desea ejecutar los algoritmos en modo interactivo? (s/n): ").strip().lower() in ['s', 'si', 'sí', 'y', 'yes']
    
    # Preguntar al usuario qué algoritmos ejecutar
    print("Algoritmos disponibles:")
    for i, algo in enumerate(shift_assignment_service.get_available_algorithms()):
        print(f"{i+1}. {algo}")
    
    selected_indices = input("Seleccione los algoritmos a ejecutar (números separados por coma, o Enter para todos): ")
    
    algorithms_to_run = None
    if selected_indices.strip():
        try:
            indices = [int(i.strip()) - 1 for i in selected_indices.split(',')]
            available_algos = shift_assignment_service.get_available_algorithm_enums()
            algorithms_to_run = [available_algos[i] for i in indices if 0 <= i < len(available_algos)]
        except (ValueError, IndexError):
            logger.warning("Selección de algoritmos inválida. Ejecutando todos los algoritmos.")
    
    # Preguntar cuántas ejecuciones por algoritmo
    runs = ALGORITHM_RUNS
    try:
        user_runs = input(f"Número de ejecuciones por algoritmo para evaluar consistencia (Enter para usar {ALGORITHM_RUNS}): ")
        if user_runs.strip():
            runs = max(1, int(user_runs.strip()))
    except ValueError:
        logger.warning(f"Entrada inválida. Usando {ALGORITHM_RUNS} ejecuciones por algoritmo.")
    
    # Ejecutar comparación de algoritmos
    logger.info(f"Iniciando comparación con {runs} ejecuciones por algoritmo")
    results, raw_results = compare_algorithms(
        shift_assignment_service, 
        export_adapter, 
        algorithms=algorithms_to_run, 
        runs=runs, 
        interactive=interactive_mode
    )
    
    # Graficar resultados
    plot_comparison(results)
    
    # Mostrar resultados en la consola
    print("\n" + "="*60)
    print(" RESUMEN DE RESULTADOS ".center(60, "="))
    print("="*60)
    
    for algorithm, stats in results.items():
        print(f"\nAlgoritmo: {algorithm}")
        print(f"  Costo promedio: {stats['avg_cost']:.2f} ± {stats['std_cost']:.2f}")
        print(f"  Fitness promedio: {stats['avg_fitness']:.4f} ± {stats['std_fitness']:.4f}")
        print(f"  Tiempo promedio: {stats['avg_time']:.2f}s ± {stats['std_time']:.2f}s")
        print(f"  Cobertura promedio: {stats['avg_coverage']:.1f}% ± {stats['std_coverage']:.1f}%")
    
    print("\n" + "="*60)
    
    logger.info("Ejemplo completado con éxito.")


if __name__ == "__main__":
    main()