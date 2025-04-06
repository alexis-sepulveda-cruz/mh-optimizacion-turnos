#!/usr/bin/env python
"""
Ejemplo de uso del Sistema de Asignación Óptima de Turnos de Trabajo.

Este script demuestra la implementación completa del sistema utilizando
diferentes algoritmos metaheurísticos (Algoritmo Genético, Búsqueda Tabú y GRASP)
para optimizar la asignación de turnos a empleados.
"""

import logging
import sys
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

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
import os

# Constantes para la configuración de datos de prueba
NUM_EMPLOYEES = 10          # Número total de empleados a crear
MIN_EMPLOYEE_ID = 1         # ID inicial para empleados
MAX_HOURS_PER_WEEK = 40     # Máximo de horas por semana por empleado
MAX_CONSECUTIVE_DAYS = 5    # Máximo de días consecutivos por empleado
MIN_HOURLY_COST = 10.0      # Costo mínimo por hora
MAX_HOURLY_COST = 20.0      # Costo máximo por hora
EMPLOYEES_PER_SHIFT = 3     # Número de empleados requeridos por turno
MIN_EMPLOYEE_SKILLS = 1     # Mínimo de habilidades por empleado
MAX_EMPLOYEE_SKILLS = 3     # Máximo de habilidades por empleado

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
    handlers=[logging.StreamHandler(sys.stdout)]
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
    days = [Day.LUNES, Day.MARTES, Day.MIERCOLES, Day.JUEVES, Day.VIERNES]
    shift_types = [ShiftType.MAÑANA, ShiftType.TARDE, ShiftType.NOCHE]
    skills = [Skill.ATENCION_AL_CLIENTE, Skill.CAJA, Skill.INVENTARIO, Skill.LIMPIEZA, Skill.SUPERVISOR]
    
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
                required_skills = {skills[0], skills[1]}  # Atención al cliente, Caja
            elif shift_type == ShiftType.TARDE:
                required_skills = {skills[0], skills[2]}  # Atención al cliente, Inventario
            elif shift_type == ShiftType.NOCHE:
                required_skills = {skills[3], skills[4]}  # Limpieza, Supervisor
            
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


def compare_algorithms(service: ShiftAssignmentServiceAdapter, export_adapter: ScheduleExportAdapter, interactive=False):
    """Compara los diferentes algoritmos metaheurísticos."""
    # Obtener enums de algoritmos disponibles en lugar de strings
    algorithms = service.get_available_algorithm_enums()
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"Probando algoritmo: {algorithm.to_string()}")
        start_time = time.time()
        
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
        
        # Generar solución con el enum directamente
        solution = service.generate_schedule(algorithm=algorithm, algorithm_config=config)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Almacenar resultados usando el string del enum como clave para mantener compatibilidad
        algorithm_name = algorithm.to_string()
        results[algorithm_name] = {
            "execution_time": execution_time,
            "cost": solution.total_cost,
            "violations": solution.constraint_violations,
            "solution": solution
        }
        
        # Mostrar resultados
        logger.info(f"Algoritmo: {algorithm_name}")
        logger.info(f"Tiempo de ejecución: {execution_time:.2f} segundos")
        logger.info(f"Costo total: {solution.total_cost:.2f}")
        logger.info(f"Violaciones: {solution.constraint_violations}")
        logger.info("-" * 50)
        
        # Exportar solución como texto usando el enum ExportFormat
        solution_text = export_adapter.export_solution(solution, ExportFormat.TEXT)
        logger.info(f"\nSolución con {algorithm_name}:\n{solution_text}\n")
    
    return results


def plot_comparison(results: Dict[str, Dict[str, Any]]):
    """Grafica la comparación de resultados entre algoritmos."""
    algorithms = list(results.keys())
    
    # Crear gráficos
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Gráfico de tiempo de ejecución
    execution_times = [results[algo]["execution_time"] for algo in algorithms]
    ax1.bar(algorithms, execution_times, color='blue')
    ax1.set_title('Tiempo de Ejecución')
    ax1.set_ylabel('Tiempo (segundos)')
    
    # Gráfico de costo
    costs = [results[algo]["cost"] for algo in algorithms]
    ax2.bar(algorithms, costs, color='green')
    ax2.set_title('Costo Total')
    ax2.set_ylabel('Costo')
    
    # Gráfico de violaciones
    violations = [results[algo]["violations"] for algo in algorithms]
    ax3.bar(algorithms, violations, color='red')
    ax3.set_title('Violaciones de Restricciones')
    ax3.set_ylabel('Número de violaciones')
    
    plt.tight_layout()
    
    # Guardar y mostrar el gráfico
    os.makedirs('./assets/plots', exist_ok=True)
    plt.savefig('./assets/plots/comparacion_algoritmos.png')
    logger.info("Gráfico de comparación guardado como './assets/plots/comparacion_algoritmos.png'")
    plt.show()


def main():
    """Función principal de ejemplo."""
    logger.info("Iniciando ejemplo del Sistema de Asignación Óptima de Turnos de Trabajo")
    
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
    
    # Ejecutar comparación de algoritmos
    results = compare_algorithms(shift_assignment_service, export_adapter, interactive=interactive_mode)
    
    # Graficar resultados
    plot_comparison(results)
    
    logger.info("Ejemplo completado con éxito.")


if __name__ == "__main__":
    main()