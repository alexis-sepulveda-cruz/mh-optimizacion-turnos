#!/usr/bin/env python
"""
Ejemplo de uso del Sistema de Asignación Óptima de Turnos de Trabajo.

Este script demuestra la implementación completa del sistema utilizando
diferentes algoritmos metaheurísticos (Algoritmo Genético, Búsqueda Tabú y GRASP)
para optimizar la asignación de turnos a empleados.
"""

import logging
import sys
import os
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

from mh_optimizacion_turnos.domain.models.day import Day
from mh_optimizacion_turnos.domain.models.shift_type import ShiftType
from mh_optimizacion_turnos.domain.models.skill import Skill
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.services.shift_optimizer_service import ShiftOptimizerService

from mh_optimizacion_turnos.infrastructure.repositories.in_memory_employee_repository import InMemoryEmployeeRepository
from mh_optimizacion_turnos.infrastructure.repositories.in_memory_shift_repository import InMemoryShiftRepository

from mh_optimizacion_turnos.application.ports.input.shift_assignment_service_port import ShiftAssignmentServicePort
from mh_optimizacion_turnos.infrastructure.adapters.input.shift_assignment_service_adapter import ShiftAssignmentServiceAdapter
from mh_optimizacion_turnos.infrastructure.adapters.output.schedule_export_adapter import ScheduleExportAdapter

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
        ShiftType.MAÑANA: (datetime(2025, 1, 1, 8, 0), datetime(2025, 1, 1, 16, 0)),
        ShiftType.TARDE: (datetime(2025, 1, 1, 16, 0), datetime(2025, 1, 1, 0, 0)),
        ShiftType.NOCHE: (datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 8, 0))
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
                required_employees=3,  # Cada turno necesita 3 empleados
                required_skills=required_skills,
                priority=2 if shift_type == ShiftType.MAÑANA else 1  # Mañana mayor prioridad
            )
            shift_repo.save(shift)
    
    # Crear empleados
    for i in range(1, 11):  # 10 empleados
        # Seleccionar aleatoriamente entre 1 y 3 habilidades
        random_skill_count = np.random.randint(1, 4)
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
            max_hours_per_week=40,
            max_consecutive_days=5,
            skills=random_skills,
            hourly_cost=np.random.uniform(10, 20)  # Costo por hora entre 10 y 20
        )
        
        # Definir disponibilidad aleatoria utilizando los enums
        availability = {}
        for day in days:
            # Seleccionar aleatoriamente entre 1 y 3 tipos de turnos
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
                        preference = np.random.randint(3, 6)
                    else:
                        preference = np.random.randint(1, 4)
                    # Guardar preferencia usando directamente los enums
                    preferences[day][shift_type] = preference
        
        employee.preferences = preferences
        employee_repo.save(employee)
    
    return employee_repo, shift_repo


def compare_algorithms(service: ShiftAssignmentServicePort, export_adapter: ScheduleExportAdapter, interactive=False):
    """Compara los diferentes algoritmos metaheurísticos."""
    algorithms = service.get_available_algorithms()
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"Probando algoritmo: {algorithm}")
        start_time = time.time()
        
        # Configuración específica para cada algoritmo
        if algorithm == "genetic":
            config = {"population_size": 30, "generations": 50, "interactive": interactive}
        elif algorithm == "tabu":
            config = {"max_iterations": 50, "tabu_tenure": 10, "interactive": interactive}
        elif algorithm == "grasp":
            config = {"max_iterations": 30, "alpha": 0.3, "interactive": interactive}
        else:
            config = {"interactive": interactive}
        
        # Generar solución
        solution = service.generate_schedule(algorithm=algorithm, algorithm_config=config)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Almacenar resultados
        results[algorithm] = {
            "execution_time": execution_time,
            "cost": solution.total_cost,
            "violations": solution.constraint_violations,
            "solution": solution
        }
        
        # Mostrar resultados
        logger.info(f"Algoritmo: {algorithm}")
        logger.info(f"Tiempo de ejecución: {execution_time:.2f} segundos")
        logger.info(f"Costo total: {solution.total_cost:.2f}")
        logger.info(f"Violaciones: {solution.constraint_violations}")
        logger.info("-" * 50)
        
        # Exportar solución como texto
        solution_text = export_adapter.export_solution(solution, "text")
        logger.info(f"\nSolución con {algorithm}:\n{solution_text}\n")
    
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
    plt.savefig('comparacion_algoritmos.png')
    logger.info("Gráfico de comparación guardado como 'comparacion_algoritmos.png'")
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