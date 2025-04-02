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


def setup_test_data():
    """Configura datos de ejemplo para pruebas."""
    # Crear repositorios
    employee_repo = InMemoryEmployeeRepository()
    shift_repo = InMemoryShiftRepository()
    
    # Crear turnos
    days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    shift_types = ["Mañana", "Tarde", "Noche"]
    skills = ["Atención al cliente", "Caja", "Inventario", "Limpieza", "Supervisor"]
    
    # Horas para cada tipo de turno
    shift_hours = {
        "Mañana": (datetime(2023, 1, 1, 8, 0), datetime(2023, 1, 1, 16, 0)),
        "Tarde": (datetime(2023, 1, 1, 16, 0), datetime(2023, 1, 1, 0, 0)),
        "Noche": (datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 8, 0))
    }
    
    # Crear turnos para cada día y tipo
    for day in days:
        for shift_type in shift_types:
            start_time, end_time = shift_hours[shift_type]
            required_skills = set()
            
            # Diferentes habilidades requeridas según el turno
            if shift_type == "Mañana":
                required_skills = {skills[0], skills[1]}
            elif shift_type == "Tarde":
                required_skills = {skills[0], skills[2]}
            elif shift_type == "Noche":
                required_skills = {skills[3], skills[4]}
            
            shift = Shift(
                name=shift_type,
                day=day,
                start_time=start_time,
                end_time=end_time,
                required_employees=3,  # Cada turno necesita 3 empleados
                required_skills=required_skills,
                priority=2 if shift_type == "Mañana" else 1  # Mañana mayor prioridad
            )
            shift_repo.save(shift)
    
    # Crear empleados
    for i in range(1, 11):  # 10 empleados
        employee = Employee(
            name=f"Empleado {i}",
            max_hours_per_week=40,
            max_consecutive_days=5,
            skills=set(np.random.choice(skills, size=np.random.randint(1, 4), replace=False)),
            hourly_cost=np.random.uniform(10, 20)  # Costo por hora entre 10 y 20
        )
        
        # Definir disponibilidad aleatoria
        availability = {}
        for day in days:
            available_shifts = np.random.choice(shift_types, 
                                              size=np.random.randint(1, len(shift_types) + 1),
                                              replace=False)
            availability[day] = list(available_shifts)
        
        employee.availability = availability
        
        # Definir preferencias aleatorias
        preferences = {}
        for day in days:
            for shift_type in shift_types:
                if shift_type in availability[day]:
                    # Mayor probabilidad de preferir mañana
                    if shift_type == "Mañana":
                        preference = np.random.randint(3, 6)
                    else:
                        preference = np.random.randint(1, 4)
                    preferences[f"{day}_{shift_type}"] = preference
        
        employee.preferences = preferences
        employee_repo.save(employee)
    
    return employee_repo, shift_repo


def compare_algorithms(service: ShiftAssignmentServicePort, export_adapter: ScheduleExportAdapter):
    """Compara los diferentes algoritmos metaheurísticos."""
    algorithms = service.get_available_algorithms()
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"Probando algoritmo: {algorithm}")
        start_time = time.time()
        
        # Configuración específica para cada algoritmo
        if algorithm == "genetic":
            config = {"population_size": 30, "generations": 50}
        elif algorithm == "tabu":
            config = {"max_iterations": 50, "tabu_tenure": 10}
        elif algorithm == "grasp":
            config = {"max_iterations": 30, "alpha": 0.3}
        else:
            config = None
        
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
    
    # Ejecutar comparación de algoritmos
    results = compare_algorithms(shift_assignment_service, export_adapter)
    
    # Graficar resultados
    plot_comparison(results)
    
    logger.info("Ejemplo completado con éxito.")


if __name__ == "__main__":
    main()