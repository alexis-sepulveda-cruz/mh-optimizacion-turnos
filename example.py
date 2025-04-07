#!/usr/bin/env python
"""
Ejemplo de uso del Sistema de Asignación Óptima de Turnos de Trabajo.

Este script demuestra la implementación completa del sistema utilizando
diferentes algoritmos metaheurísticos (Algoritmo Genético, Búsqueda Tabú y GRASP)
para optimizar la asignación de turnos a empleados.
"""

import os
import logging
import sys

# Importar componentes del dominio
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.services.shift_optimizer_service import ShiftOptimizerService

# Importar adaptadores y puertos
from mh_optimizacion_turnos.application.usecases.setup_test_data_usecase import SetupTestDataUseCase
from mh_optimizacion_turnos.application.usecases.algorithm_comparison_usecase import AlgorithmComparisonUseCase
from mh_optimizacion_turnos.infrastructure.adapters.input.shift_assignment_service_adapter import ShiftAssignmentServiceAdapter
from mh_optimizacion_turnos.infrastructure.adapters.input.cli_interface_adapter import CLIInterfaceAdapter
from mh_optimizacion_turnos.infrastructure.adapters.output.schedule_export_adapter import ScheduleExportAdapter
from mh_optimizacion_turnos.infrastructure.adapters.output.visualization_adapter import AlgorithmVisualizationAdapter


def main():
    """Función principal del ejemplo."""
    # 1. Configurar logging básico antes de iniciar cualquier componente
    os.makedirs('./assets/logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("./assets/logs/mh-optimizacion-turnos.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Iniciando sistema de asignación óptima de turnos")

    # 2. Crear instancias de servicios del dominio
    solution_validator = SolutionValidator()
    
    # 3. Crear instancias de casos de uso
    test_data_usecase = SetupTestDataUseCase()
    
    # 4. Configurar datos de prueba
    employee_repo, shift_repo = test_data_usecase.setup_test_data()
    
    # 5. Instanciar servicios del dominio con sus dependencias
    shift_optimizer_service = ShiftOptimizerService(
        employee_repository=employee_repo,
        shift_repository=shift_repo,
        solution_validator=solution_validator
    )
    
    # 6. Crear adaptadores de entrada y salida
    shift_assignment_adapter = ShiftAssignmentServiceAdapter(
        shift_optimizer_service=shift_optimizer_service
    )
    
    schedule_export_adapter = ScheduleExportAdapter(
        employee_repository=employee_repo,
        shift_repository=shift_repo
    )
    
    visualization_adapter = AlgorithmVisualizationAdapter()
    
    # 7. Instanciar caso de uso de comparación con sus dependencias
    algorithm_comparison_usecase = AlgorithmComparisonUseCase(
        shift_assignment_service=shift_assignment_adapter,
        schedule_export_service=schedule_export_adapter,
        visualization_service=visualization_adapter
    )
    
    # 8. Crear adaptador de interfaz CLI
    cli_interface = CLIInterfaceAdapter(
        test_data_service=test_data_usecase,
        shift_assignment_service=shift_assignment_adapter,
        algorithm_comparison_usecase=algorithm_comparison_usecase
    )
    
    # 9. Ejecutar la interfaz CLI
    cli_interface.run_cli_interface()


if __name__ == "__main__":
    main()