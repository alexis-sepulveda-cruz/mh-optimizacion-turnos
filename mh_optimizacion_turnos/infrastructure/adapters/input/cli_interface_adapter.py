"""Adaptador para la interfaz de línea de comandos."""

import os
import logging
import sys

from mh_optimizacion_turnos.domain.value_objects.algorithm_type import AlgorithmType
from mh_optimizacion_turnos.application.ports.input.shift_assignment_service_port import ShiftAssignmentServicePort
from mh_optimizacion_turnos.application.ports.input.test_data_setup_port import TestDataSetupPort
from mh_optimizacion_turnos.application.usecases.algorithm_comparison_usecase import AlgorithmComparisonUseCase

logger = logging.getLogger(__name__)


class CLIInterfaceAdapter:
    """Adaptador para la interfaz de línea de comandos del sistema."""
    
    def __init__(
        self,
        test_data_service: TestDataSetupPort,
        shift_assignment_service: ShiftAssignmentServicePort,
        algorithm_comparison_usecase: AlgorithmComparisonUseCase
    ):
        self.test_data_service = test_data_service
        self.shift_assignment_service = shift_assignment_service
        self.algorithm_comparison_usecase = algorithm_comparison_usecase
    
    def ask_continue_iteration(self, algorithm: AlgorithmType, run: int, total_runs: int) -> bool:
        """
        Pregunta al usuario si desea continuar con la siguiente iteración.
        
        Args:
            algorithm: Algoritmo actual
            run: Número de ejecución actual
            total_runs: Número total de ejecuciones
            
        Returns:
            True si el usuario desea continuar, False si no
        """
        while True:
            response = input(
                f"¿Desea continuar con la iteración {run}/{total_runs} del algoritmo {algorithm.to_string()}? (s/n): "
            ).strip().lower()
            
            if response in ['s', 'si', 'sí', 'y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Respuesta no válida. Por favor, ingrese 's' para sí o 'n' para no.")
    
    def configure_logging(self, log_file: str = "./assets/logs/mh-optimizacion-turnos.log"):
        """
        Configura el logging para la aplicación.
        
        Args:
            log_file: Ruta al archivo de log
        """
        # Asegurar que el directorio de logs exista
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configuración de registro
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def run_cli_interface(self):
        """Ejecuta la interfaz de línea de comandos principal."""
        logger.info("Iniciando Sistema de Asignación Óptima de Turnos de Trabajo con restricciones duras")
        
        # Crear el directorio de salida si no existe
        os.makedirs('./assets/plots', exist_ok=True)
        
        # Preguntar al usuario si desea modo interactivo
        interactive_mode = input(
            "¿Desea ejecutar los algoritmos en modo interactivo? (s/n): "
        ).strip().lower() in ['s', 'si', 'sí', 'y', 'yes']
        
        # Preguntar al usuario qué algoritmos ejecutar
        print("Algoritmos disponibles:")
        for i, algo in enumerate(self.shift_assignment_service.get_available_algorithms()):
            print(f"{i+1}. {algo}")
        
        selected_indices = input(
            "Seleccione los algoritmos a ejecutar (números separados por coma, o Enter para todos): "
        )
        
        algorithms_to_run = None
        if selected_indices.strip():
            try:
                indices = [int(i.strip()) - 1 for i in selected_indices.split(',')]
                available_algos = self.shift_assignment_service.get_available_algorithm_enums()
                algorithms_to_run = [available_algos[i] for i in indices if 0 <= i < len(available_algos)]
            except (ValueError, IndexError):
                logger.warning("Selección de algoritmos inválida. Ejecutando todos los algoritmos.")
        
        # Preguntar cuántas ejecuciones por algoritmo
        default_runs = 3
        try:
            user_runs = input(
                f"Número de ejecuciones por algoritmo para evaluar consistencia (Enter para usar {default_runs}): "
            )
            runs = default_runs
            if user_runs.strip():
                runs = max(1, int(user_runs.strip()))
        except ValueError:
            logger.warning(f"Entrada inválida. Usando {default_runs} ejecuciones por algoritmo.")
            runs = default_runs
        
        # Obtener configuración para algoritmos genéticos
        try:
            population_size = int(input("Tamaño de población para algoritmo genético (default: 20): ") or "20")
            generations = int(input("Número de generaciones para algoritmo genético (default: 30): ") or "30")
        except ValueError:
            logger.warning("Valores inválidos para algoritmo genético. Usando valores por defecto.")
            population_size = 20
            generations = 30
        
        # Obtener configuración para búsqueda tabú
        try:
            max_iterations = int(input("Máximo de iteraciones para búsqueda tabú y GRASP (default: 40): ") or "40")
            tabu_tenure = int(input("Tenencia tabú para búsqueda tabú (default: 8): ") or "8")
        except ValueError:
            logger.warning("Valores inválidos para búsqueda tabú. Usando valores por defecto.")
            max_iterations = 40
            tabu_tenure = 8
        
        # Obtener configuración para GRASP
        try:
            grasp_alpha = float(input("Factor alpha para GRASP (default: 0.3): ") or "0.3")
        except ValueError:
            logger.warning("Valor inválido para GRASP. Usando valor por defecto.")
            grasp_alpha = 0.3
        
        # Ejecutar comparación de algoritmos
        logger.info(f"Iniciando comparación con {runs} ejecuciones por algoritmo")
        
        results, raw_results = self.algorithm_comparison_usecase.compare_algorithms(
            algorithms=algorithms_to_run,
            runs=runs,
            interactive=interactive_mode,
            population_size=population_size,
            generations=generations,
            max_iterations=max_iterations,
            tabu_tenure=tabu_tenure,
            grasp_alpha=grasp_alpha,
            ask_continue_callback=self.ask_continue_iteration if interactive_mode else None
        )
        
        # Mostrar un mensaje final en consola
        print("\nProceso completado. Revisa los resultados en la carpeta 'assets/plots'.")
        print("¡Gracias por usar el Sistema de Asignación Óptima de Turnos de Trabajo!")
        
        logger.info("Ejemplo completado con éxito.")