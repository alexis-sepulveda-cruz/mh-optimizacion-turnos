"""Caso de uso para comparación de algoritmos de optimización."""

import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple

from mh_optimizacion_turnos.domain.value_objects.algorithm_type import AlgorithmType
from mh_optimizacion_turnos.domain.value_objects.export_format import ExportFormat
from mh_optimizacion_turnos.application.ports.input.shift_assignment_service_port import ShiftAssignmentServicePort
from mh_optimizacion_turnos.application.ports.output.schedule_export_port import ScheduleExportPort
from mh_optimizacion_turnos.application.ports.input.algorithm_visualization_port import AlgorithmVisualizationPort

logger = logging.getLogger(__name__)


class AlgorithmComparisonUseCase:
    """Caso de uso para comparar diferentes algoritmos de optimización."""
    
    def __init__(
        self, 
        shift_assignment_service: ShiftAssignmentServicePort,
        schedule_export_service: ScheduleExportPort,
        visualization_service: AlgorithmVisualizationPort
    ):
        self.shift_assignment_service = shift_assignment_service
        self.schedule_export_service = schedule_export_service
        self.visualization_service = visualization_service
    
    def get_algorithm_config(self, algorithm: AlgorithmType, 
                            population_size: int = 20, 
                            generations: int = 30,
                            max_iterations: int = 40,
                            tabu_tenure: int = 8,
                            grasp_alpha: float = 0.3,
                            interactive: bool = False) -> Dict[str, Any]:
        """
        Obtiene la configuración específica para cada algoritmo.
        
        Args:
            algorithm: Tipo de algoritmo
            population_size: Tamaño de población para algoritmos genéticos
            generations: Número de generaciones para algoritmos genéticos
            max_iterations: Número máximo de iteraciones para búsqueda tabú y GRASP
            tabu_tenure: Tenencia tabú para búsqueda tabú
            grasp_alpha: Factor alpha para GRASP
            interactive: Modo interactivo habilitado
            
        Returns:
            Diccionario con la configuración específica del algoritmo
        """
        if algorithm == AlgorithmType.GENETIC:
            return {
                "population_size": population_size, 
                "generations": generations, 
                "interactive": interactive
            }
        elif algorithm == AlgorithmType.TABU:
            return {
                "max_iterations": max_iterations, 
                "tabu_tenure": tabu_tenure, 
                "interactive": interactive
            }
        elif algorithm == AlgorithmType.GRASP:
            return {
                "max_iterations": max_iterations, 
                "alpha": grasp_alpha, 
                "interactive": interactive
            }
        else:
            return {"interactive": interactive}
    
    def run_algorithm(self, algorithm: AlgorithmType, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un algoritmo específico y recopila métricas de rendimiento.
        
        Args:
            algorithm: Tipo de algoritmo a ejecutar
            config: Configuración del algoritmo
            
        Returns:
            Diccionario con las métricas de rendimiento
        """
        metrics = {
            "algorithm": algorithm.to_string(),
            "start_time": time.time(),
            "solution": None,
            "execution_time": None,
            "cost": None,
            "fitness": None,
            "violations": None,
            "assignments_count": None,
            "coverage_percentage": None
        }
        
        # Establecer el algoritmo
        self.shift_assignment_service.set_algorithm(algorithm)
        
        # Ejecutar el algoritmo
        solution = self.shift_assignment_service.generate_schedule(algorithm_config=config)
        
        # Capturar métricas
        metrics["execution_time"] = time.time() - metrics["start_time"]
        metrics["solution"] = solution
        metrics["cost"] = solution.total_cost
        metrics["fitness"] = solution.fitness_score
        metrics["violations"] = solution.constraint_violations
        metrics["assignments_count"] = len(solution.assignments)
        
        # Calcular cobertura
        shift_repo = self.shift_assignment_service.shift_optimizer_service.shift_repository
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
    
    def compare_algorithms(
        self, 
        algorithms: Optional[List[AlgorithmType]] = None,
        runs: int = 3,
        interactive: bool = False,
        output_dir: str = "./assets/plots",
        population_size: int = 20, 
        generations: int = 30,
        max_iterations: int = 40,
        tabu_tenure: int = 8,
        grasp_alpha: float = 0.3,
        ask_continue_callback = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Compara diferentes algoritmos metaheurísticos.
        
        Args:
            algorithms: Lista de algoritmos a comparar, o None para usar todos
            runs: Número de ejecuciones por algoritmo
            interactive: Modo interactivo habilitado
            output_dir: Directorio para exportar resultados
            population_size: Tamaño de población para algoritmos genéticos
            generations: Número de generaciones para algoritmos genéticos
            max_iterations: Número máximo de iteraciones para búsqueda tabú y GRASP
            tabu_tenure: Tenencia tabú para búsqueda tabú
            grasp_alpha: Factor alpha para GRASP
            ask_continue_callback: Función de retrollamada para preguntar 
                                  si continuar con las iteraciones en modo interactivo
            
        Returns:
            Tupla con (resultados_por_algoritmo, resultados_individuales_todas_ejecuciones)
        """
        if algorithms is None:
            algorithms = self.shift_assignment_service.get_available_algorithm_enums()
        
        all_results = []
        algorithm_results = {}
        
        for algorithm in algorithms:
            logger.info(f"Evaluando algoritmo: {algorithm.to_string()}")
            
            # Configuración específica para cada algoritmo
            config = self.get_algorithm_config(
                algorithm=algorithm,
                population_size=population_size,
                generations=generations,
                max_iterations=max_iterations,
                tabu_tenure=tabu_tenure,
                grasp_alpha=grasp_alpha,
                interactive=interactive
            )
            
            # Ejecutar el algoritmo varias veces para medir consistencia
            run_results = []
            for run in range(runs):
                logger.info(f"Ejecutando {algorithm.to_string()} - Corrida {run+1}/{runs}")
                
                if interactive and run > 0 and ask_continue_callback:
                    # Preguntar si continuar con las siguientes ejecuciones
                    print(f"\nPrepárando corrida {run+1}/{runs} del algoritmo {algorithm.to_string()}")
                    if not ask_continue_callback(algorithm, run+1, runs):
                        logger.info(f"Usuario decidió detener las ejecuciones de {algorithm.to_string()} después de {run} corridas.")
                        break
                
                metrics = self.run_algorithm(algorithm, config)
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
                solution_text = self.schedule_export_service.export_solution(metrics['solution'], ExportFormat.TEXT)
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
        
        # Generar visualizaciones de los resultados
        self.visualization_service.plot_comparison(algorithm_results, output_dir)
        
        # Encontrar el mejor algoritmo
        best_algorithm, rankings, scores = self.visualization_service.find_best_algorithm(algorithm_results)
        if best_algorithm:
            # Generar resumen del mejor algoritmo
            best_stats = algorithm_results[best_algorithm]
            
            # Analizar fortalezas del mejor algoritmo
            strengths = []
            if best_stats["avg_cost"] == min(algorithm_results[algo]["avg_cost"] for algo in algorithm_results):
                strengths.append("menor costo total")
            
            if best_stats["avg_coverage"] == max(algorithm_results[algo]["avg_coverage"] for algo in algorithm_results):
                strengths.append("mayor cobertura de turnos")
                
            if best_stats["avg_fitness"] == max(algorithm_results[algo]["avg_fitness"] for algo in algorithm_results):
                strengths.append("mejor fitness")
                
            if best_stats["avg_time"] == min(algorithm_results[algo]["avg_time"] for algo in algorithm_results):
                strengths.append("menor tiempo de ejecución")
                
            if not strengths:
                strengths = ["mejor equilibrio general entre todos los criterios de evaluación"]
            
            # Generar resumen del algoritmo
            self.visualization_service.generate_algorithm_summary(
                algorithm_name=best_algorithm,
                stats=best_stats,
                scores=scores,
                rankings=rankings,
                strengths=strengths,
                output_dir=output_dir
            )
            
            logger.info(f"El mejor algoritmo según los criterios de optimización es: {best_algorithm}")
        
        return algorithm_results, all_results