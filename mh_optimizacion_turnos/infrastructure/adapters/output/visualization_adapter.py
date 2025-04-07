"""Adaptador para visualización de comparación de algoritmos."""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from mh_optimizacion_turnos.application.ports.input.algorithm_visualization_port import AlgorithmVisualizationPort

logger = logging.getLogger(__name__)


class AlgorithmVisualizationAdapter(AlgorithmVisualizationPort):
    """Adaptador para visualización y comparación de algoritmos."""
    
    def plot_comparison(self, results: Dict[str, Any], output_dir: str = "./assets/plots", 
                       show_plots: bool = False) -> Dict[str, str]:
        """
        Crea gráficos de comparación entre algoritmos y los guarda en el directorio especificado.
        
        Args:
            results: Resultados de algoritmos con métricas
            output_dir: Directorio donde guardar los gráficos generados
            show_plots: Si es True, muestra los gráficos interactivamente
            
        Returns:
            Diccionario con rutas de los archivos generados
        """
        algorithms = list(results.keys())
        
        # Crear el directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # 1. Gráfico de barras comparando tiempo, costo, fitness y cobertura
        fig_comparison, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
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
        
        fig_comparison.tight_layout()
        fig_comparison.suptitle('Comparación de Algoritmos', size=16, y=1.02)
        
        # Guardar el gráfico
        comparison_plot_path = f'{output_dir}/comparacion_algoritmos.png'
        fig_comparison.savefig(comparison_plot_path)
        generated_files['comparison_plot'] = comparison_plot_path
        logger.info(f"Gráfico de comparación guardado como '{comparison_plot_path}'")
        
        # Mostrar el gráfico si se solicita
        if show_plots:
            plt.figure(fig_comparison.number)
            plt.show()
        else:
            plt.close(fig_comparison)  # Cerrar la figura para liberar memoria si no se muestra
        
        # 2. Gráfico radar para comparar los algoritmos en múltiples dimensiones
        # Crear una nueva figura para el gráfico radar para evitar interferencias
        fig_radar = plt.figure(figsize=(10, 8))
        ax_radar = fig_radar.add_subplot(111, polar=True)
        
        # Categorías para el gráfico radar
        categories = ['Tiempo\n(inverso)', 'Costo\n(inverso)', 'Fitness', 'Cobertura', 'Consistencia\n(inverso)']
        N = len(categories)
        
        # Ángulos del gráfico (dividimos el espacio por igual)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Cerrar el polígono
        
        # Normalizar los datos para que estén entre 0 y 1
        max_time = max(avg_times) if avg_times else 1
        max_cost = max(avg_costs) if avg_costs else 1
        max_fitness = max(avg_fitness) if avg_fitness else 1
        max_coverage = max(avg_coverage) if avg_coverage else 1
        # Evitar división por cero para consistencia
        max_std = max([(results[algo]["std_cost"] / results[algo]["avg_cost"]) 
                      if results[algo]["avg_cost"] > 0 else 0 
                      for algo in algorithms]) or 1
        
        # Inicializar gráfico radar
        ax_radar.set_theta_offset(np.pi / 2)  # Rotar para que comience desde arriba
        ax_radar.set_theta_direction(-1)      # Dirección horaria
        
        # Establecer los límites del radar y las etiquetas
        ax_radar.set_ylim(0, 1)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        
        # Dibujar para cada algoritmo
        for i, algorithm in enumerate(algorithms):
            # Normalizar y convertir los valores (para tiempo, costo y std, menor es mejor, así que invertimos)
            values = [
                1 - (results[algorithm]["avg_time"] / max_time if max_time > 0 else 0),  # Tiempo (inverso)
                1 - (results[algorithm]["avg_cost"] / max_cost if max_cost > 0 else 0),  # Costo (inverso) 
                results[algorithm]["avg_fitness"] / max_fitness if max_fitness > 0 else 0,  # Fitness
                results[algorithm]["avg_coverage"] / max_coverage if max_coverage > 0 else 0,  # Cobertura
                1 - ((results[algorithm]["std_cost"] / results[algorithm]["avg_cost"]) / max_std 
                    if results[algorithm]["avg_cost"] > 0 and max_std > 0 else 0)  # Consistencia (inverso)
            ]
            values += values[:1]  # Cerrar el polígono
            
            # Dibujar el polígono y agregar etiqueta
            ax_radar.plot(angles, values, linewidth=2, label=algorithm)
            ax_radar.fill(angles, values, alpha=0.25)
        
        # Añadir leyenda y título
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        fig_radar.suptitle('Comparación multidimensional de algoritmos', size=15, y=0.95)
        
        # Ajustar el layout y guardar explícitamente
        fig_radar.tight_layout()
        
        # Guardar el gráfico radar
        radar_plot_path = f'{output_dir}/comparacion_radar.png'
        fig_radar.savefig(radar_plot_path)
        generated_files['radar_plot'] = radar_plot_path
        logger.info(f"Gráfico radar guardado como '{radar_plot_path}'")
        
        # Mostrar el gráfico si se solicita
        if show_plots:
            plt.figure(fig_radar.number)
            plt.show()
        else:
            plt.close(fig_radar)  # Cerrar la figura para liberar memoria si no se muestra
        
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
        generated_files['results_csv'] = csv_path
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
        
        best_solutions_path = f'{output_dir}/mejores_soluciones.json'
        with open(best_solutions_path, 'w') as f:
            json.dump(best_solutions, f, indent=2)
        generated_files['best_solutions_json'] = best_solutions_path
        logger.info(f"Mejores soluciones exportadas a '{best_solutions_path}'")
        
        return generated_files
    
    def find_best_algorithm(self, results: Dict[str, Any]) -> Tuple[str, Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Determina el mejor algoritmo en base a múltiples criterios.
        
        Args:
            results: Resultados de algoritmos con métricas
            
        Returns:
            Tuple con (mejor_algoritmo, criterios_ranking, puntaje_algoritmos)
        """
        algorithms = list(results.keys())
        if not algorithms:
            return None, {}, {}
        
        # Definimos los criterios y su importancia (pesos)
        criteria = {
            "costo": {"mejor": "menor", "peso": 0.40},  # El costo es el criterio más importante
            "cobertura": {"mejor": "mayor", "peso": 0.30},  # La cobertura de turnos es el segundo más importante
            "fitness": {"mejor": "mayor", "peso": 0.20},  # El fitness es el tercer criterio más importante
            "tiempo": {"mejor": "menor", "peso": 0.10}   # El tiempo es importante pero menos crítico
        }
        
        # Inicializar diccionario para almacenar puntuaciones
        scores = {algo: 0.0 for algo in algorithms}
        rankings = {}
        
        # Evaluar cada criterio
        for criterio, config in criteria.items():
            if criterio == "costo":
                values = [results[algo]["avg_cost"] for algo in algorithms]
                mejor_valor = min(values) if config["mejor"] == "menor" else max(values)
                # Para costo, un valor más bajo es mejor
                rankings["costo"] = {
                    algo: results[algo]["avg_cost"] for algo in algorithms
                }
                # Calcular puntuación normalizada (valores más bajos son mejores)
                for algo in algorithms:
                    # Si el mejor es el valor menor, invertimos la normalización
                    if config["mejor"] == "menor":
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = 1 - (results[algo]["avg_cost"] - mejor_valor) / (max_val - mejor_valor) if max_val > mejor_valor else 1
                    else:
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = results[algo]["avg_cost"] / max_val
                    
                    scores[algo] += norm_value * config["peso"]
            
            elif criterio == "cobertura":
                values = [results[algo]["avg_coverage"] for algo in algorithms]
                mejor_valor = max(values) if config["mejor"] == "mayor" else min(values)
                # Para cobertura, un valor más alto es mejor
                rankings["cobertura"] = {
                    algo: results[algo]["avg_coverage"] for algo in algorithms
                }
                # Calcular puntuación normalizada
                for algo in algorithms:
                    if config["mejor"] == "mayor":
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = results[algo]["avg_coverage"] / max_val
                    else:
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = 1 - (results[algo]["avg_coverage"] - mejor_valor) / (max_val - mejor_valor) if max_val > mejor_valor else 1
                    
                    scores[algo] += norm_value * config["peso"]
            
            elif criterio == "fitness":
                values = [results[algo]["avg_fitness"] for algo in algorithms]
                mejor_valor = max(values) if config["mejor"] == "mayor" else min(values)
                # Para fitness, un valor más alto es mejor
                rankings["fitness"] = {
                    algo: results[algo]["avg_fitness"] for algo in algorithms
                }
                # Calcular puntuación normalizada
                for algo in algorithms:
                    if config["mejor"] == "mayor":
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = results[algo]["avg_fitness"] / max_val
                    else:
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = 1 - (results[algo]["avg_fitness"] - mejor_valor) / (max_val - mejor_valor) if max_val > mejor_valor else 1
                    
                    scores[algo] += norm_value * config["peso"]
            
            elif criterio == "tiempo":
                values = [results[algo]["avg_time"] for algo in algorithms]
                mejor_valor = min(values) if config["mejor"] == "menor" else max(values)
                # Para tiempo, un valor más bajo es mejor
                rankings["tiempo"] = {
                    algo: results[algo]["avg_time"] for algo in algorithms
                }
                # Calcular puntuación normalizada
                for algo in algorithms:
                    if config["mejor"] == "menor":
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = 1 - (results[algo]["avg_time"] - mejor_valor) / (max_val - mejor_valor) if max_val > mejor_valor else 1
                    else:
                        # Evitar división por cero
                        max_val = max(values) if max(values) > 0 else 1
                        norm_value = results[algo]["avg_time"] / max_val
                    
                    scores[algo] += norm_value * config["peso"]
        
        # Encontrar el algoritmo con la mejor puntuación
        best_algorithm = max(scores.items(), key=lambda x: x[1])[0]
        
        return best_algorithm, rankings, scores
    
    def generate_algorithm_summary(self, algorithm_name: str, stats: Dict[str, Any], 
                                  scores: Dict[str, float], rankings: Dict[str, Dict[str, float]], 
                                  strengths: List[str], output_dir: str = "./assets/plots",
                                  show_plots: bool = False) -> Dict[str, str]:
        """
        Genera un resumen detallado del algoritmo y lo guarda en formato texto y JSON.
        
        Args:
            algorithm_name: Nombre del algoritmo
            stats: Estadísticas del algoritmo
            scores: Puntuaciones globales de algoritmos 
            rankings: Rankings por criterio
            strengths: Lista de fortalezas principales
            output_dir: Directorio donde guardar los resultados
            show_plots: Si es True, muestra los gráficos interactivamente
            
        Returns:
            Diccionario con rutas de los archivos generados
        """
        # Crear el directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Generar resumen en texto
        summary_lines = [
            f"ANÁLISIS DEL MEJOR ALGORITMO: {algorithm_name}",
            "=" * 60,
            f"Puntuación global: {scores[algorithm_name]:.4f} (escala 0-1)",
            "",
            "Criterios de evaluación:",
            f"  • Costo: {rankings['costo'][algorithm_name]:.2f} (40% del puntaje)",
            f"  • Cobertura: {rankings['cobertura'][algorithm_name]:.2f}% (30% del puntaje)",
            f"  • Fitness: {rankings['fitness'][algorithm_name]:.4f} (20% del puntaje)",
            f"  • Tiempo: {rankings['tiempo'][algorithm_name]:.2f}s (10% del puntaje)",
            "",
            "Métricas de rendimiento:",
            f"  • Costo promedio: {stats['avg_cost']:.2f} ± {stats['std_cost']:.2f}",
            f"  • Cobertura promedio: {stats['avg_coverage']:.1f}% ± {stats['std_coverage']:.1f}%",
            f"  • Fitness promedio: {stats['avg_fitness']:.4f} ± {stats['std_fitness']:.4f}",
            f"  • Tiempo promedio: {stats['avg_time']:.2f}s ± {stats['std_time']:.2f}s",
            "",
            "Fortalezas principales:"
        ]
        
        for strength in strengths:
            summary_lines.append(f"  • {strength.capitalize()}")
        
        summary_text = "\n".join(summary_lines)

        print(summary_text)  # Imprimir en consola para revisión
        
        # Guardar resumen en texto
        text_path = f'{output_dir}/mejor_algoritmo_resumen.txt'
        with open(text_path, 'w') as f:
            f.write(summary_text)
        
        # 2. Generar datos en JSON para análisis detallado
        json_data = {
            "nombre": algorithm_name,
            "puntuacion_global": scores[algorithm_name],
            "criterios": {
                "costo": rankings["costo"][algorithm_name],
                "cobertura": rankings["cobertura"][algorithm_name],
                "fitness": rankings["fitness"][algorithm_name],
                "tiempo": rankings["tiempo"][algorithm_name]
            },
            "metricas": {
                "costo_promedio": stats["avg_cost"],
                "costo_desviacion": stats["std_cost"],
                "cobertura_promedio": stats["avg_coverage"],
                "cobertura_desviacion": stats["std_coverage"],
                "fitness_promedio": stats["avg_fitness"],
                "fitness_desviacion": stats["std_fitness"],
                "tiempo_promedio": stats["avg_time"],
                "tiempo_desviacion": stats["std_time"]
            },
            "fortalezas": strengths
        }
        
        # Guardar datos en JSON
        json_path = f'{output_dir}/mejor_algoritmo.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        # 3. Generar visualización para el mejor algoritmo
        fig_metrics = plt.figure(figsize=(12, 10))
        gs = fig_metrics.add_gridspec(2, 2)
        ax1 = fig_metrics.add_subplot(gs[0, 0])
        ax2 = fig_metrics.add_subplot(gs[0, 1])
        ax3 = fig_metrics.add_subplot(gs[1, 0])
        ax4 = fig_metrics.add_subplot(gs[1, 1])
        
        # Gráfico de costo
        ax1.bar([algorithm_name], [stats["avg_cost"]], yerr=[stats["std_cost"]], 
               capsize=5, color='green', alpha=0.7)
        ax1.set_title('Costo Total')
        ax1.set_ylabel('Costo')
        
        # Gráfico de cobertura
        ax2.bar([algorithm_name], [stats["avg_coverage"]], yerr=[stats["std_coverage"]], 
               capsize=5, color='red', alpha=0.7)
        ax2.set_title('Cobertura de Turnos')
        ax2.set_ylabel('% Cobertura')
        
        # Gráfico de fitness
        ax3.bar([algorithm_name], [stats["avg_fitness"]], yerr=[stats["std_fitness"]], 
               capsize=5, color='purple', alpha=0.7)
        ax3.set_title('Fitness')
        ax3.set_ylabel('Fitness')
        
        # Gráfico de tiempo
        ax4.bar([algorithm_name], [stats["avg_time"]], yerr=[stats["std_time"]], 
               capsize=5, color='blue', alpha=0.7)
        ax4.set_title('Tiempo de Ejecución')
        ax4.set_ylabel('Tiempo (segundos)')
        
        fig_metrics.suptitle(f'Métricas detalladas para {algorithm_name}', size=16)
        fig_metrics.tight_layout()
        
        # Guardar visualización
        metrics_plot_path = f'{output_dir}/mejor_algoritmo_metricas.png'
        fig_metrics.savefig(metrics_plot_path)
        
        # Mostrar el gráfico si se solicita
        if show_plots:
            plt.figure(fig_metrics.number)
            plt.show()
        else:
            plt.close(fig_metrics)  # Cerrar la figura para liberar memoria si no se muestra
        
        # 4. Generar versión amigable para visualizar en consola o HTML
        friendly_text = f"""
===========================================
  ⭐ MEJOR ALGORITMO: {algorithm_name}
===========================================
PUNTUACIÓN GLOBAL: {scores[algorithm_name]:.4f}/1.0
MÉTRICAS DE RENDIMIENTO:
  • Costo: {stats["avg_cost"]:.2f} ± {stats["std_cost"]:.2f}
  • Cobertura: {stats["avg_coverage"]:.1f}% ± {stats["std_coverage"]:.1f}%
  • Fitness: {stats["avg_fitness"]:.4f} ± {stats["std_fitness"]:.4f}
  • Tiempo: {stats["avg_time"]:.2f}s ± {stats["std_time"]:.2f}s
FORTALEZAS PRINCIPALES:
  {" • ".join([s.capitalize() for s in strengths])}
===========================================
        """
        
        friendly_path = f'{output_dir}/tabu_friendly.txt'
        with open(friendly_path, 'w') as f:
            f.write(friendly_text)
        
        logger.info(f"Resumen del algoritmo {algorithm_name} guardado en {text_path}")
        
        return {
            'summary_text': text_path,
            'summary_json': json_path,
            'metrics_plot': metrics_plot_path,
            'friendly_text': friendly_path
        }
        
    def show_plot(self, plot_path: str) -> None:
        """
        Muestra un gráfico guardado previamente.
        
        Args:
            plot_path: Ruta al archivo de imagen del gráfico
        """
        try:
            # Cargar la imagen con matplotlib
            img = plt.imread(plot_path)
            
            # Crear una nueva figura y mostrar la imagen
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')  # Ocultar ejes
            plt.tight_layout()
            plt.show()
            
            logger.info(f"Mostrando gráfico: {plot_path}")
        except Exception as e:
            logger.error(f"Error al mostrar el gráfico {plot_path}: {str(e)}")
            print(f"No se pudo mostrar el gráfico: {str(e)}")