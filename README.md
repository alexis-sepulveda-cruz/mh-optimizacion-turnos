# Sistema de Asignación Óptima de Turnos de Trabajo

Sistema para optimizar la asignación de empleados a turnos de trabajo minimizando costos y asegurando el cumplimiento de restricciones como disponibilidad, horas máximas permitidas y balance de carga laboral.

## Características

- **Arquitectura Hexagonal**: Separa la lógica de negocio de los detalles de implementación
- **Domain-Driven Design (DDD)**: Modelado del dominio basado en entidades y servicios de dominio
- **Patrón Strategy**: Implementación intercambiable de algoritmos metaheurísticos
- **Algoritmos implementados**:
  - Algoritmo Genético
  - Búsqueda Tabú
  - GRASP (Greedy Randomized Adaptive Search Procedure)
- **Principios SOLID**: Aplicados en todo el diseño
- **Exportación de resultados**: En múltiples formatos (JSON, CSV, Excel, texto)

## Estructura del Proyecto

```
mh_optimizacion_turnos/
├── application/                # Capa de aplicación
│   └── ports/                  # Puertos de entrada y salida
├── domain/                     # Núcleo de dominio
│   ├── models/                 # Entidades del dominio
│   ├── repositories/           # Interfaces de repositorios
│   └── services/               # Servicios de dominio
│       └── optimizers/         # Implementaciones de algoritmos
├── infrastructure/             # Capa de infraestructura
│   ├── adapters/               # Adaptadores para puertos
│   └── repositories/           # Implementaciones de repositorios
└── config/                     # Configuraciones
```

## Requisitos

- Python 3.12+
- Dependencias (instalables con Poetry):
  - numpy
  - pandas
  - matplotlib

## Instalación

```bash
# Clonar el repositorio
git clone [url-del-repositorio]
cd mh-optimizacion-turnos

# Instalar dependencias con Poetry
poetry install
```

## Uso

### Ejemplo básico

```python
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.services.solution_validator import SolutionValidator
from mh_optimizacion_turnos.domain.services.shift_optimizer_service import ShiftOptimizerService
from mh_optimizacion_turnos.infrastructure.repositories.in_memory_employee_repository import InMemoryEmployeeRepository
from mh_optimizacion_turnos.infrastructure.repositories.in_memory_shift_repository import InMemoryShiftRepository
from mh_optimizacion_turnos.infrastructure.adapters.input.shift_assignment_service_adapter import ShiftAssignmentServiceAdapter

# Crear repositorios
employee_repo = InMemoryEmployeeRepository()
shift_repo = InMemoryShiftRepository()

# Agregar empleados y turnos a los repositorios
# ...

# Crear servicios
solution_validator = SolutionValidator()
optimizer_service = ShiftOptimizerService(
    employee_repository=employee_repo,
    shift_repository=shift_repo,
    solution_validator=solution_validator
)

# Crear adaptador de servicio
assignment_service = ShiftAssignmentServiceAdapter(optimizer_service)

# Generar solución con diferentes algoritmos
solution_genetic = assignment_service.generate_schedule(algorithm="genetic")
solution_tabu = assignment_service.generate_schedule(algorithm="tabu")
solution_grasp = assignment_service.generate_schedule(algorithm="grasp")
```

### Ejemplo completo

El archivo `example.py` incluido en el repositorio contiene un ejemplo completo que:
1. Crea datos de prueba aleatorios (empleados y turnos)
2. Ejecuta los tres algoritmos metaheurísticos
3. Compara resultados (tiempo de ejecución, costo, violaciones)
4. Genera gráficos comparativos

Para ejecutar el ejemplo:

```bash
python example.py
```

## Configuración de algoritmos

Cada algoritmo puede ser configurado con diferentes parámetros:

### Algoritmo Genético
```python
config = {
    "population_size": 50,      # Tamaño de población 
    "generations": 100,         # Número de generaciones
    "mutation_rate": 0.1,       # Tasa de mutación
    "crossover_rate": 0.8,      # Tasa de cruzamiento
    "elitism_count": 5,         # Número de soluciones élite que se mantienen
    "tournament_size": 3        # Tamaño del torneo para selección
}
```

### Búsqueda Tabú
```python
config = {
    "max_iterations": 1000,     # Número máximo de iteraciones
    "tabu_tenure": 20,          # Duración de la prohibición tabú
    "neighborhood_size": 30,    # Tamaño del vecindario a explorar
    "max_iterations_without_improvement": 200  # Criterio de parada
}
```

### GRASP
```python
config = {
    "max_iterations": 100,      # Número máximo de iteraciones
    "alpha": 0.3,               # Factor de aleatoriedad (0=greedy, 1=aleatorio)
    "local_search_iterations": 50  # Iteraciones de búsqueda local
}
```

## Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

1. Para añadir un nuevo algoritmo metaheurístico:
   - Crear una nueva clase que herede de `OptimizerStrategy`
   - Implementar los métodos requeridos
   - Registrar el algoritmo en el adaptador de servicio

2. Para cambiar la persistencia:
   - Implementar nuevas clases que hereden de los repositorios de dominio
   - Configurar las dependencias acordemente

## Licencia

Este proyecto está licenciado bajo [licencia a definir].
