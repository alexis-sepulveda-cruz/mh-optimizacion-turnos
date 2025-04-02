# Optimización de Turnos - Metaheurística

Este proyecto implementa técnicas metaheurísticas para optimizar la asignación de turnos.

## Requisitos previos

- Python 3.8 o superior
- Poetry (gestor de dependencias)

## Instalación de Poetry

Poetry es una herramienta para la gestión de dependencias y empaquetado en Python que permite declarar las librerías de las que depende tu proyecto y administrarlas (instalarlas/actualizarlas).

### Para instalar Poetry:

**En Linux, macOS, Windows (WSL)**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**En Windows (PowerShell)**:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Verificar la instalación**:
```bash
poetry --version
```

## Uso de Poetry

### Inicializar un nuevo proyecto
```bash
poetry new nombre-proyecto
```

### Inicializar un proyecto existente
```bash
cd mi-proyecto-existente
poetry init
```

### Agregar dependencias
```bash
poetry add pandas numpy matplotlib
```

### Agregar dependencias de desarrollo
```bash
poetry add pytest --dev
```

### Instalar dependencias del proyecto
```bash
poetry install
```

### Ejecutar un comando dentro del entorno virtual
```bash
poetry run python main.py
```

### Activar el entorno virtual
```bash
poetry shell
```

### Actualizar dependencias
```bash
poetry update
```

### Exportar dependencias a requirements.txt
```bash
poetry export -f requirements.txt --output requirements.txt
```

## Estructura del proyecto
```
mh-optimizacion-turnos/
├── pyproject.toml         # Configuración de Poetry y dependencias
├── poetry.lock           # Versiones exactas de dependencias
└── src/                  # Código fuente
```

## Cómo empezar

1. Clona el repositorio
2. Instala Poetry siguiendo las instrucciones anteriores
3. Instala las dependencias del proyecto:
```bash
poetry install
```
4. Ejecuta el código:
```bash
poetry run python src/main.py
```
