"""Caso de uso para configuración de datos de prueba."""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift
from mh_optimizacion_turnos.domain.value_objects.day import Day
from mh_optimizacion_turnos.domain.value_objects.shift_type import ShiftType
from mh_optimizacion_turnos.domain.value_objects.skill import Skill
from mh_optimizacion_turnos.domain.repositories.employee_repository import EmployeeRepository
from mh_optimizacion_turnos.domain.repositories.shift_repository import ShiftRepository
from mh_optimizacion_turnos.infrastructure.repositories.in_memory_employee_repository import InMemoryEmployeeRepository
from mh_optimizacion_turnos.infrastructure.repositories.in_memory_shift_repository import InMemoryShiftRepository
from mh_optimizacion_turnos.application.ports.input.test_data_setup_port import TestDataSetupPort

logger = logging.getLogger(__name__)


class SetupTestDataUseCase(TestDataSetupPort):
    """Implementación del caso de uso para configuración de datos de prueba."""
    
    def get_default_config(self) -> Dict[str, Any]:
        """Obtiene la configuración por defecto para la generación de datos."""
        return {
            "NUM_EMPLOYEES": 30,
            "MIN_EMPLOYEE_ID": 1,
            "MAX_HOURS_PER_WEEK": 40,
            "MAX_CONSECUTIVE_DAYS": 5,
            "MIN_HOURLY_COST": 10.0,
            "MAX_HOURLY_COST": 20.0,
            "EMPLOYEES_PER_SHIFT": 1,
            "MIN_EMPLOYEE_SKILLS": 3,
            "MAX_EMPLOYEE_SKILLS": 6,
            "MORNING_SHIFT_START": 8,
            "MORNING_SHIFT_END": 16,
            "AFTERNOON_SHIFT_START": 16,
            "AFTERNOON_SHIFT_END": 0,
            "NIGHT_SHIFT_START": 0,
            "NIGHT_SHIFT_END": 8,
            "HIGH_PRIORITY": 2,
            "NORMAL_PRIORITY": 1,
            "MIN_REGULAR_PREFERENCE": 1,
            "MAX_REGULAR_PREFERENCE": 4,
            "MIN_MORNING_PREFERENCE": 3,
            "MAX_MORNING_PREFERENCE": 6
        }
    
    def setup_test_data(self, config: Dict[str, Any] = None) -> Tuple[EmployeeRepository, ShiftRepository]:
        """Configura datos de ejemplo para pruebas."""
        if config is None:
            config = self.get_default_config()
            
        # Crear repositorios
        employee_repo = InMemoryEmployeeRepository()
        shift_repo = InMemoryShiftRepository()
        
        # Extraer configuración
        NUM_EMPLOYEES = config.get("NUM_EMPLOYEES", 30)
        MIN_EMPLOYEE_ID = config.get("MIN_EMPLOYEE_ID", 1)
        MAX_HOURS_PER_WEEK = config.get("MAX_HOURS_PER_WEEK", 40)
        MAX_CONSECUTIVE_DAYS = config.get("MAX_CONSECUTIVE_DAYS", 5)
        MIN_HOURLY_COST = config.get("MIN_HOURLY_COST", 10.0)
        MAX_HOURLY_COST = config.get("MAX_HOURLY_COST", 20.0)
        EMPLOYEES_PER_SHIFT = config.get("EMPLOYEES_PER_SHIFT", 1)
        MIN_EMPLOYEE_SKILLS = config.get("MIN_EMPLOYEE_SKILLS", 3)
        MAX_EMPLOYEE_SKILLS = config.get("MAX_EMPLOYEE_SKILLS", 6)
        
        MORNING_SHIFT_START = config.get("MORNING_SHIFT_START", 8)
        MORNING_SHIFT_END = config.get("MORNING_SHIFT_END", 16)
        AFTERNOON_SHIFT_START = config.get("AFTERNOON_SHIFT_START", 16)
        AFTERNOON_SHIFT_END = config.get("AFTERNOON_SHIFT_END", 0)
        NIGHT_SHIFT_START = config.get("NIGHT_SHIFT_START", 0)
        NIGHT_SHIFT_END = config.get("NIGHT_SHIFT_END", 8)
        HIGH_PRIORITY = config.get("HIGH_PRIORITY", 2)
        NORMAL_PRIORITY = config.get("NORMAL_PRIORITY", 1)
        
        MIN_REGULAR_PREFERENCE = config.get("MIN_REGULAR_PREFERENCE", 1)
        MAX_REGULAR_PREFERENCE = config.get("MAX_REGULAR_PREFERENCE", 4)
        MIN_MORNING_PREFERENCE = config.get("MIN_MORNING_PREFERENCE", 3)
        MAX_MORNING_PREFERENCE = config.get("MAX_MORNING_PREFERENCE", 6)
        
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
                    required_skills = {skills[0], skills[1], skills[2]}  # Atención al cliente, Manufactura, Caja
                elif shift_type == ShiftType.TARDE:
                    required_skills = {skills[0], skills[3], skills[1]}  # Atención al cliente, Inventario, Manufactura
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
        self._create_employees(
            employee_repo=employee_repo,
            employee_count=NUM_EMPLOYEES,
            min_employee_id=MIN_EMPLOYEE_ID,
            max_hours_per_week=MAX_HOURS_PER_WEEK,
            max_consecutive_days=MAX_CONSECUTIVE_DAYS,
            min_hourly_cost=MIN_HOURLY_COST,
            max_hourly_cost=MAX_HOURLY_COST,
            min_skills=MIN_EMPLOYEE_SKILLS,
            max_skills=MAX_EMPLOYEE_SKILLS,
            skills_list=skills,
            days=days,
            shift_types=shift_types,
            min_regular_preference=MIN_REGULAR_PREFERENCE,
            max_regular_preference=MAX_REGULAR_PREFERENCE,
            min_morning_preference=MIN_MORNING_PREFERENCE,
            max_morning_preference=MAX_MORNING_PREFERENCE
        )
        
        logger.info(f"Configurados {NUM_EMPLOYEES} empleados y {len(days) * len(shift_types)} turnos de prueba")
        
        return employee_repo, shift_repo
    
    def _create_employees(self, employee_repo: EmployeeRepository, employee_count: int, 
                         min_employee_id: int, max_hours_per_week: int, 
                         max_consecutive_days: int, min_hourly_cost: float, 
                         max_hourly_cost: float, min_skills: int, max_skills: int,
                         skills_list: list, days: list, shift_types: list,
                         min_regular_preference: int, max_regular_preference: int,
                         min_morning_preference: int, max_morning_preference: int) -> None:
        """Crea empleados con características aleatorias."""
        for i in range(min_employee_id, min_employee_id + employee_count):
            # Seleccionar aleatoriamente entre min_skills y max_skills habilidades
            random_skill_count = np.random.randint(min_skills, max_skills + 1)
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
                max_hours_per_week=max_hours_per_week,
                max_consecutive_days=max_consecutive_days,
                skills=random_skills,
                hourly_cost=np.random.uniform(min_hourly_cost, max_hourly_cost)
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
                            preference = np.random.randint(min_morning_preference, max_morning_preference)
                        else:
                            preference = np.random.randint(min_regular_preference, max_regular_preference)
                        # Guardar preferencia usando directamente los enums
                        preferences[day][shift_type] = preference
            
            employee.preferences = preferences
            employee_repo.save(employee)