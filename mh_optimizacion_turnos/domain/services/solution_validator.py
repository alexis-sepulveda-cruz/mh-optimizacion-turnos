from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

from mh_optimizacion_turnos.domain.models.solution import Solution
from mh_optimizacion_turnos.domain.models.employee import Employee
from mh_optimizacion_turnos.domain.models.shift import Shift


@dataclass
class ValidationResult:
    """Resultado de validar una solución."""
    is_valid: bool = True
    violations: int = 0
    violation_details: List[str] = None
    
    def __post_init__(self):
        if self.violation_details is None:
            self.violation_details = []


class SolutionValidator:
    """Servicio para validar si una solución cumple con todas las restricciones."""
    
    def validate(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> ValidationResult:
        """Validar una solución contra todas las restricciones definidas.
        
        Args:
            solution: La solución a validar
            employees: Lista de todos los empleados
            shifts: Lista de todos los turnos
            
        Returns:
            Objeto ValidationResult con el estado de validación y detalles
        """
        result = ValidationResult()
        
        # Crear diccionarios de búsqueda
        employee_dict: Dict[int, Employee] = {emp.id: emp for emp in employees}
        shift_dict: Dict[int, Shift] = {shift.id: shift for shift in shifts}
        
        # Verificar que todos los turnos requeridos estén cubiertos
        self._validate_shift_coverage(solution, shifts, result)
        
        # Verificar horas máximas de empleados
        self._validate_employee_hours(solution, employee_dict, shift_dict, result)
        
        # Verificar días consecutivos de empleados
        self._validate_consecutive_days(solution, employee_dict, shift_dict, result)
        
        # Verificar que las habilidades de los empleados coincidan con los requisitos del turno
        self._validate_employee_skills(solution, employee_dict, shift_dict, result)
        
        # Verificar disponibilidad de empleados
        self._validate_employee_availability(solution, employee_dict, shift_dict, result)
        
        # Verificar asignaciones duplicadas 
        self._validate_duplicate_assignments(solution, employee_dict, shift_dict, shifts, result)
        
        # Establecer la bandera is_valid si no hay violaciones
        result.is_valid = (result.violations == 0)
        
        return result
    
    def _validate_shift_coverage(self, solution: Solution, shifts: List[Shift], 
                                result: ValidationResult) -> None:
        """Validar que todos los turnos tengan el número requerido de empleados."""
        for shift in shifts:
            assigned_employees = solution.get_shift_employees(shift.id)
            if len(assigned_employees) < shift.required_employees:
                result.violations += 1
                result.violation_details.append(
                    f"El turno {shift.name} del {shift.day} tiene {len(assigned_employees)} empleados " 
                    f"pero requiere {shift.required_employees}"
                )
    
    def _validate_employee_hours(self, solution: Solution, employee_dict: Dict[int, Employee], 
                               shift_dict: Dict[int, Shift], result: ValidationResult) -> None:
        """Validar que los empleados no excedan sus horas máximas."""
        employee_hours = defaultdict(float)
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                employee_hours[emp_id] += shift.duration_hours
                
                if employee_hours[emp_id] > employee.max_hours_per_week:
                    result.violations += 1
                    result.violation_details.append(
                        f"El empleado {employee.name} excede las horas semanales máximas: "
                        f"{employee_hours[emp_id]} > {employee.max_hours_per_week}"
                    )
    
    def _validate_consecutive_days(self, solution: Solution, employee_dict: Dict[int, Employee], 
                                 shift_dict: Dict[int, Shift], result: ValidationResult) -> None:
        """Validar que los empleados no trabajen más días consecutivos de lo permitido."""
        # Agrupar turnos por empleado y día
        employee_days = defaultdict(set)
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                shift = shift_dict[shift_id]
                employee_days[emp_id].add(shift.day)
        
        # Verificar días consecutivos
        for emp_id, days in employee_days.items():
            if emp_id in employee_dict:
                employee = employee_dict[emp_id]
                # Esta es una verificación simplificada - en una implementación real,
                # necesitaríamos convertir las fechas a objetos de fecha reales y verificar secuencias
                if len(days) > employee.max_consecutive_days:
                    result.violations += 1
                    result.violation_details.append(
                        f"El empleado {employee.name} trabaja {len(days)} días, "
                        f"excediendo el máximo de días consecutivos de {employee.max_consecutive_days}"
                    )
    
    def _validate_employee_skills(self, solution: Solution, employee_dict: Dict[int, Employee], 
                                shift_dict: Dict[int, Shift], result: ValidationResult) -> None:
        """Validar que los empleados tengan las habilidades requeridas para sus turnos."""
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                if not shift.required_skills.issubset(employee.skills):
                    missing_skills = shift.required_skills - employee.skills
                    # Convertir cada objeto Skill a su representación en cadena
                    missing_skills_str = [skill.to_string() for skill in missing_skills]
                    result.violations += 1
                    result.violation_details.append(
                        f"El empleado {employee.name} carece de las habilidades requeridas para el turno {shift.name}: "
                        f"Falta {', '.join(missing_skills_str)}"
                    )
    
    def _validate_employee_availability(self, solution: Solution, employee_dict: Dict[int, Employee], 
                                      shift_dict: Dict[int, Shift], result: ValidationResult) -> None:
        """Validar que los empleados estén disponibles para sus turnos asignados."""
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                if not employee.is_available(shift.day, shift.name):
                    result.violations += 1
                    result.violation_details.append(
                        f"El empleado {employee.name} no está disponible para el turno {shift.name} el {shift.day}"
                    )

    def _validate_duplicate_assignments(self, solution: Solution, employee_dict: Dict[int, Employee],
                                      shift_dict: Dict[int, Shift], all_shifts: List[Shift], result: ValidationResult) -> None:
        """Validar que no existan asignaciones duplicadas de turnos.
        
        Detecta dos tipos de duplicaciones:
        1. El mismo empleado asignado al mismo tipo de turno en el mismo día
        2. El mismo tipo de turno en el mismo día asignado a varios empleados cuando no se requiere
        """
        # Crear un registro de asignaciones por día y tipo de turno
        day_shift_assignments = defaultdict(list)
        
        # Registro de empleados asignados por día y tipo de turno
        day_shift_employees = defaultdict(set)
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                # Clave para identificar un tipo de turno en un día específico
                day_shift_key = (shift.day, shift.name)
                
                # Verificar si este empleado ya está asignado a este mismo tipo de turno en el mismo día
                if emp_id in day_shift_employees[day_shift_key]:
                    result.violations += 1
                    result.violation_details.append(
                        f"El empleado {employee.name} está asignado más de una vez al turno {shift.name} "
                        f"el día {shift.day}"
                    )
                
                # Añadir esta asignación al registro
                day_shift_assignments[day_shift_key].append(assignment)
                day_shift_employees[day_shift_key].add(emp_id)
        
        # Verificar si hay más empleados asignados a un mismo turno de lo necesario
        for (day, shift_type), assignments in day_shift_assignments.items():
            # Encontrar el objeto shift correspondiente a este día y tipo
            matching_shifts = [s for s in all_shifts if s.day == day and s.name == shift_type]
            
            for shift in matching_shifts:
                # Contar cuántos empleados están asignados a este turno específico
                employees_assigned = len(day_shift_employees[(day, shift_type)])
                
                # Si hay más de un empleado asignado pero el turno solo necesita uno
                if employees_assigned > shift.required_employees:
                    result.violations += 1
                    result.violation_details.append(
                        f"El turno {shift_type} del día {day} tiene {employees_assigned} empleados asignados "
                        f"pero solo requiere {shift.required_employees}"
                    )