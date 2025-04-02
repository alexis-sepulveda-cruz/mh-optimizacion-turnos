from dataclasses import dataclass
from typing import List, Dict, Set
from collections import defaultdict

from ..models.solution import Solution
from ..models.employee import Employee
from ..models.shift import Shift


@dataclass
class ValidationResult:
    """Result of validating a solution."""
    is_valid: bool = True
    violations: int = 0
    violation_details: List[str] = None
    
    def __post_init__(self):
        if self.violation_details is None:
            self.violation_details = []


class SolutionValidator:
    """Service to validate if a solution meets all constraints."""
    
    def validate(self, solution: Solution, employees: List[Employee], shifts: List[Shift]) -> ValidationResult:
        """Validate a solution against all defined constraints.
        
        Args:
            solution: The solution to validate
            employees: List of all employees
            shifts: List of all shifts
            
        Returns:
            ValidationResult object with validation status and details
        """
        result = ValidationResult()
        
        # Create lookup dictionaries
        employee_dict = {emp.id: emp for emp in employees}
        shift_dict = {shift.id: shift for shift in shifts}
        
        # Check that all required shifts are covered
        self._validate_shift_coverage(solution, shifts, result)
        
        # Check employee maximum hours
        self._validate_employee_hours(solution, employee_dict, shift_dict, result)
        
        # Check employee consecutive days
        self._validate_consecutive_days(solution, employee_dict, shift_dict, result)
        
        # Check employee skills match shift requirements
        self._validate_employee_skills(solution, employee_dict, shift_dict, result)
        
        # Check employee availability
        self._validate_employee_availability(solution, employee_dict, shift_dict, result)
        
        # Set the is_valid flag if there are no violations
        result.is_valid = (result.violations == 0)
        
        return result
    
    def _validate_shift_coverage(self, solution: Solution, shifts: List[Shift], 
                                result: ValidationResult) -> None:
        """Validate that all shifts have the required number of employees."""
        for shift in shifts:
            assigned_employees = solution.get_shift_employees(shift.id)
            if len(assigned_employees) < shift.required_employees:
                result.violations += 1
                result.violation_details.append(
                    f"Shift {shift.name} on {shift.day} has {len(assigned_employees)} employees " 
                    f"but requires {shift.required_employees}"
                )
    
    def _validate_employee_hours(self, solution: Solution, employee_dict: Dict, 
                               shift_dict: Dict, result: ValidationResult) -> None:
        """Validate that employees don't exceed their maximum hours."""
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
                        f"Employee {employee.name} exceeds maximum weekly hours: "
                        f"{employee_hours[emp_id]} > {employee.max_hours_per_week}"
                    )
    
    def _validate_consecutive_days(self, solution: Solution, employee_dict: Dict, 
                                 shift_dict: Dict, result: ValidationResult) -> None:
        """Validate that employees don't work more consecutive days than allowed."""
        # Group shifts by employee and day
        employee_days = defaultdict(set)
        
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                shift = shift_dict[shift_id]
                employee_days[emp_id].add(shift.day)
        
        # Check consecutive days
        for emp_id, days in employee_days.items():
            if emp_id in employee_dict:
                employee = employee_dict[emp_id]
                # This is a simplified check - in a real implementation,
                # we would need to convert dates to actual date objects and check sequences
                if len(days) > employee.max_consecutive_days:
                    result.violations += 1
                    result.violation_details.append(
                        f"Employee {employee.name} works {len(days)} days, "
                        f"exceeding maximum consecutive days of {employee.max_consecutive_days}"
                    )
    
    def _validate_employee_skills(self, solution: Solution, employee_dict: Dict, 
                                shift_dict: Dict, result: ValidationResult) -> None:
        """Validate that employees have the required skills for their shifts."""
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                if not shift.required_skills.issubset(employee.skills):
                    missing_skills = shift.required_skills - employee.skills
                    result.violations += 1
                    result.violation_details.append(
                        f"Employee {employee.name} lacks required skills for shift {shift.name}: "
                        f"Missing {', '.join(missing_skills)}"
                    )
    
    def _validate_employee_availability(self, solution: Solution, employee_dict: Dict, 
                                      shift_dict: Dict, result: ValidationResult) -> None:
        """Validate that employees are available for their assigned shifts."""
        for assignment in solution.assignments:
            emp_id = assignment.employee_id
            shift_id = assignment.shift_id
            
            if emp_id in employee_dict and shift_id in shift_dict:
                employee = employee_dict[emp_id]
                shift = shift_dict[shift_id]
                
                if not employee.is_available(shift.day, shift.name):
                    result.violations += 1
                    result.violation_details.append(
                        f"Employee {employee.name} is not available for shift {shift.name} on {shift.day}"
                    )