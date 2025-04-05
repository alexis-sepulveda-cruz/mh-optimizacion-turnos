from dataclasses import dataclass, field
from typing import List
from uuid import UUID

from .assignment import Assignment


@dataclass
class Solution:
    """Representa una solución completa al problema de asignación de turnos."""
    
    assignments: List[Assignment] = field(default_factory=list)
    total_cost: float = 0.0
    constraint_violations: int = 0
    fitness_score: float = 0.0  # Más alto es mejor
    
    def add_assignment(self, assignment: Assignment) -> None:
        """Añadir una asignación a la solución."""
        self.assignments.append(assignment)
        
    def get_employee_shifts(self, employee_id: UUID) -> List[UUID]:
        """Obtener todos los turnos asignados a un empleado específico."""
        return [a.shift_id for a in self.assignments if a.employee_id == employee_id]
    
    def get_shift_employees(self, shift_id: UUID) -> List[UUID]:
        """Obtener todos los empleados asignados a un turno específico."""
        return [a.employee_id for a in self.assignments if a.shift_id == shift_id]
    
    def calculate_total_cost(self) -> float:
        """Calcular el costo total de la solución."""
        cost_sum = sum(a.cost for a in self.assignments)
        # Asegurar que el costo nunca sea negativo
        self.total_cost = max(0.1, cost_sum)
        return self.total_cost
    
    def clone(self) -> 'Solution':
        """Crear una copia profunda de la solución."""
        new_solution = Solution()
        new_solution.assignments = [Assignment(
            employee_id=a.employee_id,
            shift_id=a.shift_id,
            cost=a.cost
        ) for a in self.assignments]
        new_solution.total_cost = self.total_cost
        new_solution.constraint_violations = self.constraint_violations
        new_solution.fitness_score = self.fitness_score
        return new_solution