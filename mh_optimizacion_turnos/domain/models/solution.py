from dataclasses import dataclass, field
from typing import List, Dict, Set
from uuid import UUID

from .assignment import Assignment


@dataclass
class Solution:
    """Represents a complete solution to the shift assignment problem."""
    
    assignments: List[Assignment] = field(default_factory=list)
    total_cost: float = 0.0
    constraint_violations: int = 0
    fitness_score: float = 0.0  # Higher is better
    
    def add_assignment(self, assignment: Assignment) -> None:
        """Add an assignment to the solution."""
        self.assignments.append(assignment)
        
    def get_employee_shifts(self, employee_id: UUID) -> List[UUID]:
        """Get all shifts assigned to a specific employee."""
        return [a.shift_id for a in self.assignments if a.employee_id == employee_id]
    
    def get_shift_employees(self, shift_id: UUID) -> List[UUID]:
        """Get all employees assigned to a specific shift."""
        return [a.employee_id for a in self.assignments if a.shift_id == shift_id]
    
    def calculate_total_cost(self) -> float:
        """Calculate the total cost of the solution."""
        cost_sum = sum(a.cost for a in self.assignments)
        # Asegurar que el costo nunca sea negativo
        self.total_cost = max(0.1, cost_sum)
        return self.total_cost
    
    def clone(self) -> 'Solution':
        """Create a deep copy of the solution."""
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