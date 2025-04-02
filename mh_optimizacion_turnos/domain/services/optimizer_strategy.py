from abc import ABC, abstractmethod
from typing import Dict, Any, List
from uuid import UUID

from ..models.solution import Solution
from ..models.employee import Employee
from ..models.shift import Shift


class OptimizerStrategy(ABC):
    """Strategy interface for optimization algorithms.
    
    This follows the Strategy pattern to allow interchangeable metaheuristic algorithms.
    Each algorithm implementation must follow this interface.
    """
    
    @abstractmethod
    def optimize(self, 
                employees: List[Employee], 
                shifts: List[Shift],
                config: Dict[str, Any] = None) -> Solution:
        """Optimize shift assignments based on the given employees and shifts.
        
        Args:
            employees: List of available employees
            shifts: List of shifts to be assigned
            config: Algorithm-specific configuration parameters
            
        Returns:
            A Solution object containing the optimized assignments
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the optimization strategy."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration parameters for this algorithm."""
        pass