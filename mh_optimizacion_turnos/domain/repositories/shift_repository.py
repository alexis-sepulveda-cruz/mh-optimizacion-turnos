from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..models.shift import Shift


class ShiftRepository(ABC):
    """Repository interface for Shift entities."""
    
    @abstractmethod
    def get_by_id(self, shift_id: UUID) -> Optional[Shift]:
        """Get a shift by ID."""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Shift]:
        """Get all shifts."""
        pass
    
    @abstractmethod
    def save(self, shift: Shift) -> Shift:
        """Save a shift (create or update)."""
        pass
    
    @abstractmethod
    def delete(self, shift_id: UUID) -> None:
        """Delete a shift."""
        pass
    
    @abstractmethod
    def get_shifts_by_day(self, day: str) -> List[Shift]:
        """Get all shifts for a specific day."""
        pass
    
    @abstractmethod
    def get_shifts_by_period(self, start_date: str, end_date: str) -> List[Shift]:
        """Get all shifts within a date range."""
        pass