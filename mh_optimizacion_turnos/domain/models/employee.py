from dataclasses import dataclass, field
from typing import Dict, List, Set
from uuid import UUID, uuid4


@dataclass
class Employee:
    """Employee domain entity representing a worker that can be assigned to shifts."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    max_hours_per_week: int = 40
    max_consecutive_days: int = 5
    skills: Set[str] = field(default_factory=set)
    availability: Dict[str, List[str]] = field(default_factory=dict)
    preferences: Dict[str, int] = field(default_factory=dict)
    hourly_cost: float = 0.0

    def is_available(self, day: str, shift: str) -> bool:
        """Check if an employee is available for a given day and shift."""
        if day not in self.availability:
            return False
        return shift in self.availability[day]
    
    def get_preference_score(self, day: str, shift: str) -> int:
        """Get the preference score for a day-shift combination."""
        key = f"{day}_{shift}"
        return self.preferences.get(key, 0)