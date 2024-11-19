from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class ModuleMetadata:
    """Tracks information about a module's execution"""
    module_id: str
    module_type: str
    params: Dict[str, Any]
    execution_time: datetime
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    additional_metrics: Dict[str, Any] = None


class Module(ABC):
    """Base class for all modules"""
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.module_id = f"{self.name}_{uuid.uuid4().hex[:8]}"
        self.metadata = None
        pass

    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """Main computation"""
        pass

    def __call__(self, input_data: Any) -> Any:
        """Wrapper around forward to track metadata"""
        output = self.forward(input_data)
        self.metadata = ModuleMetadata(
            module_id=self.module_id,
            module_type=self.name,
            params=self.get_params(),
            execution_time=datetime.now(),
            input_shape=self._get_shape(input_data),
            output_shape=self._get_shape(output)
        )

        return output
    
    def get_params(self) -> Dict:
        """Return all the attributes of the class."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k not in ['name']}
    
    @staticmethod
    def _get_shape(data: Any) -> Optional[tuple]:
        """Utility to get shape of various data types"""
        try:
            return data.shape
        except:
            return None