"""Tensor field manager for robotics workbench."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from ..coresys import CoreSysAttributes

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass
class TensorField:
    """Represents a tensor field with metadata."""
    
    field_id: str
    name: str
    shape: Tuple[int, ...]
    dtype: str
    data: Optional[np.ndarray]
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float

    def __post_init__(self):
        """Initialize computed properties."""
        if self.data is None and self.shape:
            # Initialize with zeros
            self.data = np.zeros(self.shape, dtype=self.dtype)
        
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = time.time()
        self.updated_at = self.created_at

    @property
    def element_count(self) -> int:
        """Get total number of elements."""
        return int(np.prod(self.shape)) if self.shape else 0

    @property
    def degrees_of_freedom(self) -> int:
        """Get degrees of freedom (dimensionality)."""
        return len(self.shape)

    def update_data(self, data: np.ndarray) -> None:
        """Update tensor data."""
        if data.shape != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {data.shape}")
        
        self.data = data.astype(self.dtype)
        self.updated_at = time.time()


class TensorManager(CoreSysAttributes):
    """Manager for tensor fields and their operations."""

    def __init__(self, coresys):
        """Initialize tensor manager."""
        self.coresys = coresys
        self._tensor_fields: Dict[str, TensorField] = {}
        self._field_groups: Dict[str, List[str]] = {}  # Group name -> Field IDs
        self._visualization_data: Dict[str, Any] = {}

    def create_tensor_field(self, name: str, shape: Tuple[int, ...], dtype: str = "float32",
                           field_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                           initial_data: Optional[np.ndarray] = None) -> str:
        """Create a new tensor field."""
        if field_id is None:
            field_id = str(uuid4())
        
        if field_id in self._tensor_fields:
            raise ValueError(f"Tensor field {field_id} already exists")
        
        tensor_field = TensorField(
            field_id=field_id,
            name=name,
            shape=shape,
            dtype=dtype,
            data=initial_data,
            metadata=metadata or {},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self._tensor_fields[field_id] = tensor_field
        
        _LOGGER.info("Created tensor field: %s (%s) with shape %s", field_id, name, shape)
        return field_id

    def get_tensor_field(self, field_id: str) -> Optional[TensorField]:
        """Get tensor field by ID."""
        return self._tensor_fields.get(field_id)

    def update_tensor_field(self, field_id: str, data: np.ndarray) -> bool:
        """Update tensor field data."""
        field = self._tensor_fields.get(field_id)
        if not field:
            return False
        
        try:
            field.update_data(data)
            self._update_visualization_data(field_id)
            return True
        except ValueError as e:
            _LOGGER.error("Failed to update tensor field %s: %s", field_id, e)
            return False

    def create_device_tensor_field(self, device_id: str, device_type: str, 
                                  channels: int = 1, temporal_length: int = 100) -> str:
        """Create a tensor field for a device."""
        name = f"{device_type}_{device_id}"
        shape = (channels, temporal_length)
        
        metadata = {
            "device_id": device_id,
            "device_type": device_type,
            "channels": channels,
            "temporal_length": temporal_length,
            "field_type": "device_data"
        }
        
        return self.create_tensor_field(name, shape, metadata=metadata)

    def create_sensor_tensor_field(self, sensor_id: str, sensor_type: str,
                                  channels: int = 1, sampling_rate: float = 100.0,
                                  buffer_duration: float = 1.0) -> str:
        """Create a tensor field for sensor data."""
        name = f"sensor_{sensor_type}_{sensor_id}"
        temporal_samples = int(sampling_rate * buffer_duration)
        shape = (channels, temporal_samples)
        
        metadata = {
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "channels": channels,
            "sampling_rate": sampling_rate,
            "buffer_duration": buffer_duration,
            "field_type": "sensor_data"
        }
        
        return self.create_tensor_field(name, shape, metadata=metadata)

    def create_actuator_tensor_field(self, actuator_id: str, actuator_type: str,
                                    dof: int = 1, control_history: int = 50) -> str:
        """Create a tensor field for actuator control."""
        name = f"actuator_{actuator_type}_{actuator_id}"
        shape = (dof, control_history)
        
        metadata = {
            "actuator_id": actuator_id,
            "actuator_type": actuator_type,
            "degrees_of_freedom": dof,
            "control_history": control_history,
            "field_type": "actuator_control"
        }
        
        return self.create_tensor_field(name, shape, metadata=metadata)

    def create_agent_state_tensor_field(self, agent_id: str, state_dimension: int = 256,
                                       sequence_length: int = 1) -> str:
        """Create a tensor field for agent state."""
        name = f"agent_state_{agent_id}"
        shape = (sequence_length, state_dimension)
        
        metadata = {
            "agent_id": agent_id,
            "state_dimension": state_dimension,
            "sequence_length": sequence_length,
            "field_type": "agent_state"
        }
        
        return self.create_tensor_field(name, shape, metadata=metadata)

    def create_field_group(self, group_name: str, field_ids: List[str]) -> bool:
        """Create a group of related tensor fields."""
        # Validate all fields exist
        for field_id in field_ids:
            if field_id not in self._tensor_fields:
                _LOGGER.error("Field %s not found for group %s", field_id, group_name)
                return False
        
        self._field_groups[group_name] = field_ids
        _LOGGER.info("Created field group: %s with %d fields", group_name, len(field_ids))
        return True

    def get_field_group(self, group_name: str) -> List[TensorField]:
        """Get all tensor fields in a group."""
        field_ids = self._field_groups.get(group_name, [])
        return [self._tensor_fields[fid] for fid in field_ids if fid in self._tensor_fields]

    def get_tensor_statistics(self, field_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a tensor field."""
        field = self._tensor_fields.get(field_id)
        if not field or field.data is None:
            return None
        
        data = field.data
        return {
            "shape": field.shape,
            "dtype": field.dtype,
            "element_count": field.element_count,
            "degrees_of_freedom": field.degrees_of_freedom,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "norm": float(np.linalg.norm(data)),
            "updated_at": field.updated_at,
        }

    def get_all_tensor_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tensor fields."""
        stats = {}
        for field_id in self._tensor_fields:
            field_stats = self.get_tensor_statistics(field_id)
            if field_stats:
                stats[field_id] = field_stats
        return stats

    def _update_visualization_data(self, field_id: str) -> None:
        """Update visualization data for live monitoring."""
        field = self._tensor_fields.get(field_id)
        if not field or field.data is None:
            return
        
        # Store reduced data for visualization
        data = field.data
        
        # For 1D tensors, store full data
        if len(data.shape) == 1:
            vis_data = data.copy()
        # For 2D tensors, store means along each axis
        elif len(data.shape) == 2:
            vis_data = {
                "axis0_mean": np.mean(data, axis=0),
                "axis1_mean": np.mean(data, axis=1),
                "global_mean": np.mean(data),
            }
        # For higher dimensions, store summary statistics
        else:
            vis_data = {
                "mean": np.mean(data),
                "std": np.std(data),
                "shape": data.shape,
            }
        
        self._visualization_data[field_id] = {
            "data": vis_data,
            "timestamp": time.time(),
            "field_name": field.name,
            "shape": field.shape,
        }

    def get_visualization_data(self, field_id: Optional[str] = None) -> Dict[str, Any]:
        """Get tensor visualization data for live monitoring."""
        if field_id:
            return self._visualization_data.get(field_id, {})
        return self._visualization_data.copy()

    def compute_tensor_operations(self, operation: str, field_ids: List[str], 
                                output_field_id: Optional[str] = None) -> Optional[str]:
        """Perform tensor operations between fields."""
        if not field_ids:
            return None
        
        fields = [self._tensor_fields.get(fid) for fid in field_ids]
        if any(f is None or f.data is None for f in fields):
            _LOGGER.error("Some fields not found or have no data")
            return None
        
        try:
            if operation == "add":
                result_data = sum(f.data for f in fields)
            elif operation == "multiply":
                result_data = fields[0].data
                for f in fields[1:]:
                    result_data = result_data * f.data
            elif operation == "mean":
                result_data = np.mean([f.data for f in fields], axis=0)
            elif operation == "concatenate":
                result_data = np.concatenate([f.data for f in fields], axis=-1)
            else:
                _LOGGER.error("Unknown operation: %s", operation)
                return None
            
            # Create output field
            if output_field_id is None:
                output_field_id = f"computed_{operation}_{int(time.time())}"
            
            output_field_id = self.create_tensor_field(
                name=f"computed_{operation}",
                shape=result_data.shape,
                dtype=str(result_data.dtype),
                field_id=output_field_id,
                metadata={"operation": operation, "input_fields": field_ids},
                initial_data=result_data
            )
            
            _LOGGER.info("Computed %s operation on fields %s -> %s", 
                        operation, field_ids, output_field_id)
            return output_field_id
            
        except Exception as e:
            _LOGGER.error("Failed to compute %s operation: %s", operation, e)
            return None

    def export_tensor_configuration(self) -> Dict[str, Any]:
        """Export complete tensor configuration for GGUF/P-System compatibility."""
        config = {
            "total_fields": len(self._tensor_fields),
            "total_elements": sum(f.element_count for f in self._tensor_fields.values()),
            "total_degrees_of_freedom": sum(f.degrees_of_freedom for f in self._tensor_fields.values()),
            "field_groups": {name: len(fields) for name, fields in self._field_groups.items()},
            "fields": {}
        }
        
        for field_id, field in self._tensor_fields.items():
            config["fields"][field_id] = {
                "name": field.name,
                "shape": field.shape,
                "dtype": field.dtype,
                "element_count": field.element_count,
                "degrees_of_freedom": field.degrees_of_freedom,
                "metadata": field.metadata,
                "created_at": field.created_at,
                "updated_at": field.updated_at,
            }
        
        return config

    def list_tensor_fields(self) -> List[Dict[str, Any]]:
        """List all tensor fields with summary information."""
        fields = []
        for field_id, field in self._tensor_fields.items():
            fields.append({
                "field_id": field_id,
                "name": field.name,
                "shape": field.shape,
                "dtype": field.dtype,
                "element_count": field.element_count,
                "degrees_of_freedom": field.degrees_of_freedom,
                "metadata": field.metadata,
            })
        return fields

    def clear_all_fields(self) -> None:
        """Clear all tensor fields."""
        self._tensor_fields.clear()
        self._field_groups.clear()
        self._visualization_data.clear()
        _LOGGER.info("Cleared all tensor fields")

    def remove_tensor_field(self, field_id: str) -> bool:
        """Remove a tensor field."""
        if field_id not in self._tensor_fields:
            return False
        
        # Remove from groups
        for group_name, field_ids in self._field_groups.items():
            if field_id in field_ids:
                field_ids.remove(field_id)
        
        # Remove field
        del self._tensor_fields[field_id]
        
        # Remove visualization data
        if field_id in self._visualization_data:
            del self._visualization_data[field_id]
        
        _LOGGER.info("Removed tensor field: %s", field_id)
        return True