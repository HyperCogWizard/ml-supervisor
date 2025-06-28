"""GGUF (GPT-Generated Unified Format) integration for agent state serialization."""

import io
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..coresys import CoreSysAttributes
from ..utils.json import write_json_file

_LOGGER: logging.Logger = logging.getLogger(__name__)

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
GGUF_VERSION = 3

class GGUFValueType:
    """GGUF value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class AgentState:
    """Represents the state of a robotics agent."""
    
    def __init__(self, agent_id: str, name: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Initialize agent state."""
        self.agent_id = agent_id
        self.name = name or f"Agent_{agent_id[:8]}"
        self.metadata = metadata or {}
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.device_configs: Dict[str, Dict[str, Any]] = {}
        self.control_loops: List[Dict[str, Any]] = []
        self.created_at = None
        self.modified_at = None

    def add_tensor(self, name: str, shape: List[int], dtype: str, data: Optional[Any] = None):
        """Add a tensor to the agent state."""
        self.tensors[name] = {
            "shape": shape,
            "dtype": dtype,
            "data": data,
            "degrees_of_freedom": len(shape),
        }
        
    def add_device_config(self, device_id: str, config: Dict[str, Any]):
        """Add device configuration."""
        self.device_configs[device_id] = config

    def add_control_loop(self, loop_config: Dict[str, Any]):
        """Add agentic control loop configuration."""
        self.control_loops.append(loop_config)


class GGUFManager(CoreSysAttributes):
    """Manager for GGUF serialization of agent states and device configurations."""

    def __init__(self, coresys):
        """Initialize GGUF manager."""
        self.coresys = coresys
        self._agent_states: Dict[str, AgentState] = {}

    def create_agent_state(self, agent_id: Optional[str] = None, name: str = "", metadata: Optional[Dict[str, Any]] = None) -> AgentState:
        """Create a new agent state."""
        if agent_id is None:
            agent_id = str(uuid4())
        
        agent_state = AgentState(agent_id, name, metadata)
        self._agent_states[agent_id] = agent_state
        
        _LOGGER.info("Created agent state: %s (%s)", agent_id, name)
        return agent_state

    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state by ID."""
        return self._agent_states.get(agent_id)

    def list_agent_states(self) -> List[str]:
        """List all agent state IDs."""
        return list(self._agent_states.keys())

    def _write_gguf_header(self, buffer: io.BytesIO, metadata: Dict[str, Any]) -> None:
        """Write GGUF header to buffer."""
        # Magic number
        buffer.write(struct.pack("<I", GGUF_MAGIC))
        # Version
        buffer.write(struct.pack("<I", GGUF_VERSION))
        # Tensor count (placeholder, will be updated)
        tensor_count_pos = buffer.tell()
        buffer.write(struct.pack("<Q", 0))
        # Metadata count
        buffer.write(struct.pack("<Q", len(metadata)))
        
        # Write metadata
        for key, value in metadata.items():
            self._write_string(buffer, key)
            self._write_value(buffer, value)
            
        return tensor_count_pos

    def _write_string(self, buffer: io.BytesIO, string: str) -> None:
        """Write string to GGUF buffer."""
        encoded = string.encode("utf-8")
        buffer.write(struct.pack("<Q", len(encoded)))
        buffer.write(encoded)

    def _write_value(self, buffer: io.BytesIO, value: Any) -> None:
        """Write value to GGUF buffer based on type."""
        if isinstance(value, bool):
            buffer.write(struct.pack("<I", GGUFValueType.BOOL))
            buffer.write(struct.pack("<?", value))
        elif isinstance(value, int):
            buffer.write(struct.pack("<I", GGUFValueType.INT64))
            buffer.write(struct.pack("<q", value))
        elif isinstance(value, float):
            buffer.write(struct.pack("<I", GGUFValueType.FLOAT64))
            buffer.write(struct.pack("<d", value))
        elif isinstance(value, str):
            buffer.write(struct.pack("<I", GGUFValueType.STRING))
            self._write_string(buffer, value)
        elif isinstance(value, list):
            buffer.write(struct.pack("<I", GGUFValueType.ARRAY))
            # Array type (assuming string array for simplicity)
            buffer.write(struct.pack("<I", GGUFValueType.STRING))
            buffer.write(struct.pack("<Q", len(value)))
            for item in value:
                self._write_string(buffer, str(item))
        else:
            # Default to string representation
            buffer.write(struct.pack("<I", GGUFValueType.STRING))
            self._write_string(buffer, str(value))

    def serialize_agent_state(self, agent_id: str) -> bytes:
        """Serialize agent state to GGUF format."""
        agent_state = self._agent_states.get(agent_id)
        if not agent_state:
            raise ValueError(f"Agent state {agent_id} not found")

        buffer = io.BytesIO()
        
        # Prepare metadata
        metadata = {
            "agent_id": agent_state.agent_id,
            "agent_name": agent_state.name,
            "tensor_count": len(agent_state.tensors),
            "device_count": len(agent_state.device_configs),
            "control_loop_count": len(agent_state.control_loops),
            "supervisor_version": "marduk-robotics-lab-1.0",
        }
        metadata.update(agent_state.metadata)

        # Write header
        tensor_count_pos = self._write_gguf_header(buffer, metadata)
        
        # Update tensor count
        current_pos = buffer.tell()
        buffer.seek(tensor_count_pos)
        buffer.write(struct.pack("<Q", len(agent_state.tensors)))
        buffer.seek(current_pos)

        # Write tensor info
        for tensor_name, tensor_info in agent_state.tensors.items():
            self._write_string(buffer, tensor_name)
            # Write dimensions
            buffer.write(struct.pack("<I", len(tensor_info["shape"])))
            for dim in tensor_info["shape"]:
                buffer.write(struct.pack("<Q", dim))
            # Write type (simplified)
            buffer.write(struct.pack("<I", GGUFValueType.FLOAT32))
            # Tensor data offset (placeholder)
            buffer.write(struct.pack("<Q", 0))

        return buffer.getvalue()

    def export_agent_state(self, agent_id: str, export_path: Path) -> None:
        """Export agent state to GGUF file."""
        gguf_data = self.serialize_agent_state(agent_id)
        
        # Create GGUF file
        gguf_path = export_path / f"agent_{agent_id}.gguf"
        with open(gguf_path, "wb") as f:
            f.write(gguf_data)
            
        # Create companion JSON with human-readable metadata
        agent_state = self._agent_states[agent_id]
        json_data = {
            "agent_id": agent_state.agent_id,
            "name": agent_state.name,
            "metadata": agent_state.metadata,
            "tensors": {name: {k: v for k, v in info.items() if k != "data"} 
                       for name, info in agent_state.tensors.items()},
            "device_configs": agent_state.device_configs,
            "control_loops": agent_state.control_loops,
        }
        
        json_path = export_path / f"agent_{agent_id}.json"
        write_json_file(json_path, json_data)
        
        _LOGGER.info("Exported agent state %s to %s", agent_id, export_path)

    def import_agent_state(self, import_path: Path) -> str:
        """Import agent state from GGUF file."""
        # For now, implement basic JSON import (full GGUF parsing would be more complex)
        json_files = list(import_path.glob("agent_*.json"))
        if not json_files:
            raise ValueError("No agent JSON files found in import path")
            
        # Import first found agent
        json_path = json_files[0]
        import json
        with open(json_path) as f:
            data = json.load(f)
            
        agent_id = data["agent_id"]
        agent_state = self.create_agent_state(agent_id, data["name"], data["metadata"])
        
        # Restore tensors
        for name, info in data["tensors"].items():
            agent_state.add_tensor(name, info["shape"], info["dtype"])
            
        # Restore device configs
        for device_id, config in data["device_configs"].items():
            agent_state.add_device_config(device_id, config)
            
        # Restore control loops
        for loop_config in data["control_loops"]:
            agent_state.add_control_loop(loop_config)
            
        _LOGGER.info("Imported agent state %s from %s", agent_id, import_path)
        return agent_id

    def get_agent_tensor_schema(self, agent_id: str) -> Dict[str, Any]:
        """Get self-descriptive tensor schema for P-System compatibility."""
        agent_state = self._agent_states.get(agent_id)
        if not agent_state:
            return {}
            
        schema = {
            "agent_id": agent_id,
            "tensor_count": len(agent_state.tensors),
            "total_parameters": sum(
                # Calculate total elements in tensor
                1 if not info["shape"] else 
                eval("*".join(str(dim) for dim in info["shape"]) or "1")
                for info in agent_state.tensors.values()
            ),
            "tensors": {}
        }
        
        for name, info in agent_state.tensors.items():
            schema["tensors"][name] = {
                "shape": info["shape"],
                "dtype": info["dtype"],
                "degrees_of_freedom": info["degrees_of_freedom"],
                "element_count": 1 if not info["shape"] else eval("*".join(str(dim) for dim in info["shape"]) or "1"),
            }
            
        return schema