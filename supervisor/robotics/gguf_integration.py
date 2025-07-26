"""GGUF (GPT-Generated Unified Format) integration for agent state serialization."""

import io
import logging
import struct
import time
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

    def create_agent_state_from_tensor_manager(self, tensor_manager, agent_id: Optional[str] = None, 
                                             name: str = "", metadata: Optional[Dict[str, Any]] = None) -> AgentState:
        """Create agent state from tensor manager data."""
        if agent_id is None:
            agent_id = str(uuid4())
        
        agent_state = AgentState(agent_id, name or f"TensorManager_Agent_{agent_id[:8]}", metadata)
        
        # Import all tensor fields from tensor manager
        for field_id, field in tensor_manager._tensor_fields.items():
            agent_state.add_tensor(
                name=field.name,
                shape=list(field.shape),
                dtype=field.dtype,
                data=field.data
            )
            
            # Add field metadata as device config if it represents a device
            if field.metadata.get("field_type") in ["device_data", "sensor_data", "actuator_control"]:
                device_id = field.metadata.get("device_id") or field.metadata.get("sensor_id") or field.metadata.get("actuator_id")
                if device_id:
                    agent_state.add_device_config(device_id, {
                        "field_id": field_id,
                        "field_type": field.metadata.get("field_type"),
                        "tensor_name": field.name,
                        **field.metadata
                    })
        
        # Import tensor field groups as control loops
        for group_name, field_ids in tensor_manager._field_groups.items():
            agent_state.add_control_loop({
                "loop_id": f"tensor_group_{group_name}",
                "type": "tensor_group",
                "field_ids": field_ids,
                "field_count": len(field_ids)
            })
        
        self._agent_states[agent_id] = agent_state
        _LOGGER.info("Created agent state from tensor manager: %s with %d tensors", 
                    agent_id, len(agent_state.tensors))
        return agent_state

    def create_agent_state_from_kernelizer(self, kernelizer, agent_id: Optional[str] = None,
                                         name: str = "", metadata: Optional[Dict[str, Any]] = None) -> AgentState:
        """Create agent state from HomeAssistant kernelizer data.""" 
        if agent_id is None:
            agent_id = str(uuid4())
            
        agent_state = AgentState(agent_id, name or f"Kernelizer_Agent_{agent_id[:8]}", metadata)
        
        # Import agentic automations as control loops
        for auto_id, automation in kernelizer._agentic_automations.items():
            agent_state.add_control_loop({
                "loop_id": auto_id,
                "type": "agentic_automation", 
                "name": automation.name,
                "entity_ids": automation.entity_ids,
                "execution_count": automation.execution_count,
                "active": automation.active,
                "control_logic": automation.control_logic
            })
            
        # Import entity tensor mappings as device configs
        for entity_id, field_id in kernelizer._entity_tensor_mappings.items():
            domain = entity_id.split(".")[0]
            entity_config = kernelizer._get_entity_tensor_config(domain, entity_id)
            
            agent_state.add_device_config(entity_id, {
                "device_type": f"homeassistant_{domain}",
                "tensor_field_id": field_id,
                "entity_id": entity_id,
                "domain": domain,
                **entity_config
            })
            
        # Import active control loops
        for auto_id, loop_info in kernelizer._active_control_loops.items():
            agent_state.add_control_loop({
                "loop_id": loop_info["loop_id"],
                "type": "neural_symbolic_control",
                "automation_id": auto_id,
                "experiment_id": loop_info["experiment_id"],
                **loop_info["config"]
            })
        
        self._agent_states[agent_id] = agent_state
        _LOGGER.info("Created agent state from kernelizer: %s with %d automations", 
                    agent_id, len(kernelizer._agentic_automations))
        return agent_state

    def create_comprehensive_agent_state(self, workbench, agent_id: Optional[str] = None,
                                       name: str = "", metadata: Optional[Dict[str, Any]] = None) -> AgentState:
        """Create comprehensive agent state from entire workbench."""
        if agent_id is None:
            agent_id = str(uuid4())
            
        # Start with tensor manager data
        agent_state = self.create_agent_state_from_tensor_manager(
            workbench.tensor_manager, agent_id, name or f"Workbench_Agent_{agent_id[:8]}", metadata
        )
        
        # Merge kernelizer data 
        kernelizer_data = self.create_agent_state_from_kernelizer(workbench.ha_kernelizer)
        
        # Merge device configs (avoid duplicates)
        for device_id, config in kernelizer_data.device_configs.items():
            if device_id not in agent_state.device_configs:
                agent_state.add_device_config(device_id, config)
            else:
                # Merge configs
                agent_state.device_configs[device_id].update(config)
                
        # Add kernelizer control loops
        for loop_config in kernelizer_data.control_loops:
            agent_state.add_control_loop(loop_config)
            
        # Add workbench experiment information  
        if workbench._experiments:
            for exp_id, exp_data in workbench._experiments.items():
                agent_state.add_control_loop({
                    "loop_id": f"experiment_{exp_id}",
                    "type": "workbench_experiment",
                    "experiment_id": exp_id,
                    **exp_data
                })
        
        # Update metadata with comprehensive info
        agent_state.metadata.update({
            "comprehensive_capture": True,
            "tensor_manager_fields": len(workbench.tensor_manager._tensor_fields),
            "kernelizer_automations": len(workbench.ha_kernelizer._agentic_automations),
            "workbench_experiments": len(workbench._experiments),
            "capture_timestamp": time.time()
        })
        
        _LOGGER.info("Created comprehensive agent state: %s with %d tensors, %d devices, %d control loops",
                    agent_id, len(agent_state.tensors), len(agent_state.device_configs), 
                    len(agent_state.control_loops))
        return agent_state
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
        """Serialize agent state to GGUF format with actual tensor data."""
        agent_state = self._agent_states.get(agent_id)
        if not agent_state:
            raise ValueError(f"Agent state {agent_id} not found")

        buffer = io.BytesIO()
        
        # Prepare comprehensive metadata including device configs and control loops
        metadata = {
            "agent_id": agent_state.agent_id,
            "agent_name": agent_state.name,
            "tensor_count": len(agent_state.tensors),
            "device_count": len(agent_state.device_configs),
            "control_loop_count": len(agent_state.control_loops),
            "supervisor_version": "marduk-robotics-lab-1.0",
            "gguf_version": GGUF_VERSION,
        }
        metadata.update(agent_state.metadata)
        
        # Add device configurations to metadata
        if agent_state.device_configs:
            metadata["device_configurations"] = agent_state.device_configs
            
        # Add control loops to metadata  
        if agent_state.control_loops:
            metadata["control_loops"] = agent_state.control_loops

        # Write header
        tensor_count_pos = self._write_gguf_header(buffer, metadata)
        
        # Update tensor count
        current_pos = buffer.tell()
        buffer.seek(tensor_count_pos)
        buffer.write(struct.pack("<Q", len(agent_state.tensors)))
        buffer.seek(current_pos)

        # Collect tensor data offsets to write later
        tensor_data_offsets = []
        
        # Write tensor info (metadata section)
        for tensor_name, tensor_info in agent_state.tensors.items():
            self._write_string(buffer, tensor_name)
            # Write dimensions
            buffer.write(struct.pack("<I", len(tensor_info["shape"])))
            for dim in tensor_info["shape"]:
                buffer.write(struct.pack("<Q", dim))
            # Write type based on actual dtype
            gguf_type = self._get_gguf_type_for_dtype(tensor_info["dtype"])
            buffer.write(struct.pack("<I", gguf_type))
            # Store current position for data offset (will update later)
            offset_pos = buffer.tell()
            buffer.write(struct.pack("<Q", 0))  # Placeholder offset
            tensor_data_offsets.append((offset_pos, tensor_name, tensor_info))

        # Now write actual tensor data and update offsets
        for offset_pos, tensor_name, tensor_info in tensor_data_offsets:
            # Mark current position as data start
            data_start = buffer.tell()
            
            # Write tensor data if available
            if tensor_info.get("data") is not None:
                import numpy as np
                data = tensor_info["data"]
                if isinstance(data, np.ndarray):
                    # Ensure data matches expected shape and type
                    if data.shape != tuple(tensor_info["shape"]):
                        _LOGGER.warning(
                            "Tensor %s shape mismatch: expected %s, got %s", 
                            tensor_name, tensor_info["shape"], data.shape
                        )
                        # Reshape or pad/truncate as needed
                        data = self._reshape_tensor_data(data, tensor_info["shape"])
                    
                    # Convert to expected dtype
                    data = data.astype(tensor_info["dtype"])
                    
                    # Write raw tensor data
                    buffer.write(data.tobytes())
                else:
                    # Convert non-numpy data to numpy array
                    import numpy as np
                    data = np.array(data, dtype=tensor_info["dtype"])
                    data = data.reshape(tensor_info["shape"])
                    buffer.write(data.tobytes())
            else:
                # Write zeros for missing data
                import numpy as np
                shape = tuple(tensor_info["shape"]) if tensor_info["shape"] else (1,)
                zeros = np.zeros(shape, dtype=tensor_info["dtype"])
                buffer.write(zeros.tobytes())
            
            # Update offset in header
            current_pos = buffer.tell()
            buffer.seek(offset_pos)
            buffer.write(struct.pack("<Q", data_start))
            buffer.seek(current_pos)

        return buffer.getvalue()

    def _get_gguf_type_for_dtype(self, dtype: str) -> int:
        """Map numpy dtype to GGUF type."""
        dtype_map = {
            "uint8": GGUFValueType.UINT8,
            "int8": GGUFValueType.INT8,
            "uint16": GGUFValueType.UINT16,
            "int16": GGUFValueType.INT16,
            "uint32": GGUFValueType.UINT32,
            "int32": GGUFValueType.INT32,
            "uint64": GGUFValueType.UINT64,
            "int64": GGUFValueType.INT64,
            "float32": GGUFValueType.FLOAT32,
            "float64": GGUFValueType.FLOAT64,
            "bool": GGUFValueType.BOOL,
        }
        return dtype_map.get(dtype, GGUFValueType.FLOAT32)  # Default to float32

    def _reshape_tensor_data(self, data, target_shape):
        """Safely reshape tensor data to target shape."""
        import numpy as np
        
        target_size = np.prod(target_shape) if target_shape else 1
        current_size = data.size
        
        if current_size == target_size:
            # Simple reshape
            return data.reshape(target_shape)
        elif current_size > target_size:
            # Truncate data
            _LOGGER.warning("Truncating tensor data from %d to %d elements", current_size, target_size)
            return data.flat[:target_size].reshape(target_shape)
        else:
            # Pad with zeros
            _LOGGER.warning("Padding tensor data from %d to %d elements", current_size, target_size)
            padded = np.zeros(target_size, dtype=data.dtype)
            padded[:current_size] = data.flat
            return padded.reshape(target_shape)
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

    def export_as_p_system_membrane(self, agent_id: str, export_path: Path, 
                                   membrane_name: Optional[str] = None) -> Dict[str, Any]:
        """Export agent state as P-System membrane with nested GGUF structure."""
        agent_state = self._agent_states.get(agent_id)
        if not agent_state:
            raise ValueError(f"Agent state {agent_id} not found")
            
        membrane_name = membrane_name or f"membrane_{agent_id}"
        membrane_dir = export_path / membrane_name
        membrane_dir.mkdir(parents=True, exist_ok=True)
        
        # Create membrane manifest
        membrane_manifest = {
            "membrane_id": membrane_name,
            "agent_id": agent_id,
            "membrane_type": "p_system_agent_membrane",
            "created_at": time.time(),
            "components": {},
            "sub_membranes": {},
            "tensor_files": {},
            "device_configs": agent_state.device_configs,
            "control_loops": agent_state.control_loops
        }
        
        # Export main agent GGUF
        main_gguf_path = membrane_dir / f"{agent_id}_main.gguf"
        with open(main_gguf_path, "wb") as f:
            f.write(self.serialize_agent_state(agent_id))
        membrane_manifest["components"]["main_agent"] = str(main_gguf_path.name)
        
        # Create sub-membranes for device groups
        device_groups = self._group_devices_by_type(agent_state.device_configs)
        for device_type, devices in device_groups.items():
            if len(devices) > 1:  # Only create sub-membrane if multiple devices
                sub_membrane_dir = membrane_dir / f"submembrane_{device_type}"
                sub_membrane_dir.mkdir(exist_ok=True)
                
                # Create sub-agent for this device group
                sub_agent_id = f"{agent_id}_{device_type}_group"
                sub_agent = self.create_agent_state(
                    agent_id=sub_agent_id,
                    name=f"{device_type.title()} Device Group",
                    metadata={"parent_agent": agent_id, "device_type": device_type}
                )
                
                # Add tensors and configs for devices in this group
                for device_id in devices:
                    device_config = agent_state.device_configs[device_id]
                    sub_agent.add_device_config(device_id, device_config)
                    
                    # Find related tensors
                    related_tensor = self._find_tensor_for_device(agent_state, device_id)
                    if related_tensor:
                        tensor_name, tensor_info = related_tensor
                        sub_agent.add_tensor(tensor_name, tensor_info["shape"], 
                                           tensor_info["dtype"], tensor_info.get("data"))
                
                # Export sub-membrane GGUF
                sub_gguf_path = sub_membrane_dir / f"{sub_agent_id}.gguf"
                with open(sub_gguf_path, "wb") as f:
                    f.write(self.serialize_agent_state(sub_agent_id))
                
                membrane_manifest["sub_membranes"][device_type] = {
                    "path": str(sub_membrane_dir.name),
                    "agent_id": sub_agent_id,
                    "device_count": len(devices),
                    "gguf_file": str(sub_gguf_path.name)
                }
        
        # Export individual tensor files for large tensors
        for tensor_name, tensor_info in agent_state.tensors.items():
            if tensor_info.get("data") is not None:
                import numpy as np
                data = tensor_info["data"]
                if isinstance(data, np.ndarray) and data.nbytes > 1024 * 1024:  # > 1MB
                    tensor_file = membrane_dir / f"tensor_{tensor_name}.bin"
                    with open(tensor_file, "wb") as f:
                        f.write(data.tobytes())
                    membrane_manifest["tensor_files"][tensor_name] = {
                        "file": str(tensor_file.name),
                        "shape": tensor_info["shape"],
                        "dtype": tensor_info["dtype"],
                        "size_bytes": data.nbytes
                    }
        
        # Write membrane manifest
        manifest_path = membrane_dir / "membrane_manifest.json"
        from ..utils.json import write_json_file
        write_json_file(manifest_path, membrane_manifest)
        
        _LOGGER.info("Exported P-System membrane: %s with %d sub-membranes",
                    membrane_name, len(membrane_manifest["sub_membranes"]))
        return membrane_manifest

    def _group_devices_by_type(self, device_configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group devices by their type for sub-membrane creation."""
        groups = {}
        for device_id, config in device_configs.items():
            device_type = config.get("device_type", "unknown")
            if device_type not in groups:
                groups[device_type] = []
            groups[device_type].append(device_id)
        return groups

    def _find_tensor_for_device(self, agent_state: AgentState, device_id: str) -> Optional[tuple]:
        """Find tensor associated with a device."""
        for tensor_name, tensor_info in agent_state.tensors.items():
            # Check if tensor name contains device ID or if metadata links them
            if device_id in tensor_name or tensor_name.replace("_", ".") == device_id:
                return (tensor_name, tensor_info)
        return None

    def import_p_system_membrane(self, membrane_path: Path) -> str:
        """Import P-System membrane and reconstruct agent state."""
        manifest_path = membrane_path / "membrane_manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Membrane manifest not found at {manifest_path}")
            
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
            
        agent_id = manifest["agent_id"]
        membrane_id = manifest["membrane_id"]
        
        # Import main agent GGUF
        main_gguf_file = manifest["components"]["main_agent"]
        main_gguf_path = membrane_path / main_gguf_file
        
        if main_gguf_path.exists():
            # For now, use JSON import (full GGUF parsing would be more complex)
            json_files = list(membrane_path.glob("*.json"))
            if json_files:
                imported_agent_id = self.import_agent_state(membrane_path)
                agent_state = self.get_agent_state(imported_agent_id)
            else:
                # Create basic agent state from manifest
                agent_state = self.create_agent_state(
                    agent_id=agent_id,
                    name=f"Imported {membrane_id}",
                    metadata={"imported_from_membrane": membrane_id}
                )
        else:
            raise ValueError(f"Main GGUF file not found: {main_gguf_path}")
            
        # Import sub-membranes
        for sub_type, sub_info in manifest.get("sub_membranes", {}).items():
            sub_membrane_path = membrane_path / sub_info["path"]
            if sub_membrane_path.exists():
                try:
                    sub_agent_id = self.import_p_system_membrane(sub_membrane_path)
                    sub_agent = self.get_agent_state(sub_agent_id)
                    if sub_agent:
                        # Merge sub-agent data into main agent
                        for device_id, config in sub_agent.device_configs.items():
                            agent_state.add_device_config(device_id, config)
                        for loop_config in sub_agent.control_loops:
                            agent_state.add_control_loop(loop_config)
                except Exception as e:
                    _LOGGER.warning("Failed to import sub-membrane %s: %s", sub_type, e)
        
        # Import individual tensor files
        for tensor_name, tensor_info in manifest.get("tensor_files", {}).items():
            tensor_file = membrane_path / tensor_info["file"]
            if tensor_file.exists():
                import numpy as np
                data = np.frombuffer(tensor_file.read_bytes(), dtype=tensor_info["dtype"])
                data = data.reshape(tensor_info["shape"])
                agent_state.add_tensor(tensor_name, tensor_info["shape"], 
                                     tensor_info["dtype"], data)
        
        _LOGGER.info("Imported P-System membrane: %s as agent %s", membrane_id, agent_id)
        return agent_id

    def export_all_environment_tensors(self, workbench, export_path: Path) -> Dict[str, Any]:
        """Export all environment tensors as GGUF files."""
        export_path.mkdir(parents=True, exist_ok=True)
        
        export_manifest = {
            "export_type": "environment_tensors",
            "export_timestamp": time.time(),
            "tensor_files": {},
            "tensor_statistics": workbench.tensor_manager.get_all_tensor_statistics()
        }
        
        # Export each tensor field as separate GGUF
        for field_id, field in workbench.tensor_manager._tensor_fields.items():
            # Create mini agent state for this tensor
            tensor_agent = self.create_agent_state(
                agent_id=f"tensor_{field_id}",
                name=f"Environment Tensor {field.name}",
                metadata={"field_id": field_id, "export_type": "environment_tensor"}
            )
            
            tensor_agent.add_tensor(
                name=field.name,
                shape=list(field.shape),
                dtype=field.dtype,
                data=field.data
            )
            
            # Export as GGUF
            gguf_file = export_path / f"tensor_{field_id}.gguf"
            with open(gguf_file, "wb") as f:
                f.write(self.serialize_agent_state(tensor_agent.agent_id))
            
            export_manifest["tensor_files"][field_id] = {
                "file": str(gguf_file.name),
                "tensor_name": field.name,
                "shape": list(field.shape),
                "dtype": field.dtype,
                "metadata": field.metadata
            }
        
        # Write export manifest
        manifest_path = export_path / "environment_tensors_manifest.json"
        from ..utils.json import write_json_file
        write_json_file(manifest_path, export_manifest)
        
        _LOGGER.info("Exported %d environment tensors to %s", 
                    len(export_manifest["tensor_files"]), export_path)
        return export_manifest