"""Main Robotics Engineering Workbench for Marduk's Robotics Lab."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..coresys import CoreSysAttributes
from ..homeassistant.module import HomeAssistant
from .gguf_integration import AgentState, GGUFManager
from .hypergraph import HypergraphEngine
from .tensor_manager import TensorManager

_LOGGER: logging.Logger = logging.getLogger(__name__)


class RoboticsWorkbench(CoreSysAttributes):
    """Main robotics engineering workbench integrating all components."""

    def __init__(self, coresys):
        """Initialize robotics workbench."""
        self.coresys = coresys
        
        # Initialize core components
        self.gguf_manager = GGUFManager(coresys)
        self.hypergraph_engine = HypergraphEngine(coresys)
        self.tensor_manager = TensorManager(coresys)
        
        # Workbench state
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._active_experiment: Optional[str] = None
        self._monitoring_active: bool = False

    async def initialize(self) -> None:
        """Initialize the robotics workbench."""
        _LOGGER.info("Initializing Marduk's Robotics Lab workbench")
        
        # Create default experiment
        await self.create_experiment("default", "Default Robotics Experiment")
        
        _LOGGER.info("Robotics workbench initialized successfully")

    async def create_experiment(self, experiment_id: str, name: str, 
                               description: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new robotics experiment."""
        if experiment_id in self._experiments:
            _LOGGER.error("Experiment %s already exists", experiment_id)
            return False
        
        experiment = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "metadata": metadata or {},
            "created_at": None,
            "agent_ids": [],
            "device_nodes": [],
            "tensor_fields": [],
            "status": "created"
        }
        
        self._experiments[experiment_id] = experiment
        
        # Create default agent for this experiment
        agent_state = self.gguf_manager.create_agent_state(
            name=f"Agent_{experiment_id}",
            metadata={"experiment_id": experiment_id}
        )
        experiment["agent_ids"].append(agent_state.agent_id)
        
        _LOGGER.info("Created experiment: %s (%s)", experiment_id, name)
        return True

    async def configure_device(self, experiment_id: str, device_id: str, 
                              device_type: str, name: str, config: Dict[str, Any]) -> bool:
        """Configure a device for an experiment."""
        if experiment_id not in self._experiments:
            _LOGGER.error("Experiment %s not found", experiment_id)
            return False
        
        experiment = self._experiments[experiment_id]
        
        # Create hypergraph node for device
        tensor_dims = config.get("tensor_dimensions", [1, 100])
        modalities = config.get("modalities", [device_type])
        
        node_id = self.hypergraph_engine.add_device_node(
            device_id=device_id,
            device_type=device_type,
            name=name,
            tensor_dims=tensor_dims,
            modalities=modalities,
            properties=config
        )
        
        experiment["device_nodes"].append(node_id)
        
        # Create tensor field for device
        field_id = self.tensor_manager.create_device_tensor_field(
            device_id=device_id,
            device_type=device_type,
            channels=config.get("channels", 1),
            temporal_length=config.get("temporal_length", 100)
        )
        
        experiment["tensor_fields"].append(field_id)
        
        _LOGGER.info("Configured device %s (%s) for experiment %s", device_id, name, experiment_id)
        return True

    async def configure_sensor(self, experiment_id: str, sensor_id: str,
                              sensor_type: str, name: str, config: Dict[str, Any]) -> bool:
        """Configure a sensor for an experiment."""
        if experiment_id not in self._experiments:
            return False
        
        experiment = self._experiments[experiment_id]
        
        # Create hypergraph node for sensor
        channels = config.get("channels", 1)
        sampling_rate = config.get("sampling_rate", 100.0)
        
        node_id = self.hypergraph_engine.add_sensor_node(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            name=name,
            channels=channels,
            sampling_rate=sampling_rate,
            properties=config
        )
        
        experiment["device_nodes"].append(node_id)
        
        # Create tensor field for sensor
        field_id = self.tensor_manager.create_sensor_tensor_field(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            channels=channels,
            sampling_rate=sampling_rate,
            buffer_duration=config.get("buffer_duration", 1.0)
        )
        
        experiment["tensor_fields"].append(field_id)
        
        _LOGGER.info("Configured sensor %s (%s) for experiment %s", sensor_id, name, experiment_id)
        return True

    async def configure_actuator(self, experiment_id: str, actuator_id: str,
                                actuator_type: str, name: str, config: Dict[str, Any]) -> bool:
        """Configure an actuator for an experiment."""
        if experiment_id not in self._experiments:
            return False
        
        experiment = self._experiments[experiment_id]
        
        # Create hypergraph node for actuator
        dof = config.get("degrees_of_freedom", 1)
        control_type = config.get("control_type", "position")
        
        node_id = self.hypergraph_engine.add_actuator_node(
            actuator_id=actuator_id,
            actuator_type=actuator_type,
            name=name,
            dof=dof,
            control_type=control_type,
            properties=config
        )
        
        experiment["device_nodes"].append(node_id)
        
        # Create tensor field for actuator
        field_id = self.tensor_manager.create_actuator_tensor_field(
            actuator_id=actuator_id,
            actuator_type=actuator_type,
            dof=dof,
            control_history=config.get("control_history", 50)
        )
        
        experiment["tensor_fields"].append(field_id)
        
        _LOGGER.info("Configured actuator %s (%s) for experiment %s", actuator_id, name, experiment_id)
        return True

    async def create_agentic_control_loop(self, experiment_id: str, loop_id: str,
                                         name: str, config: Dict[str, Any]) -> bool:
        """Create an agentic control loop replacing static automation."""
        if experiment_id not in self._experiments:
            return False
        
        experiment = self._experiments[experiment_id]
        
        # Create control loop node
        loop_type = config.get("loop_type", "neural_symbolic")
        update_rate = config.get("update_rate", 10.0)
        
        node_id = self.hypergraph_engine.add_control_loop_node(
            loop_id=loop_id,
            name=name,
            loop_type=loop_type,
            update_rate=update_rate,
            properties=config
        )
        
        experiment["device_nodes"].append(node_id)
        
        # Add control loop to agent state
        if experiment["agent_ids"]:
            agent_id = experiment["agent_ids"][0]
            agent_state = self.gguf_manager.get_agent_state(agent_id)
            if agent_state:
                agent_state.add_control_loop({
                    "loop_id": loop_id,
                    "name": name,
                    "config": config,
                    "node_id": node_id
                })
        
        _LOGGER.info("Created agentic control loop %s for experiment %s", loop_id, experiment_id)
        return True

    async def connect_components(self, experiment_id: str, source_ids: List[str], 
                                target_ids: List[str], connection_type: str = "data_flow") -> bool:
        """Connect components in the hypergraph."""
        if experiment_id not in self._experiments:
            return False
        
        all_node_ids = source_ids + target_ids
        edge_id = self.hypergraph_engine.connect_nodes(
            node_ids=all_node_ids,
            edge_type=connection_type,
            name=f"{connection_type}_connection",
            directional=True
        )
        
        _LOGGER.info("Connected components %s -> %s with edge %s", source_ids, target_ids, edge_id)
        return True

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start a robotics experiment."""
        if experiment_id not in self._experiments:
            return False
        
        experiment = self._experiments[experiment_id]
        experiment["status"] = "running"
        self._active_experiment = experiment_id
        
        # Start monitoring
        await self._start_monitoring()
        
        _LOGGER.info("Started experiment: %s", experiment_id)
        return True

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a robotics experiment."""
        if experiment_id not in self._experiments:
            return False
        
        experiment = self._experiments[experiment_id]
        experiment["status"] = "stopped"
        
        if self._active_experiment == experiment_id:
            self._active_experiment = None
            await self._stop_monitoring()
        
        _LOGGER.info("Stopped experiment: %s", experiment_id)
        return True

    async def _start_monitoring(self) -> None:
        """Start live tensor field monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        asyncio.create_task(self._monitoring_loop())

    async def _stop_monitoring(self) -> None:
        """Stop live tensor field monitoring."""
        self._monitoring_active = False

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for live tensor visualization."""
        while self._monitoring_active:
            try:
                # Update visualization data for all tensor fields
                for field_id in self.tensor_manager._tensor_fields:
                    field = self.tensor_manager.get_tensor_field(field_id)
                    if field and field.data is not None:
                        self.tensor_manager._update_visualization_data(field_id)
                
                # Sleep for update interval
                await asyncio.sleep(0.1)  # 10 Hz update rate
                
            except Exception as e:
                _LOGGER.error("Error in monitoring loop: %s", e)
                await asyncio.sleep(1.0)

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an experiment."""
        if experiment_id not in self._experiments:
            return None
        
        experiment = self._experiments[experiment_id]
        
        # Get hypergraph summary
        hypergraph_summary = self.hypergraph_engine.get_hypergraph_summary()
        
        # Get tensor statistics
        tensor_stats = self.tensor_manager.get_all_tensor_statistics()
        
        return {
            "experiment": experiment,
            "hypergraph_summary": hypergraph_summary,
            "tensor_statistics": tensor_stats,
            "active": experiment_id == self._active_experiment,
            "monitoring_active": self._monitoring_active,
        }

    def get_live_visualization_data(self) -> Dict[str, Any]:
        """Get live tensor field visualization data."""
        return {
            "tensor_fields": self.tensor_manager.get_visualization_data(),
            "hypergraph_summary": self.hypergraph_engine.get_hypergraph_summary(),
            "active_experiment": self._active_experiment,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def export_experiment(self, experiment_id: str, export_path: Path) -> bool:
        """Export complete experiment configuration as GGUF/P-System schema."""
        if experiment_id not in self._experiments:
            return False
        
        experiment = self._experiments[experiment_id]
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export agent states to GGUF
        for agent_id in experiment["agent_ids"]:
            try:
                self.gguf_manager.export_agent_state(agent_id, export_path)
            except Exception as e:
                _LOGGER.error("Failed to export agent %s: %s", agent_id, e)
        
        # Export hypergraph structure
        hypergraph_data = self.hypergraph_engine.export_hypergraph_structure()
        from ..utils.json import write_json_file
        write_json_file(export_path / "hypergraph.json", hypergraph_data)
        
        # Export tensor configuration
        tensor_config = self.tensor_manager.export_tensor_configuration()
        write_json_file(export_path / "tensor_config.json", tensor_config)
        
        # Export experiment metadata
        experiment_data = {
            "experiment": experiment,
            "export_timestamp": asyncio.get_event_loop().time(),
            "workbench_version": "marduk-robotics-lab-1.0"
        }
        write_json_file(export_path / "experiment.json", experiment_data)
        
        _LOGGER.info("Exported experiment %s to %s", experiment_id, export_path)
        return True

    async def import_experiment(self, import_path: Path) -> Optional[str]:
        """Import experiment from exported data."""
        import_path = Path(import_path)
        
        if not import_path.exists():
            _LOGGER.error("Import path does not exist: %s", import_path)
            return None
        
        try:
            # Import experiment metadata
            from ..utils.json import read_json_file
            experiment_data = read_json_file(import_path / "experiment.json")
            experiment = experiment_data["experiment"]
            
            experiment_id = experiment["experiment_id"]
            
            # Create experiment
            await self.create_experiment(
                experiment_id=experiment_id,
                name=experiment["name"],
                description=experiment["description"],
                metadata=experiment["metadata"]
            )
            
            # Import agent states
            for agent_id in experiment["agent_ids"]:
                try:
                    imported_agent_id = self.gguf_manager.import_agent_state(import_path)
                    if imported_agent_id != agent_id:
                        _LOGGER.warning("Agent ID mismatch during import: %s != %s", imported_agent_id, agent_id)
                except Exception as e:
                    _LOGGER.error("Failed to import agent %s: %s", agent_id, e)
            
            _LOGGER.info("Imported experiment %s from %s", experiment_id, import_path)
            return experiment_id
            
        except Exception as e:
            _LOGGER.error("Failed to import experiment: %s", e)
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = []
        for exp_id, exp_data in self._experiments.items():
            experiments.append({
                "experiment_id": exp_id,
                "name": exp_data["name"],
                "description": exp_data["description"],
                "status": exp_data["status"],
                "device_count": len(exp_data["device_nodes"]),
                "tensor_field_count": len(exp_data["tensor_fields"]),
                "agent_count": len(exp_data["agent_ids"]),
            })
        return experiments

    async def kernelize_homeassistant_entity(self, entity_id: str, experiment_id: str) -> bool:
        """Convert HomeAssistant entity to agentic tensor node."""
        if experiment_id not in self._experiments:
            return False
        
        # Create tensor node for HomeAssistant entity
        node_id = self.hypergraph_engine.add_device_node(
            device_id=entity_id,
            device_type="homeassistant_entity",
            name=f"HA_Entity_{entity_id}",
            tensor_dims=[1, 1],  # Simple scalar value
            properties={
                "entity_id": entity_id,
                "homeassistant_integration": True,
                "agentic_control": True
            }
        )
        
        # Create tensor field for entity state
        field_id = self.tensor_manager.create_tensor_field(
            name=f"ha_entity_{entity_id}",
            shape=(1, 100),  # [state_value, history]
            metadata={
                "entity_id": entity_id,
                "homeassistant_entity": True
            }
        )
        
        experiment = self._experiments[experiment_id]
        experiment["device_nodes"].append(node_id)
        experiment["tensor_fields"].append(field_id)
        
        _LOGGER.info("Kernelized HomeAssistant entity %s for experiment %s", entity_id, experiment_id)
        return True