"""Abstract base classes and interfaces for composable robotics middleware components."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .hypergraph import HypergraphEngine, TensorDimensionSpec, MiddlewareInterface

_LOGGER: logging.Logger = logging.getLogger(__name__)


class MiddlewareComponent(ABC):
    """Abstract base class for all middleware components in the robotics workbench."""
    
    def __init__(self, component_id: str, name: str, component_type: str):
        """Initialize middleware component."""
        self.component_id = component_id
        self.name = name
        self.component_type = component_type
        self.hypergraph_node_id: Optional[str] = None
        self._initialized = False
        self._active = False
    
    @abstractmethod
    def get_tensor_specification(self) -> TensorDimensionSpec:
        """Get the tensor dimension specification for this component."""
        pass
    
    @abstractmethod
    def get_interface_specification(self) -> MiddlewareInterface:
        """Get the interface specification for this component."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the middleware component."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the middleware component operation."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the middleware component operation."""
        pass
    
    @abstractmethod
    async def process_data(self, input_data: Any) -> Any:
        """Process input data and return output."""
        pass
    
    def register_with_hypergraph(self, hypergraph_engine: HypergraphEngine) -> str:
        """Register this component as a node in the hypergraph."""
        tensor_spec = self.get_tensor_specification()
        interface_spec = self.get_interface_specification()
        
        self.hypergraph_node_id = hypergraph_engine.add_middleware_component(
            component_id=self.component_id,
            middleware_type=self.component_type,
            name=self.name,
            tensor_spec=tensor_spec,
            interface=interface_spec,
            properties=self.get_component_properties()
        )
        
        _LOGGER.info("Registered component %s with hypergraph as %s", 
                    self.component_id, self.hypergraph_node_id)
        return self.hypergraph_node_id
    
    def get_component_properties(self) -> Dict[str, Any]:
        """Get component-specific properties for hypergraph registration."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "initialized": self._initialized,
            "active": self._active,
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized
    
    @property
    def is_active(self) -> bool:
        """Check if component is active."""
        return self._active


class HardwareInterfaceComponent(MiddlewareComponent):
    """Base class for hardware interface components (sensors, actuators, devices)."""
    
    def __init__(self, component_id: str, name: str, hardware_type: str, 
                 degrees_of_freedom: int = 1, channels: int = 1):
        """Initialize hardware interface component."""
        super().__init__(component_id, name, "hardware_interface")
        self.hardware_type = hardware_type
        self.degrees_of_freedom = degrees_of_freedom
        self.channels = channels
    
    def get_tensor_specification(self) -> TensorDimensionSpec:
        """Get tensor specification for hardware interface."""
        return TensorDimensionSpec(
            degrees_of_freedom=self.degrees_of_freedom,
            channels=self.channels,
            modalities=[self.hardware_type],
            temporal_length=100  # Default buffer size
        )
    
    def get_interface_specification(self) -> MiddlewareInterface:
        """Get interface specification for hardware interface."""
        return MiddlewareInterface(
            interface_type="hardware_interface",
            data_format="tensor",
            communication_protocol="direct"
        )


class DataProcessorComponent(MiddlewareComponent):
    """Base class for data processing components."""
    
    def __init__(self, component_id: str, name: str, processor_type: str,
                 input_channels: int = 1, output_channels: int = 1):
        """Initialize data processor component."""
        super().__init__(component_id, name, "data_processor")
        self.processor_type = processor_type
        self.input_channels = input_channels
        self.output_channels = output_channels
    
    def get_tensor_specification(self) -> TensorDimensionSpec:
        """Get tensor specification for data processor."""
        return TensorDimensionSpec(
            degrees_of_freedom=1,  # Processing transforms, DoF=1
            channels=max(self.input_channels, self.output_channels),
            modalities=[self.processor_type, "processing"],
            temporal_length=50  # Processing buffer
        )
    
    def get_interface_specification(self) -> MiddlewareInterface:
        """Get interface specification for data processor."""
        return MiddlewareInterface(
            interface_type="data_stream",
            data_format="tensor",
            communication_protocol="message_queue"
        )


class ControllerComponent(MiddlewareComponent):
    """Base class for controller components."""
    
    def __init__(self, component_id: str, name: str, controller_type: str,
                 control_frequency: float = 10.0):
        """Initialize controller component."""
        super().__init__(component_id, name, "controller")
        self.controller_type = controller_type
        self.control_frequency = control_frequency
    
    def get_tensor_specification(self) -> TensorDimensionSpec:
        """Get tensor specification for controller."""
        return TensorDimensionSpec(
            degrees_of_freedom=1,  # Control signal DoF
            channels=1,  # Single control channel
            modalities=[self.controller_type, "control"],
            temporal_length=100  # Control history
        )
    
    def get_interface_specification(self) -> MiddlewareInterface:
        """Get interface specification for controller."""
        return MiddlewareInterface(
            interface_type="control_signal",
            data_format="tensor",
            communication_protocol="direct",
            update_frequency=self.control_frequency
        )


class AIAgentComponent(MiddlewareComponent):
    """Base class for AI agent components."""
    
    def __init__(self, component_id: str, name: str, agent_type: str,
                 hidden_size: int = 256):
        """Initialize AI agent component."""
        super().__init__(component_id, name, "ai_agent")
        self.agent_type = agent_type
        self.hidden_size = hidden_size
    
    def get_tensor_specification(self) -> TensorDimensionSpec:
        """Get tensor specification for AI agent."""
        return TensorDimensionSpec(
            degrees_of_freedom=1,  # Agent state DoF
            channels=self.hidden_size,
            modalities=["cognitive", "neural", "symbolic"],
            temporal_length=None  # Variable length sequences
        )
    
    def get_interface_specification(self) -> MiddlewareInterface:
        """Get interface specification for AI agent."""
        return MiddlewareInterface(
            interface_type="control_signal",
            data_format="tensor",
            communication_protocol="message_queue"
        )


class ComponentRegistry:
    """Registry for managing middleware components in the robotics workbench."""
    
    def __init__(self, hypergraph_engine: HypergraphEngine):
        """Initialize component registry."""
        self.hypergraph_engine = hypergraph_engine
        self._components: Dict[str, MiddlewareComponent] = {}
        self._component_nodes: Dict[str, str] = {}  # component_id -> node_id
    
    def register_component(self, component: MiddlewareComponent) -> str:
        """Register a middleware component."""
        if component.component_id in self._components:
            raise ValueError(f"Component {component.component_id} already registered")
        
        # Register with hypergraph
        node_id = component.register_with_hypergraph(self.hypergraph_engine)
        
        # Store in registry
        self._components[component.component_id] = component
        self._component_nodes[component.component_id] = node_id
        
        _LOGGER.info("Registered component %s in registry", component.component_id)
        return node_id
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a middleware component."""
        if component_id not in self._components:
            return False
        
        # Remove from hypergraph if node exists
        if component_id in self._component_nodes:
            node_id = self._component_nodes[component_id]
            self.hypergraph_engine.remove_node(node_id)
            del self._component_nodes[component_id]
        
        # Remove from registry
        del self._components[component_id]
        
        _LOGGER.info("Unregistered component %s from registry", component_id)
        return True
    
    def get_component(self, component_id: str) -> Optional[MiddlewareComponent]:
        """Get a registered component."""
        return self._components.get(component_id)
    
    def get_components_by_type(self, component_type: str) -> List[MiddlewareComponent]:
        """Get all components of a specific type."""
        return [comp for comp in self._components.values() 
                if comp.component_type == component_type]
    
    def list_components(self) -> List[Dict[str, Any]]:
        """List all registered components."""
        components = []
        for comp_id, component in self._components.items():
            components.append({
                "component_id": comp_id,
                "name": component.name,
                "component_type": component.component_type,
                "hypergraph_node_id": self._component_nodes.get(comp_id),
                "initialized": component.is_initialized,
                "active": component.is_active,
            })
        return components
    
    async def initialize_all_components(self) -> List[str]:
        """Initialize all registered components."""
        failed_components = []
        for comp_id, component in self._components.items():
            try:
                success = await component.initialize()
                if not success:
                    failed_components.append(comp_id)
            except Exception as e:
                _LOGGER.error("Failed to initialize component %s: %s", comp_id, e)
                failed_components.append(comp_id)
        
        return failed_components
    
    async def start_all_components(self) -> List[str]:
        """Start all initialized components."""
        failed_components = []
        for comp_id, component in self._components.items():
            if not component.is_initialized:
                continue
            
            try:
                success = await component.start()
                if not success:
                    failed_components.append(comp_id)
            except Exception as e:
                _LOGGER.error("Failed to start component %s: %s", comp_id, e)
                failed_components.append(comp_id)
        
        return failed_components
    
    async def stop_all_components(self) -> List[str]:
        """Stop all active components."""
        failed_components = []
        for comp_id, component in self._components.items():
            if not component.is_active:
                continue
            
            try:
                success = await component.stop()
                if not success:
                    failed_components.append(comp_id)
            except Exception as e:
                _LOGGER.error("Failed to stop component %s: %s", comp_id, e)
                failed_components.append(comp_id)
        
        return failed_components