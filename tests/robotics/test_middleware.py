"""Test middleware component functionality."""

import pytest
import asyncio

from supervisor.robotics.middleware import (
    MiddlewareComponent, HardwareInterfaceComponent, DataProcessorComponent,
    ControllerComponent, AIAgentComponent, ComponentRegistry
)
from supervisor.robotics.hypergraph import HypergraphEngine, TensorDimensionSpec, MiddlewareInterface


class TestMiddlewareComponent:
    """Test base middleware component functionality."""

    class TestComponent(MiddlewareComponent):
        """Test implementation of middleware component."""
        
        def __init__(self):
            super().__init__("test_001", "Test Component", "test_type")
        
        def get_tensor_specification(self):
            return TensorDimensionSpec(
                degrees_of_freedom=1,
                channels=2,
                modalities=["test"]
            )
        
        def get_interface_specification(self):
            return MiddlewareInterface(
                interface_type="test_interface",
                data_format="tensor",
                communication_protocol="direct"
            )
        
        async def initialize(self):
            self._initialized = True
            return True
        
        async def start(self):
            self._active = True
            return True
        
        async def stop(self):
            self._active = False
            return True
        
        async def process_data(self, input_data):
            return f"processed_{input_data}"

    def test_component_creation(self):
        """Test basic component creation."""
        component = self.TestComponent()
        
        assert component.component_id == "test_001"
        assert component.name == "Test Component"
        assert component.component_type == "test_type"
        assert not component.is_initialized
        assert not component.is_active

    @pytest.mark.asyncio
    async def test_component_lifecycle(self):
        """Test component lifecycle methods."""
        component = self.TestComponent()
        
        # Test initialization
        success = await component.initialize()
        assert success
        assert component.is_initialized
        
        # Test start
        success = await component.start()
        assert success
        assert component.is_active
        
        # Test processing
        result = await component.process_data("test_input")
        assert result == "processed_test_input"
        
        # Test stop
        success = await component.stop()
        assert success
        assert not component.is_active

    def test_tensor_specification(self):
        """Test tensor specification."""
        component = self.TestComponent()
        tensor_spec = component.get_tensor_specification()
        
        assert tensor_spec.degrees_of_freedom == 1
        assert tensor_spec.channels == 2
        assert "test" in tensor_spec.modalities
        assert tensor_spec.get_tensor_shape() == [1, 1, 2]  # [batch, dof, channels]

    def test_interface_specification(self):
        """Test interface specification.""" 
        component = self.TestComponent()
        interface_spec = component.get_interface_specification()
        
        assert interface_spec.interface_type == "test_interface"
        assert interface_spec.data_format == "tensor"
        assert interface_spec.communication_protocol == "direct"

    def test_component_properties(self):
        """Test component properties for hypergraph registration."""
        component = self.TestComponent()
        props = component.get_component_properties()
        
        assert props["component_id"] == "test_001"
        assert props["component_type"] == "test_type"
        assert props["initialized"] == False
        assert props["active"] == False


class TestHardwareInterfaceComponent:
    """Test hardware interface component."""

    class TestSensorComponent(HardwareInterfaceComponent):
        """Test sensor component implementation."""
        
        def __init__(self):
            super().__init__("sensor_001", "Test Sensor", "accelerometer", 1, 3)
        
        async def initialize(self):
            self._initialized = True
            return True
        
        async def start(self):
            self._active = True
            return True
        
        async def stop(self):
            self._active = False
            return True
        
        async def process_data(self, input_data):
            return {"x": 0.1, "y": 0.2, "z": 9.8}

    def test_hardware_component_creation(self):
        """Test hardware interface component creation."""
        component = self.TestSensorComponent()
        
        assert component.component_id == "sensor_001"
        assert component.name == "Test Sensor"
        assert component.component_type == "hardware_interface"
        assert component.hardware_type == "accelerometer"
        assert component.degrees_of_freedom == 1
        assert component.channels == 3

    def test_hardware_tensor_specification(self):
        """Test hardware component tensor specification."""
        component = self.TestSensorComponent()
        tensor_spec = component.get_tensor_specification()
        
        assert tensor_spec.degrees_of_freedom == 1
        assert tensor_spec.channels == 3
        assert "accelerometer" in tensor_spec.modalities
        assert tensor_spec.temporal_length == 100

    def test_hardware_interface_specification(self):
        """Test hardware component interface specification."""
        component = self.TestSensorComponent()
        interface_spec = component.get_interface_specification()
        
        assert interface_spec.interface_type == "hardware_interface"
        assert interface_spec.data_format == "tensor"
        assert interface_spec.communication_protocol == "direct"


class TestDataProcessorComponent:
    """Test data processor component."""

    class TestFilterComponent(DataProcessorComponent):
        """Test filter component implementation."""
        
        def __init__(self):
            super().__init__("filter_001", "Low-Pass Filter", "signal_filter", 3, 3)
        
        async def initialize(self):
            self._initialized = True
            return True
        
        async def start(self):
            self._active = True
            return True
        
        async def stop(self):
            self._active = False
            return True
        
        async def process_data(self, input_data):
            if isinstance(input_data, dict):
                return {k: v * 0.9 for k, v in input_data.items()}
            return input_data

    def test_processor_component_creation(self):
        """Test data processor component creation."""
        component = self.TestFilterComponent()
        
        assert component.component_id == "filter_001"
        assert component.name == "Low-Pass Filter"
        assert component.component_type == "data_processor"
        assert component.processor_type == "signal_filter"
        assert component.input_channels == 3
        assert component.output_channels == 3

    def test_processor_tensor_specification(self):
        """Test processor component tensor specification."""
        component = self.TestFilterComponent()
        tensor_spec = component.get_tensor_specification()
        
        assert tensor_spec.degrees_of_freedom == 1
        assert tensor_spec.channels == 3
        assert "signal_filter" in tensor_spec.modalities
        assert "processing" in tensor_spec.modalities

    def test_processor_interface_specification(self):
        """Test processor component interface specification."""
        component = self.TestFilterComponent()
        interface_spec = component.get_interface_specification()
        
        assert interface_spec.interface_type == "data_stream"
        assert interface_spec.data_format == "tensor"
        assert interface_spec.communication_protocol == "message_queue"


class TestControllerComponent:
    """Test controller component."""

    class TestPIDComponent(ControllerComponent):
        """Test PID controller component implementation."""
        
        def __init__(self):
            super().__init__("pid_001", "PID Controller", "pid", 50.0)
        
        async def initialize(self):
            self._initialized = True
            return True
        
        async def start(self):
            self._active = True
            return True
        
        async def stop(self):
            self._active = False
            return True
        
        async def process_data(self, input_data):
            return {"control_signal": 0.5}

    def test_controller_component_creation(self):
        """Test controller component creation."""
        component = self.TestPIDComponent()
        
        assert component.component_id == "pid_001"
        assert component.name == "PID Controller"
        assert component.component_type == "controller"
        assert component.controller_type == "pid"
        assert component.control_frequency == 50.0

    def test_controller_tensor_specification(self):
        """Test controller component tensor specification."""
        component = self.TestPIDComponent()
        tensor_spec = component.get_tensor_specification()
        
        assert tensor_spec.degrees_of_freedom == 1
        assert tensor_spec.channels == 1
        assert "pid" in tensor_spec.modalities
        assert "control" in tensor_spec.modalities

    def test_controller_interface_specification(self):
        """Test controller component interface specification."""
        component = self.TestPIDComponent()
        interface_spec = component.get_interface_specification()
        
        assert interface_spec.interface_type == "control_signal"
        assert interface_spec.data_format == "tensor"
        assert interface_spec.communication_protocol == "direct"
        assert interface_spec.update_frequency == 50.0


class TestAIAgentComponent:
    """Test AI agent component."""

    class TestNeuralAgent(AIAgentComponent):
        """Test neural agent component implementation."""
        
        def __init__(self):
            super().__init__("agent_001", "Neural Agent", "reinforcement_learning", 128)
        
        async def initialize(self):
            self._initialized = True
            return True
        
        async def start(self):
            self._active = True
            return True
        
        async def stop(self):
            self._active = False
            return True
        
        async def process_data(self, input_data):
            return {"action": "move_forward", "confidence": 0.95}

    def test_agent_component_creation(self):
        """Test AI agent component creation."""
        component = self.TestNeuralAgent()
        
        assert component.component_id == "agent_001"
        assert component.name == "Neural Agent"
        assert component.component_type == "ai_agent"
        assert component.agent_type == "reinforcement_learning"
        assert component.hidden_size == 128

    def test_agent_tensor_specification(self):
        """Test agent component tensor specification."""
        component = self.TestNeuralAgent()
        tensor_spec = component.get_tensor_specification()
        
        assert tensor_spec.degrees_of_freedom == 1
        assert tensor_spec.channels == 128
        assert "cognitive" in tensor_spec.modalities
        assert "neural" in tensor_spec.modalities
        assert "symbolic" in tensor_spec.modalities

    def test_agent_interface_specification(self):
        """Test agent component interface specification."""
        component = self.TestNeuralAgent()
        interface_spec = component.get_interface_specification()
        
        assert interface_spec.interface_type == "control_signal"
        assert interface_spec.data_format == "tensor"
        assert interface_spec.communication_protocol == "message_queue"


class TestComponentRegistry:
    """Test component registry functionality."""

    @pytest.fixture
    def hypergraph(self, coresys):
        """Return hypergraph engine."""
        return HypergraphEngine(coresys)

    @pytest.fixture
    def registry(self, hypergraph):
        """Return component registry."""
        return ComponentRegistry(hypergraph)

    class SimpleTestComponent(MiddlewareComponent):
        """Simple test component for registry tests."""
        
        def __init__(self, component_id: str):
            super().__init__(component_id, f"Component {component_id}", "test")
            self.init_called = False
            self.start_called = False
            self.stop_called = False
        
        def get_tensor_specification(self):
            return TensorDimensionSpec(
                degrees_of_freedom=1,
                channels=1,
                modalities=["test"]
            )
        
        def get_interface_specification(self):
            return MiddlewareInterface(
                interface_type="test",
                data_format="tensor",
                communication_protocol="direct"
            )
        
        async def initialize(self):
            self.init_called = True
            self._initialized = True
            return True
        
        async def start(self):
            self.start_called = True
            self._active = True
            return True
        
        async def stop(self):
            self.stop_called = True
            self._active = False
            return True
        
        async def process_data(self, input_data):
            return input_data

    def test_registry_creation(self, registry, hypergraph):
        """Test registry creation."""
        assert registry.hypergraph_engine is hypergraph
        assert len(registry._components) == 0

    def test_component_registration(self, registry):
        """Test component registration."""
        component = self.SimpleTestComponent("test_001")
        
        node_id = registry.register_component(component)
        
        assert node_id == "middleware_test_001"
        assert component.component_id in registry._components
        assert component.hypergraph_node_id == node_id

    def test_component_unregistration(self, registry):
        """Test component unregistration."""
        component = self.SimpleTestComponent("test_002")
        registry.register_component(component)
        
        success = registry.unregister_component("test_002")
        
        assert success
        assert "test_002" not in registry._components

    def test_duplicate_registration(self, registry):
        """Test duplicate component registration fails."""
        component = self.SimpleTestComponent("test_003")
        registry.register_component(component)
        
        with pytest.raises(ValueError):
            registry.register_component(component)

    def test_get_component(self, registry):
        """Test getting registered component."""
        component = self.SimpleTestComponent("test_004")
        registry.register_component(component)
        
        retrieved = registry.get_component("test_004")
        assert retrieved is component

    def test_get_components_by_type(self, registry):
        """Test filtering components by type."""
        comp1 = self.SimpleTestComponent("test_005")
        comp2 = self.SimpleTestComponent("test_006")
        registry.register_component(comp1)
        registry.register_component(comp2)
        
        test_components = registry.get_components_by_type("test")
        assert len(test_components) == 2

    def test_list_components(self, registry):
        """Test listing all components."""
        component = self.SimpleTestComponent("test_007")
        registry.register_component(component)
        
        components = registry.list_components()
        assert len(components) == 1
        
        comp_info = components[0]
        assert comp_info["component_id"] == "test_007"
        assert comp_info["name"] == "Component test_007"
        assert comp_info["component_type"] == "test"

    @pytest.mark.asyncio
    async def test_initialize_all_components(self, registry):
        """Test initializing all components."""
        comp1 = self.SimpleTestComponent("test_008")
        comp2 = self.SimpleTestComponent("test_009")
        registry.register_component(comp1)
        registry.register_component(comp2)
        
        failed = await registry.initialize_all_components()
        
        assert len(failed) == 0
        assert comp1.init_called
        assert comp2.init_called
        assert comp1.is_initialized
        assert comp2.is_initialized

    @pytest.mark.asyncio
    async def test_start_all_components(self, registry):
        """Test starting all components."""
        comp1 = self.SimpleTestComponent("test_010")
        comp2 = self.SimpleTestComponent("test_011")
        registry.register_component(comp1)
        registry.register_component(comp2)
        
        await registry.initialize_all_components()
        failed = await registry.start_all_components()
        
        assert len(failed) == 0
        assert comp1.start_called
        assert comp2.start_called
        assert comp1.is_active
        assert comp2.is_active

    @pytest.mark.asyncio
    async def test_stop_all_components(self, registry):
        """Test stopping all components."""
        comp1 = self.SimpleTestComponent("test_012")
        registry.register_component(comp1)
        
        await registry.initialize_all_components()
        await registry.start_all_components()
        failed = await registry.stop_all_components()
        
        assert len(failed) == 0
        assert comp1.stop_called
        assert not comp1.is_active