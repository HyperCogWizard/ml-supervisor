"""Test robotics workbench functionality."""

import pytest
import numpy as np
from pathlib import Path

from supervisor.robotics.workbench import RoboticsWorkbench
from supervisor.robotics.gguf_integration import GGUFManager, AgentState
from supervisor.robotics.hypergraph import HypergraphEngine, HypergraphNode
from supervisor.robotics.tensor_manager import TensorManager
from supervisor.robotics.middleware import HardwareInterfaceComponent, ComponentRegistry


class TestRoboticsWorkbench:
    """Test robotics workbench core functionality."""

    @pytest.fixture
    def workbench(self, coresys):
        """Return robotics workbench."""
        return RoboticsWorkbench(coresys)

    @pytest.mark.asyncio
    async def test_workbench_initialization(self, workbench):
        """Test workbench initialization."""
        await workbench.initialize()
        
        assert workbench.gguf_manager is not None
        assert workbench.hypergraph_engine is not None
        assert workbench.tensor_manager is not None
        assert "default" in workbench._experiments

    @pytest.mark.asyncio
    async def test_experiment_creation(self, workbench):
        """Test experiment creation."""
        await workbench.initialize()
        
        success = await workbench.create_experiment(
            experiment_id="test_exp",
            name="Test Experiment",
            description="Test Description"
        )
        
        assert success is True
        assert "test_exp" in workbench._experiments
        
        experiment = workbench._experiments["test_exp"]
        assert experiment["name"] == "Test Experiment"
        assert experiment["description"] == "Test Description"
        assert len(experiment["agent_ids"]) > 0  # Should have default agent

    @pytest.mark.asyncio
    async def test_device_configuration(self, workbench):
        """Test device configuration."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        success = await workbench.configure_device(
            experiment_id="test_exp",
            device_id="servo_1",
            device_type="servo_motor",
            name="Test Servo",
            config={
                "tensor_dimensions": [3, 100],
                "modalities": ["position", "velocity"],
                "channels": 3
            }
        )
        
        assert success is True
        
        # Check hypergraph node was created
        nodes = workbench.hypergraph_engine.get_nodes_by_type("device")
        device_nodes = [n for n in nodes if "servo_1" in n.properties.get("device_id", "")]
        assert len(device_nodes) > 0
        
        # Check tensor field was created
        tensor_fields = workbench.tensor_manager.list_tensor_fields()
        servo_fields = [f for f in tensor_fields if "servo_1" in f["name"]]
        assert len(servo_fields) > 0

    @pytest.mark.asyncio
    async def test_sensor_configuration(self, workbench):
        """Test sensor configuration."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        success = await workbench.configure_sensor(
            experiment_id="test_exp",
            sensor_id="imu_1",
            sensor_type="imu",
            name="Test IMU",
            config={
                "channels": 9,  # 3-axis accel, gyro, mag
                "sampling_rate": 1000.0,
                "buffer_duration": 2.0
            }
        )
        
        assert success is True
        
        # Check sensor node
        nodes = workbench.hypergraph_engine.get_nodes_by_type("sensor")
        sensor_nodes = [n for n in nodes if "imu_1" in n.properties.get("sensor_id", "")]
        assert len(sensor_nodes) > 0
        
        sensor_node = sensor_nodes[0]
        assert sensor_node.properties["channels"] == 9
        assert sensor_node.properties["sampling_rate"] == 1000.0

    @pytest.mark.asyncio
    async def test_actuator_configuration(self, workbench):
        """Test actuator configuration."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        success = await workbench.configure_actuator(
            experiment_id="test_exp",
            actuator_id="arm_1",
            actuator_type="robotic_arm",
            name="Test Robotic Arm",
            config={
                "degrees_of_freedom": 6,
                "control_type": "position",
                "control_history": 100
            }
        )
        
        assert success is True
        
        # Check actuator node
        nodes = workbench.hypergraph_engine.get_nodes_by_type("actuator")
        actuator_nodes = [n for n in nodes if "arm_1" in n.properties.get("actuator_id", "")]
        assert len(actuator_nodes) > 0
        
        actuator_node = actuator_nodes[0]
        assert actuator_node.degrees_of_freedom == 6
        assert actuator_node.properties["control_type"] == "position"

    @pytest.mark.asyncio
    async def test_agentic_control_loop(self, workbench):
        """Test agentic control loop creation."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        success = await workbench.create_agentic_control_loop(
            experiment_id="test_exp",
            loop_id="neural_controller_1",
            name="Neural Symbolic Controller",
            config={
                "loop_type": "neural_symbolic",
                "update_rate": 100.0,
                "properties": {"neural_network": "transformer", "symbolic_rules": True}
            }
        )
        
        assert success is True
        
        # Check control loop node
        nodes = workbench.hypergraph_engine.get_nodes_by_type("control_loop")
        control_nodes = [n for n in nodes if "neural_controller_1" in n.properties.get("loop_id", "")]
        assert len(control_nodes) > 0
        
        # Check agent state was updated
        experiment = workbench._experiments["test_exp"]
        agent_id = experiment["agent_ids"][0]
        agent_state = workbench.gguf_manager.get_agent_state(agent_id)
        assert len(agent_state.control_loops) > 0
        
        control_loop = agent_state.control_loops[0]
        assert control_loop["loop_id"] == "neural_controller_1"

    @pytest.mark.asyncio
    async def test_component_connections(self, workbench):
        """Test component connections in hypergraph."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        # Add components
        await workbench.configure_sensor("test_exp", "sensor_1", "camera", "Camera", {})
        await workbench.configure_actuator("test_exp", "actuator_1", "motor", "Motor", {})
        
        # Get node IDs
        sensor_nodes = workbench.hypergraph_engine.get_nodes_by_type("sensor")
        actuator_nodes = workbench.hypergraph_engine.get_nodes_by_type("actuator")
        
        sensor_id = sensor_nodes[0].node_id
        actuator_id = actuator_nodes[0].node_id
        
        # Connect components
        success = await workbench.connect_components(
            experiment_id="test_exp",
            source_ids=[sensor_id],
            target_ids=[actuator_id],
            connection_type="neural_connection"
        )
        
        assert success is True
        
        # Check connection exists
        connected_nodes = workbench.hypergraph_engine.get_connected_nodes(sensor_id)
        connected_actuators = [n for n in connected_nodes if n.node_type == "actuator"]
        assert len(connected_actuators) > 0

    @pytest.mark.asyncio
    async def test_homeassistant_kernelization(self, workbench):
        """Test HomeAssistant entity kernelization."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        success = await workbench.kernelize_homeassistant_entity(
            entity_id="light.living_room",
            experiment_id="test_exp"
        )
        
        assert success is True
        
        # Check HA entity was converted to tensor node
        experiment = workbench._experiments["test_exp"]
        assert "device_light.living_room" in [node for node in experiment["device_nodes"] if "light.living_room" in node]
        
        # Check tensor field was created
        tensor_fields = workbench.tensor_manager.list_tensor_fields()
        ha_fields = [f for f in tensor_fields if "light.living_room" in f["name"]]
        assert len(ha_fields) > 0

    @pytest.mark.asyncio
    async def test_experiment_monitoring(self, workbench):
        """Test experiment monitoring."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        # Start experiment
        success = await workbench.start_experiment("test_exp")
        assert success is True
        assert workbench._active_experiment == "test_exp"
        assert workbench._monitoring_active is True
        
        # Stop experiment
        success = await workbench.stop_experiment("test_exp")
        assert success is True
        assert workbench._active_experiment is None
        assert workbench._monitoring_active is False

    @pytest.mark.asyncio
    async def test_live_visualization_data(self, workbench):
        """Test live visualization data retrieval."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        await workbench.configure_sensor("test_exp", "sensor_1", "lidar", "LIDAR", {"channels": 360})
        
        viz_data = workbench.get_live_visualization_data()
        
        assert "tensor_fields" in viz_data
        assert "hypergraph_summary" in viz_data
        assert "active_experiment" in viz_data
        assert "timestamp" in viz_data
        
        hypergraph_summary = viz_data["hypergraph_summary"]
        assert hypergraph_summary["total_nodes"] > 0
        assert "sensor" in hypergraph_summary["node_types"]

    def test_experiment_status(self, workbench):
        """Test experiment status retrieval."""
        # This is sync since we're just testing data structures
        workbench._experiments["test_exp"] = {
            "experiment_id": "test_exp",
            "name": "Test",
            "description": "",
            "metadata": {},
            "agent_ids": ["agent_1"],
            "device_nodes": ["node_1"],
            "tensor_fields": ["field_1"],
            "status": "created"
        }
        
        status = workbench.get_experiment_status("test_exp")
        assert status is not None
        assert status["experiment"]["experiment_id"] == "test_exp"
        assert "hypergraph_summary" in status
        assert "tensor_statistics" in status

    def test_experiments_list(self, workbench):
        """Test experiment listing."""
        # Add test experiments
        workbench._experiments = {
            "exp1": {
                "experiment_id": "exp1", "name": "Exp 1", "description": "First", 
                "status": "created", "device_nodes": [], "tensor_fields": [], "agent_ids": []
            },
            "exp2": {
                "experiment_id": "exp2", "name": "Exp 2", "description": "Second",
                "status": "running", "device_nodes": ["n1"], "tensor_fields": ["f1"], "agent_ids": ["a1"]
            }
        }
        
        experiments = workbench.list_experiments()
        assert len(experiments) == 2
        
        exp1 = next(e for e in experiments if e["experiment_id"] == "exp1")
        assert exp1["name"] == "Exp 1"
        assert exp1["status"] == "created"
        assert exp1["device_count"] == 0
        
        exp2 = next(e for e in experiments if e["experiment_id"] == "exp2")
        assert exp2["name"] == "Exp 2"
        assert exp2["status"] == "running"
        assert exp2["device_count"] == 1


class TestEnhancedWorkbenchMiddleware:
    """Test enhanced workbench middleware functionality."""

    @pytest.fixture
    def workbench(self, coresys):
        """Return robotics workbench."""
        return RoboticsWorkbench(coresys)

    class TestMiddlewareComponent(HardwareInterfaceComponent):
        """Test middleware component for workbench tests."""
        
        def __init__(self, component_id: str, name: str):
            super().__init__(component_id, name, "test_hardware", 2, 3)
        
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
            return {"processed": True, "input": input_data}

    @pytest.mark.asyncio
    async def test_workbench_component_registry(self, workbench):
        """Test workbench component registry initialization."""
        await workbench.initialize()
        
        assert workbench.component_registry is not None
        assert isinstance(workbench.component_registry, ComponentRegistry)

    @pytest.mark.asyncio
    async def test_middleware_component_registration(self, workbench):
        """Test registering middleware components with workbench."""
        await workbench.initialize()
        
        component = self.TestMiddlewareComponent("test_comp_001", "Test Component")
        node_id = await workbench.register_middleware_component(component)
        
        assert node_id == "middleware_test_comp_001"
        assert component.component_id in [c["component_id"] for c in workbench.get_registered_components()]

    @pytest.mark.asyncio
    async def test_middleware_component_unregistration(self, workbench):
        """Test unregistering middleware components."""
        await workbench.initialize()
        
        component = self.TestMiddlewareComponent("test_comp_002", "Test Component 2")
        await workbench.register_middleware_component(component)
        
        success = await workbench.unregister_middleware_component("test_comp_002")
        
        assert success
        component_ids = [c["component_id"] for c in workbench.get_registered_components()]
        assert "test_comp_002" not in component_ids

    @pytest.mark.asyncio
    async def test_middleware_lifecycle_management(self, workbench):
        """Test middleware component lifecycle management."""
        await workbench.initialize()
        
        # Register multiple components
        comp1 = self.TestMiddlewareComponent("comp_001", "Component 1")
        comp2 = self.TestMiddlewareComponent("comp_002", "Component 2")
        
        await workbench.register_middleware_component(comp1)
        await workbench.register_middleware_component(comp2)
        
        # Test initialization
        success = await workbench.initialize_middleware_components()
        assert success
        assert comp1.is_initialized
        assert comp2.is_initialized
        
        # Test starting
        success = await workbench.start_middleware_components()
        assert success
        assert comp1.is_active
        assert comp2.is_active
        
        # Test stopping
        success = await workbench.stop_middleware_components()
        assert success
        assert not comp1.is_active
        assert not comp2.is_active

    @pytest.mark.asyncio
    async def test_device_as_middleware_configuration(self, workbench):
        """Test configuring devices as middleware components."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        config = {
            "degrees_of_freedom": 2,
            "channels": 3,
            "temporal_length": 200,
            "modalities": ["position", "velocity"]
        }
        
        success = await workbench.configure_device_as_middleware(
            experiment_id="test_exp",
            device_id="servo_001",
            device_type="servo_motor",
            name="Test Servo",
            config=config
        )
        
        assert success
        
        # Verify component was registered
        components = workbench.get_registered_components()
        device_components = [c for c in components if c["component_id"] == "servo_001"]
        assert len(device_components) == 1

    @pytest.mark.asyncio
    async def test_sensor_as_middleware_configuration(self, workbench):
        """Test configuring sensors as middleware components."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        config = {
            "channels": 9,
            "sampling_rate": 1000.0,
            "buffer_duration": 2.0
        }
        
        success = await workbench.configure_sensor_as_middleware(
            experiment_id="test_exp",
            sensor_id="imu_001",
            sensor_type="imu",
            name="Test IMU",
            config=config
        )
        
        assert success
        
        # Verify component was registered
        components = workbench.get_registered_components()
        sensor_components = [c for c in components if c["component_id"] == "imu_001"]
        assert len(sensor_components) == 1

    @pytest.mark.asyncio
    async def test_actuator_as_middleware_configuration(self, workbench):
        """Test configuring actuators as middleware components."""
        await workbench.initialize()
        await workbench.create_experiment("test_exp", "Test Experiment")
        
        config = {
            "degrees_of_freedom": 6,
            "channels": 6,
            "control_type": "position",
            "control_history": 100
        }
        
        success = await workbench.configure_actuator_as_middleware(
            experiment_id="test_exp",
            actuator_id="arm_001",
            actuator_type="robotic_arm",
            name="Test Arm",
            config=config
        )
        
        assert success
        
        # Verify component was registered
        components = workbench.get_registered_components()
        actuator_components = [c for c in components if c["component_id"] == "arm_001"]
        assert len(actuator_components) == 1

    @pytest.mark.asyncio
    async def test_complete_middleware_workflow(self, workbench):
        """Test complete middleware workflow with multiple components."""
        await workbench.initialize()
        await workbench.create_experiment("complete_test", "Complete Test")
        
        # Configure sensor
        sensor_config = {"channels": 3, "sampling_rate": 100.0}
        await workbench.configure_sensor_as_middleware(
            "complete_test", "camera_001", "rgb_camera", "Test Camera", sensor_config
        )
        
        # Configure actuator
        actuator_config = {"degrees_of_freedom": 2, "control_type": "velocity"}
        await workbench.configure_actuator_as_middleware(
            "complete_test", "motors_001", "drive_motors", "Drive Motors", actuator_config
        )
        
        # Register custom middleware
        processor = self.TestMiddlewareComponent("processor_001", "Data Processor")
        await workbench.register_middleware_component(processor)
        
        # Initialize and start all components
        await workbench.initialize_middleware_components()
        await workbench.start_middleware_components()
        
        # Verify all components are active
        components = workbench.get_registered_components()
        active_components = [c for c in components if c["active"]]
        assert len(active_components) == 3  # sensor, actuator, processor
        
        # Test hypergraph integration
        summary = workbench.hypergraph_engine.get_hypergraph_summary()
        assert summary["total_nodes"] >= 3
        assert "hardware_interface" in summary["middleware_types"]
        
        # Stop all components
        await workbench.stop_middleware_components()
        
        # Verify all components are stopped
        components = workbench.get_registered_components()
        active_components = [c for c in components if c["active"]]
        assert len(active_components) == 0