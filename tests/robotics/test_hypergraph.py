"""Test hypergraph engine functionality."""

import pytest

from supervisor.robotics.hypergraph import (
    HypergraphEngine, HypergraphNode, HypergraphEdge, 
    TensorDimensionSpec, MiddlewareInterface
)


class TestHypergraphEngine:
    """Test hypergraph engine for modular workbench components."""

    @pytest.fixture
    def hypergraph(self, coresys):
        """Return hypergraph engine."""
        return HypergraphEngine(coresys)

    def test_device_node_creation(self, hypergraph):
        """Test device node creation with enhanced tensor specifications."""
        node_id = hypergraph.add_device_node(
            device_id="servo_1",
            device_type="servo_motor", 
            name="Test Servo",
            tensor_dims=[1, 100],
            modalities=["position", "velocity"],
            properties={"max_torque": 10.0},
            degrees_of_freedom=2,
            channels=3
        )
        
        assert node_id == "device_servo_1"
        
        node = hypergraph.get_node(node_id)
        assert node is not None
        assert node.node_type == "device"
        assert node.middleware_type == "hardware_interface"
        assert node.name == "Test Servo"
        assert node.tensor_dimensions == [1, 100]
        assert node.degrees_of_freedom == 2
        assert "position" in node.modalities
        assert node.properties["max_torque"] == 10.0
        
        # Test enhanced tensor specification
        assert node.tensor_spec is not None
        assert node.tensor_spec.degrees_of_freedom == 2
        assert node.tensor_spec.channels == 3
        assert "position" in node.tensor_spec.modalities
        
        # Test component interface
        assert node.component_interface is not None
        assert node.component_interface.interface_type == "device_interface"

    def test_sensor_node_creation(self, hypergraph):
        """Test sensor node creation with enhanced specifications."""
        node_id = hypergraph.add_sensor_node(
            sensor_id="imu_1",
            sensor_type="imu",
            name="9-DOF IMU",
            channels=9,
            sampling_rate=1000.0,
            tensor_dims=[9, 1000],
            temporal_length=500
        )
        
        assert node_id == "sensor_imu_1"
        
        node = hypergraph.get_node(node_id)
        assert node is not None
        assert node.node_type == "sensor"
        assert node.middleware_type == "hardware_interface"
        assert node.name == "9-DOF IMU"
        assert node.tensor_dimensions == [9, 1000]
        assert node.degrees_of_freedom == 1  # Sensors have DoF=1
        assert node.properties["channels"] == 9
        assert node.properties["sampling_rate"] == 1000.0
        assert "imu" in node.modalities
        
        # Test enhanced tensor specification
        assert node.tensor_spec is not None
        assert node.tensor_spec.channels == 9
        assert node.tensor_spec.temporal_length == 500
        
        # Test component interface
        assert node.component_interface is not None
        assert node.component_interface.interface_type == "sensor_input"
        assert node.component_interface.update_frequency == 1000.0

    def test_actuator_node_creation(self, hypergraph):
        """Test actuator node creation with enhanced specifications."""
        node_id = hypergraph.add_actuator_node(
            actuator_id="arm_1",
            actuator_type="robotic_arm",
            name="6-DOF Arm",
            dof=6,
            control_type="position",
            tensor_dims=[6, 1],
            channels=6
        )
        
        assert node_id == "actuator_arm_1"
        
        node = hypergraph.get_node(node_id)
        assert node is not None
        assert node.node_type == "actuator"
        assert node.middleware_type == "hardware_interface"
        assert node.name == "6-DOF Arm"
        assert node.tensor_dimensions == [6, 1]
        assert node.degrees_of_freedom == 6
        assert node.properties["control_type"] == "position"
        
        # Test enhanced tensor specification
        assert node.tensor_spec is not None
        assert node.tensor_spec.degrees_of_freedom == 6
        assert node.tensor_spec.channels == 6
        assert "position" in node.tensor_spec.modalities
        
        # Test component interface
        assert node.component_interface is not None
        assert node.component_interface.interface_type == "actuator_output"

    def test_agent_node_creation(self, hypergraph):
        """Test agent node creation with enhanced specifications."""
        node_id = hypergraph.add_agent_node(
            agent_id="agent_1",
            name="Neural Agent",
            agent_type="autonomous",
            state_dims=[1, 512],
            hidden_size=512
        )
        
        assert node_id == "agent_agent_1"
        
        node = hypergraph.get_node(node_id)
        assert node is not None
        assert node.node_type == "agent"
        assert node.middleware_type == "ai_agent"
        assert node.name == "Neural Agent"
        assert node.tensor_dimensions == [1, 512]
        assert "cognitive" in node.modalities
        assert "neural" in node.modalities
        assert "symbolic" in node.modalities
        
        # Test enhanced tensor specification
        assert node.tensor_spec is not None
        assert node.tensor_spec.channels == 512
        assert node.tensor_spec.degrees_of_freedom == 1
        
        # Test component interface
        assert node.component_interface is not None
        assert node.component_interface.interface_type == "control_signal"

    def test_control_loop_node_creation(self, hypergraph):
        """Test control loop node creation with enhanced specifications."""
        node_id = hypergraph.add_control_loop_node(
            loop_id="pid_1",
            name="PID Controller",
            loop_type="pid",
            update_rate=100.0
        )
        
        assert node_id == "control_pid_1"
        
        node = hypergraph.get_node(node_id)
        assert node is not None
        assert node.node_type == "control_loop"
        assert node.middleware_type == "controller"
        assert node.name == "PID Controller"
        assert node.properties["update_rate"] == 100.0
        assert node.properties["loop_type"] == "pid"
        
        # Test enhanced tensor specification
        assert node.tensor_spec is not None
        assert node.tensor_spec.degrees_of_freedom == 1
        assert node.tensor_spec.channels == 1
        assert "pid" in node.tensor_spec.modalities
        
        # Test component interface
        assert node.component_interface is not None
        assert node.component_interface.interface_type == "control_signal"
        assert node.component_interface.update_frequency == 100.0

    def test_node_connections(self, hypergraph):
        """Test connecting nodes with hyperedges."""
        # Create nodes
        sensor_id = hypergraph.add_sensor_node("cam_1", "camera", "Camera", channels=3)
        actuator_id = hypergraph.add_actuator_node("motor_1", "motor", "Motor", dof=1)
        agent_id = hypergraph.add_agent_node("agent_1", "Neural Agent")
        
        # Connect sensor to agent
        edge_id = hypergraph.connect_nodes(
            node_ids=[sensor_id, agent_id],
            edge_type="data_flow",
            name="sensor_to_agent",
            directional=True
        )
        
        assert edge_id is not None
        
        edge = hypergraph.get_edge(edge_id)
        assert edge is not None
        assert edge.edge_type == "data_flow"
        assert sensor_id in edge.nodes
        assert agent_id in edge.nodes
        assert edge.directional is True

    def test_node_neighborhood(self, hypergraph):
        """Test finding connected nodes."""
        # Create nodes
        sensor_id = hypergraph.add_sensor_node("lidar_1", "lidar", "LIDAR")
        agent_id = hypergraph.add_agent_node("agent_1", "Planning Agent")
        actuator_id = hypergraph.add_actuator_node("wheels_1", "wheels", "Drive Wheels", dof=2)
        
        # Create connections
        hypergraph.connect_nodes([sensor_id, agent_id], "data_flow")
        hypergraph.connect_nodes([agent_id, actuator_id], "control")
        
        # Test connected nodes from agent
        connected_to_agent = hypergraph.get_connected_nodes(agent_id)
        assert len(connected_to_agent) == 2
        
        connected_types = {node.node_type for node in connected_to_agent}
        assert "sensor" in connected_types
        assert "actuator" in connected_types

    def test_nodes_by_type(self, hypergraph):
        """Test filtering nodes by type."""
        # Create various nodes
        hypergraph.add_sensor_node("s1", "camera", "Camera 1")
        hypergraph.add_sensor_node("s2", "lidar", "LIDAR 1")
        hypergraph.add_actuator_node("a1", "motor", "Motor 1")
        hypergraph.add_agent_node("ag1", "Agent 1")
        hypergraph.add_control_loop_node("c1", "PID 1")
        
        # Test filtering
        sensors = hypergraph.get_nodes_by_type("sensor")
        actuators = hypergraph.get_nodes_by_type("actuator")
        agents = hypergraph.get_nodes_by_type("agent")
        controls = hypergraph.get_nodes_by_type("control_loop")
        devices = hypergraph.get_nodes_by_type("device")
        
        assert len(sensors) == 2
        assert len(actuators) == 1
        assert len(agents) == 1
        assert len(controls) == 1
        assert len(devices) == 0  # No device nodes added

    def test_hypergraph_summary(self, hypergraph):
        """Test hypergraph summary generation."""
        # Create a complex hypergraph
        hypergraph.add_sensor_node("s1", "camera", "Camera", channels=3)
        hypergraph.add_sensor_node("s2", "imu", "IMU", channels=9)
        hypergraph.add_actuator_node("a1", "motor", "Motor", dof=2)
        hypergraph.add_actuator_node("a2", "servo", "Servo", dof=1)
        hypergraph.add_agent_node("ag1", "Neural Agent")
        hypergraph.add_control_loop_node("c1", "PID Controller")
        
        # Add connections
        hypergraph.connect_nodes(["sensor_s1", "agent_ag1"], "data_flow")
        hypergraph.connect_nodes(["sensor_s2", "agent_ag1"], "data_flow")
        hypergraph.connect_nodes(["agent_ag1", "actuator_a1"], "control")
        hypergraph.connect_nodes(["control_c1", "actuator_a2"], "control")
        
        summary = hypergraph.get_hypergraph_summary()
        
        assert summary["total_nodes"] == 6
        assert summary["total_edges"] == 4
        assert summary["node_types"]["sensor"] == 2
        assert summary["node_types"]["actuator"] == 2
        assert summary["node_types"]["agent"] == 1
        assert summary["node_types"]["control_loop"] == 1
        assert summary["edge_types"]["data_flow"] == 2
        assert summary["edge_types"]["control"] == 2
        assert summary["total_degrees_of_freedom"] > 0

    def test_hypergraph_export(self, hypergraph):
        """Test complete hypergraph structure export."""
        # Create minimal hypergraph
        sensor_id = hypergraph.add_sensor_node("temp_1", "temperature", "Temperature Sensor")
        actuator_id = hypergraph.add_actuator_node("fan_1", "fan", "Cooling Fan", dof=1)
        control_id = hypergraph.add_control_loop_node("thermal_ctrl", "Thermal Controller", loop_type="pid")
        
        # Connect components
        edge_id = hypergraph.connect_nodes([sensor_id, control_id, actuator_id], "control_loop")
        
        export_data = hypergraph.export_hypergraph_structure()
        
        assert "nodes" in export_data
        assert "edges" in export_data
        assert "summary" in export_data
        
        nodes = export_data["nodes"]
        assert len(nodes) == 3
        assert sensor_id in nodes
        assert actuator_id in nodes
        assert control_id in nodes
        
        edges = export_data["edges"]
        assert len(edges) == 1
        assert edge_id in edges
        
        edge_data = edges[edge_id]
        assert edge_data["edge_type"] == "control_loop"
        assert len(edge_data["nodes"]) == 3

    def test_node_removal(self, hypergraph):
        """Test node removal and cleanup."""
        # Create connected nodes
        sensor_id = hypergraph.add_sensor_node("gyro_1", "gyroscope", "Gyroscope")
        agent_id = hypergraph.add_agent_node("stabilizer", "Stabilizer Agent")
        actuator_id = hypergraph.add_actuator_node("gimbal_1", "gimbal", "Camera Gimbal", dof=3)
        
        # Connect them
        edge1_id = hypergraph.connect_nodes([sensor_id, agent_id], "data_flow")
        edge2_id = hypergraph.connect_nodes([agent_id, actuator_id], "control")
        
        # Verify initial state
        assert len(hypergraph._nodes) == 3
        assert len(hypergraph._edges) == 2
        
        # Remove agent node (should remove connected edges)
        success = hypergraph.remove_node(agent_id)
        assert success is True
        
        # Verify cleanup
        assert len(hypergraph._nodes) == 2
        assert len(hypergraph._edges) == 0  # Both edges should be removed
        assert hypergraph.get_node(agent_id) is None
        assert hypergraph.get_edge(edge1_id) is None
        assert hypergraph.get_edge(edge2_id) is None

    def test_edge_removal(self, hypergraph):
        """Test edge removal."""
        # Create nodes and edge
        node1_id = hypergraph.add_sensor_node("proximity_1", "proximity", "Proximity Sensor")
        node2_id = hypergraph.add_actuator_node("brake_1", "brake", "Emergency Brake")
        
        edge_id = hypergraph.connect_nodes([node1_id, node2_id], "emergency_stop")
        
        # Verify initial state
        assert len(hypergraph._edges) == 1
        assert edge_id in hypergraph._node_edges[node1_id]
        assert edge_id in hypergraph._node_edges[node2_id]
        
        # Remove edge
        success = hypergraph.remove_edge(edge_id)
        assert success is True
        
        # Verify cleanup
        assert len(hypergraph._edges) == 0
        assert edge_id not in hypergraph._node_edges[node1_id]
        assert edge_id not in hypergraph._node_edges[node2_id]

    def test_clear_hypergraph(self, hypergraph):
        """Test clearing entire hypergraph."""
        # Create some nodes and edges
        hypergraph.add_sensor_node("s1", "sensor", "Sensor 1")
        hypergraph.add_actuator_node("a1", "actuator", "Actuator 1")
        hypergraph.connect_nodes(["sensor_s1", "actuator_a1"], "connection")
        
        # Verify non-empty
        assert len(hypergraph._nodes) > 0
        assert len(hypergraph._edges) > 0
        
        # Clear
        hypergraph.clear_hypergraph()
        
        # Verify empty
        assert len(hypergraph._nodes) == 0
        assert len(hypergraph._edges) == 0
        assert len(hypergraph._node_edges) == 0

    def test_complex_robotics_scenario(self, hypergraph):
        """Test complex robotics hypergraph scenario."""
        # Create humanoid robot hypergraph
        
        # Sensors
        head_camera = hypergraph.add_sensor_node("head_cam", "rgb_camera", "Head Camera", channels=3)
        chest_lidar = hypergraph.add_sensor_node("chest_lidar", "lidar", "Chest LIDAR", channels=360)
        imu = hypergraph.add_sensor_node("torso_imu", "imu", "Torso IMU", channels=9)
        left_foot_force = hypergraph.add_sensor_node("left_foot", "force", "Left Foot Force", channels=6)
        right_foot_force = hypergraph.add_sensor_node("right_foot", "force", "Right Foot Force", channels=6)
        
        # Actuators
        left_arm = hypergraph.add_actuator_node("left_arm", "arm", "Left Arm", dof=7)
        right_arm = hypergraph.add_actuator_node("right_arm", "arm", "Right Arm", dof=7)
        left_leg = hypergraph.add_actuator_node("left_leg", "leg", "Left Leg", dof=6)
        right_leg = hypergraph.add_actuator_node("right_leg", "leg", "Right Leg", dof=6)
        head = hypergraph.add_actuator_node("head", "head", "Head", dof=2)
        
        # Agents (neural modules)
        vision_agent = hypergraph.add_agent_node("vision", "Vision Processing", state_dims=[1, 512])
        planning_agent = hypergraph.add_agent_node("planner", "Motion Planner", state_dims=[1, 256])
        balance_agent = hypergraph.add_agent_node("balance", "Balance Controller", state_dims=[1, 128])
        
        # Control loops
        arm_controller = hypergraph.add_control_loop_node("arm_ctrl", "Arm Controller", loop_type="neural")
        leg_controller = hypergraph.add_control_loop_node("leg_ctrl", "Leg Controller", loop_type="hybrid")
        head_controller = hypergraph.add_control_loop_node("head_ctrl", "Head Controller", loop_type="pid")
        
        # Create data flow connections
        hypergraph.connect_nodes([head_camera, vision_agent], "visual_data")
        hypergraph.connect_nodes([chest_lidar, planning_agent], "spatial_data")
        hypergraph.connect_nodes([imu, balance_agent], "inertial_data")
        hypergraph.connect_nodes([left_foot_force, balance_agent], "force_feedback")
        hypergraph.connect_nodes([right_foot_force, balance_agent], "force_feedback")
        
        # Create cognitive connections
        hypergraph.connect_nodes([vision_agent, planning_agent], "semantic_map")
        hypergraph.connect_nodes([planning_agent, balance_agent], "motion_plan")
        
        # Create control connections
        hypergraph.connect_nodes([planning_agent, arm_controller], "arm_commands")
        hypergraph.connect_nodes([balance_agent, leg_controller], "leg_commands")
        hypergraph.connect_nodes([vision_agent, head_controller], "head_commands")
        
        hypergraph.connect_nodes([arm_controller, left_arm], "left_arm_control")
        hypergraph.connect_nodes([arm_controller, right_arm], "right_arm_control")
        hypergraph.connect_nodes([leg_controller, left_leg], "left_leg_control")
        hypergraph.connect_nodes([leg_controller, right_leg], "right_leg_control")
        hypergraph.connect_nodes([head_controller, head], "head_control")
        
        # Verify complex structure
        summary = hypergraph.get_hypergraph_summary()
        assert summary["total_nodes"] == 16  # 5 sensors + 5 actuators + 3 agents + 3 controllers
        assert summary["total_edges"] > 10
        assert summary["node_types"]["sensor"] == 5
        assert summary["node_types"]["actuator"] == 5
        assert summary["node_types"]["agent"] == 3
        assert summary["node_types"]["control_loop"] == 3
        
        # Test specific connections
        vision_connected = hypergraph.get_connected_nodes(vision_agent)
        planning_connected = hypergraph.get_connected_nodes(planning_agent)
        
        assert len(vision_connected) >= 2  # Camera input, planner output, head controller
        assert len(planning_connected) >= 3  # Vision input, LIDAR input, balance output, arm controller
        
        # Verify total degrees of freedom
        total_dof = summary["total_degrees_of_freedom"]
        expected_dof = 7 + 7 + 6 + 6 + 2  # Arms + legs + head degrees of freedom
        assert total_dof >= expected_dof

    def test_middleware_component_node(self, hypergraph):
        """Test middleware component node creation."""
        tensor_spec = TensorDimensionSpec(
            degrees_of_freedom=2,
            channels=4,
            modalities=["custom", "processing"],
            temporal_length=200
        )
        
        interface = MiddlewareInterface(
            interface_type="data_stream",
            data_format="tensor",
            communication_protocol="message_queue",
            update_frequency=50.0
        )
        
        node_id = hypergraph.add_middleware_component(
            component_id="proc_001",
            middleware_type="data_processor",
            name="Custom Processor",
            tensor_spec=tensor_spec,
            interface=interface,
            properties={"algorithm": "kalman_filter"}
        )
        
        assert node_id == "middleware_proc_001"
        
        node = hypergraph.get_node(node_id)
        assert node is not None
        assert node.node_type == "middleware"
        assert node.middleware_type == "data_processor"
        assert node.name == "Custom Processor"
        assert node.properties["algorithm"] == "kalman_filter"
        
        # Test tensor specification
        assert node.tensor_spec is not None
        assert node.tensor_spec.degrees_of_freedom == 2
        assert node.tensor_spec.channels == 4
        assert node.tensor_spec.temporal_length == 200
        assert "custom" in node.tensor_spec.modalities
        
        # Test component interface
        assert node.component_interface is not None
        assert node.component_interface.interface_type == "data_stream"
        assert node.component_interface.update_frequency == 50.0

    def test_enhanced_hypergraph_summary(self, hypergraph):
        """Test enhanced hypergraph summary with middleware types."""
        # Create various middleware components
        hypergraph.add_sensor_node("s1", "camera", "Camera", channels=3)
        hypergraph.add_actuator_node("a1", "motor", "Motor", dof=2)
        hypergraph.add_agent_node("ag1", "Neural Agent")
        hypergraph.add_control_loop_node("c1", "PID Controller")
        
        # Add custom middleware
        tensor_spec = TensorDimensionSpec(degrees_of_freedom=1, channels=2, modalities=["custom"])
        interface = MiddlewareInterface("data_stream", "tensor", "direct")
        hypergraph.add_middleware_component("m1", "data_processor", "Processor", tensor_spec, interface)
        
        summary = hypergraph.get_hypergraph_summary()
        
        assert summary["total_nodes"] == 5
        assert "middleware_types" in summary
        assert summary["middleware_types"]["hardware_interface"] == 2  # sensor + actuator
        assert summary["middleware_types"]["ai_agent"] == 1
        assert summary["middleware_types"]["controller"] == 1
        assert summary["middleware_types"]["data_processor"] == 1
        assert "total_channels" in summary
        assert summary["total_channels"] > 0

    def test_enhanced_export_structure(self, hypergraph):
        """Test enhanced export with tensor specs and interfaces."""
        # Create a node with full specifications
        sensor_id = hypergraph.add_sensor_node(
            "advanced_sensor", "lidar", "Advanced LIDAR",
            channels=360, sampling_rate=20.0, temporal_length=1000
        )
        
        export_data = hypergraph.export_hypergraph_structure()
        
        assert "nodes" in export_data
        sensor_data = export_data["nodes"][sensor_id]
        
        # Test enhanced export fields
        assert "middleware_type" in sensor_data
        assert sensor_data["middleware_type"] == "hardware_interface"
        
        assert "tensor_spec" in sensor_data
        tensor_spec_data = sensor_data["tensor_spec"]
        assert tensor_spec_data["channels"] == 360
        assert tensor_spec_data["temporal_length"] == 1000
        assert tensor_spec_data["tensor_shape"] is not None
        
        assert "component_interface" in sensor_data
        interface_data = sensor_data["component_interface"]
        assert interface_data["interface_type"] == "sensor_input"
        assert interface_data["update_frequency"] == 20.0