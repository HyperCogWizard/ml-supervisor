"""Test GGUF integration functionality."""

import pytest
import tempfile
from pathlib import Path

from supervisor.robotics.gguf_integration import GGUFManager, AgentState


class TestGGUFIntegration:
    """Test GGUF integration for agent state serialization."""

    @pytest.fixture
    def gguf_manager(self, coresys):
        """Return GGUF manager."""
        return GGUFManager(coresys)

    def test_agent_state_creation(self, gguf_manager):
        """Test agent state creation."""
        agent_state = gguf_manager.create_agent_state(
            name="Test Agent",
            metadata={"experiment": "test", "version": "1.0"}
        )
        
        assert agent_state.name == "Test Agent"
        assert agent_state.metadata["experiment"] == "test"
        assert agent_state.agent_id in gguf_manager._agent_states

    def test_agent_state_tensors(self, gguf_manager):
        """Test adding tensors to agent state."""
        agent_state = gguf_manager.create_agent_state(name="Test Agent")
        
        # Add various tensor types
        agent_state.add_tensor("position", [3], "float32")
        agent_state.add_tensor("velocity", [3], "float32")
        agent_state.add_tensor("state_history", [100, 256], "float32")
        agent_state.add_tensor("action_logits", [10], "float32")
        
        assert len(agent_state.tensors) == 4
        assert agent_state.tensors["position"]["shape"] == [3]
        assert agent_state.tensors["position"]["degrees_of_freedom"] == 1
        assert agent_state.tensors["state_history"]["degrees_of_freedom"] == 2

    def test_agent_state_device_configs(self, gguf_manager):
        """Test adding device configurations."""
        agent_state = gguf_manager.create_agent_state(name="Test Agent")
        
        # Add device configurations
        agent_state.add_device_config("servo_1", {
            "type": "servo_motor",
            "min_angle": -180,
            "max_angle": 180,
            "max_speed": 100
        })
        
        agent_state.add_device_config("camera_1", {
            "type": "rgb_camera",
            "resolution": [1920, 1080],
            "fps": 30,
            "encoding": "h264"
        })
        
        assert len(agent_state.device_configs) == 2
        assert agent_state.device_configs["servo_1"]["type"] == "servo_motor"
        assert agent_state.device_configs["camera_1"]["resolution"] == [1920, 1080]

    def test_agent_state_control_loops(self, gguf_manager):
        """Test adding control loops."""
        agent_state = gguf_manager.create_agent_state(name="Test Agent")
        
        # Add control loops
        agent_state.add_control_loop({
            "loop_id": "position_controller",
            "type": "pid",
            "parameters": {"kp": 1.0, "ki": 0.1, "kd": 0.01},
            "update_rate": 100.0
        })
        
        agent_state.add_control_loop({
            "loop_id": "neural_planner", 
            "type": "neural_symbolic",
            "model": "transformer",
            "hidden_size": 256
        })
        
        assert len(agent_state.control_loops) == 2
        assert agent_state.control_loops[0]["loop_id"] == "position_controller"
        assert agent_state.control_loops[1]["type"] == "neural_symbolic"

    def test_gguf_serialization(self, gguf_manager):
        """Test GGUF serialization of agent state."""
        agent_state = gguf_manager.create_agent_state(name="Test Agent")
        
        # Add tensors and configs
        agent_state.add_tensor("position", [3], "float32")
        agent_state.add_tensor("velocity", [3], "float32")
        agent_state.add_device_config("servo_1", {"type": "servo", "dof": 1})
        agent_state.add_control_loop({"loop_id": "test_loop", "type": "pid"})
        
        # Serialize to GGUF
        gguf_data = gguf_manager.serialize_agent_state(agent_state.agent_id)
        
        assert isinstance(gguf_data, bytes)
        assert len(gguf_data) > 0
        
        # Check GGUF magic number at start
        import struct
        magic = struct.unpack("<I", gguf_data[:4])[0]
        assert magic == 0x46554747  # "GGUF" in little endian

    def test_agent_state_export_import(self, gguf_manager):
        """Test export and import of agent states."""
        # Create agent with data
        agent_state = gguf_manager.create_agent_state(
            name="Export Test Agent",
            metadata={"test": True}
        )
        agent_state.add_tensor("test_tensor", [5, 10], "float32")
        agent_state.add_device_config("test_device", {"type": "test", "param": 42})
        agent_state.add_control_loop({"loop_id": "test_loop", "rate": 50})
        
        original_agent_id = agent_state.agent_id
        
        # Export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir)
            gguf_manager.export_agent_state(original_agent_id, export_path)
            
            # Check files were created
            gguf_file = export_path / f"agent_{original_agent_id}.gguf"
            json_file = export_path / f"agent_{original_agent_id}.json"
            assert gguf_file.exists()
            assert json_file.exists()
            
            # Clear manager and import
            gguf_manager._agent_states.clear()
            imported_agent_id = gguf_manager.import_agent_state(export_path)
            
            # Verify imported agent
            assert imported_agent_id == original_agent_id
            imported_agent = gguf_manager.get_agent_state(imported_agent_id)
            assert imported_agent is not None
            assert imported_agent.name == "Export Test Agent"
            assert imported_agent.metadata["test"] is True
            assert len(imported_agent.tensors) == 1
            assert len(imported_agent.device_configs) == 1
            assert len(imported_agent.control_loops) == 1

    def test_tensor_schema_generation(self, gguf_manager):
        """Test P-System compatible tensor schema generation."""
        agent_state = gguf_manager.create_agent_state(name="Schema Test Agent")
        
        # Add tensors with different shapes and complexities
        agent_state.add_tensor("scalar", [], "float32")  # Scalar
        agent_state.add_tensor("vector", [3], "float32")  # 3D vector
        agent_state.add_tensor("matrix", [4, 4], "float32")  # 4x4 matrix
        agent_state.add_tensor("sequence", [100, 256], "float32")  # Sequence
        agent_state.add_tensor("batch_data", [32, 10, 128], "float32")  # Batched sequence
        
        schema = gguf_manager.get_agent_tensor_schema(agent_state.agent_id)
        
        assert schema["agent_id"] == agent_state.agent_id
        assert schema["tensor_count"] == 5
        assert schema["total_parameters"] == 1 + 3 + 16 + 25600 + 40960  # Sum of all elements
        
        tensors = schema["tensors"]
        assert tensors["scalar"]["degrees_of_freedom"] == 0
        assert tensors["scalar"]["element_count"] == 1
        assert tensors["vector"]["degrees_of_freedom"] == 1
        assert tensors["vector"]["element_count"] == 3
        assert tensors["matrix"]["degrees_of_freedom"] == 2
        assert tensors["matrix"]["element_count"] == 16
        assert tensors["sequence"]["element_count"] == 25600
        assert tensors["batch_data"]["element_count"] == 40960

    def test_agent_list_and_retrieval(self, gguf_manager):
        """Test listing and retrieving agent states."""
        # Create multiple agents
        agent1 = gguf_manager.create_agent_state(name="Agent 1")
        agent2 = gguf_manager.create_agent_state(name="Agent 2")
        agent3 = gguf_manager.create_agent_state(name="Agent 3")
        
        # Test listing
        agent_ids = gguf_manager.list_agent_states()
        assert len(agent_ids) == 3
        assert agent1.agent_id in agent_ids
        assert agent2.agent_id in agent_ids
        assert agent3.agent_id in agent_ids
        
        # Test retrieval
        retrieved1 = gguf_manager.get_agent_state(agent1.agent_id)
        assert retrieved1 is not None
        assert retrieved1.name == "Agent 1"
        
        # Test non-existent agent
        non_existent = gguf_manager.get_agent_state("non_existent_id")
        assert non_existent is None

    def test_complex_agent_scenario(self, gguf_manager):
        """Test complex robotics agent scenario."""
        # Create a complex robotics agent
        agent_state = gguf_manager.create_agent_state(
            name="Quadruped Robot Agent",
            metadata={
                "robot_type": "quadruped",
                "manufacturer": "Marduk Robotics",
                "software_version": "2.1.0",
                "deployment": "field_test"
            }
        )
        
        # Add various tensors for robot state
        agent_state.add_tensor("joint_positions", [12], "float32")  # 12 joints
        agent_state.add_tensor("joint_velocities", [12], "float32")
        agent_state.add_tensor("joint_torques", [12], "float32")
        agent_state.add_tensor("base_orientation", [4], "float32")  # Quaternion
        agent_state.add_tensor("base_velocity", [6], "float32")  # Linear + angular
        agent_state.add_tensor("foot_contacts", [4], "bool")  # 4 feet
        agent_state.add_tensor("imu_data", [9], "float32")  # accel + gyro + mag
        agent_state.add_tensor("lidar_scan", [360], "float32")  # 360-degree lidar
        agent_state.add_tensor("camera_image", [480, 640, 3], "uint8")  # RGB image
        agent_state.add_tensor("policy_output", [12], "float32")  # Neural policy output
        agent_state.add_tensor("value_estimate", [1], "float32")  # Value function
        
        # Add device configurations
        for i in range(12):
            agent_state.add_device_config(f"joint_{i}", {
                "type": "servo_motor",
                "max_torque": 20.0,
                "gear_ratio": 10.0,
                "encoder_resolution": 4096
            })
        
        agent_state.add_device_config("imu", {
            "type": "9dof_imu",
            "sampling_rate": 1000,
            "accelerometer_range": "±16g",
            "gyroscope_range": "±2000dps"
        })
        
        agent_state.add_device_config("lidar", {
            "type": "2d_lidar",
            "range": 10.0,
            "angular_resolution": 1.0,
            "update_rate": 40
        })
        
        agent_state.add_device_config("camera", {
            "type": "rgb_camera",
            "resolution": [640, 480],
            "fps": 30,
            "auto_exposure": True
        })
        
        # Add control loops
        agent_state.add_control_loop({
            "loop_id": "joint_position_controller",
            "type": "pid",
            "update_rate": 1000,
            "gains": {"kp": 50.0, "ki": 5.0, "kd": 1.0}
        })
        
        agent_state.add_control_loop({
            "loop_id": "balance_controller",
            "type": "neural_symbolic",
            "neural_component": "transformer",
            "symbolic_rules": ["contact_constraint", "stability_margin"],
            "update_rate": 100
        })
        
        agent_state.add_control_loop({
            "loop_id": "navigation_planner",
            "type": "reinforcement_learning",
            "algorithm": "ppo",
            "network_architecture": "mlp",
            "update_rate": 10
        })
        
        # Test serialization and schema generation
        gguf_data = gguf_manager.serialize_agent_state(agent_state.agent_id)
        assert len(gguf_data) > 1000  # Should be substantial
        
        schema = gguf_manager.get_agent_tensor_schema(agent_state.agent_id)
        assert schema["tensor_count"] == 11
        assert schema["total_parameters"] > 900000  # Camera image dominates
        
        # Verify all components
        assert len(agent_state.tensors) == 11
        assert len(agent_state.device_configs) == 16  # 12 joints + 4 sensors
        assert len(agent_state.control_loops) == 3