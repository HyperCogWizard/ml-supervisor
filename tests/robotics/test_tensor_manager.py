"""Test tensor manager functionality."""

import pytest
import numpy as np
import time

from supervisor.robotics.tensor_manager import TensorManager, TensorField


class TestTensorManager:
    """Test tensor field manager for robotics workbench."""

    @pytest.fixture
    def tensor_manager(self, coresys):
        """Return tensor manager."""
        return TensorManager(coresys)

    def test_tensor_field_creation(self, tensor_manager):
        """Test basic tensor field creation."""
        field_id = tensor_manager.create_tensor_field(
            name="test_field",
            shape=(3, 100),
            dtype="float32"
        )
        
        assert field_id is not None
        field = tensor_manager.get_tensor_field(field_id)
        assert field is not None
        assert field.name == "test_field"
        assert field.shape == (3, 100)
        assert field.dtype == "float32"
        assert field.element_count == 300
        assert field.degrees_of_freedom == 2

    def test_tensor_field_with_initial_data(self, tensor_manager):
        """Test tensor field creation with initial data."""
        initial_data = np.random.randn(2, 50).astype(np.float32)
        
        field_id = tensor_manager.create_tensor_field(
            name="initialized_field",
            shape=(2, 50),
            dtype="float32",
            initial_data=initial_data
        )
        
        field = tensor_manager.get_tensor_field(field_id)
        assert field is not None
        assert np.array_equal(field.data, initial_data)

    def test_tensor_field_update(self, tensor_manager):
        """Test updating tensor field data."""
        field_id = tensor_manager.create_tensor_field(
            name="updateable_field",
            shape=(5, 10),
            dtype="float32"
        )
        
        # Update with new data
        new_data = np.ones((5, 10), dtype=np.float32) * 2.5
        success = tensor_manager.update_tensor_field(field_id, new_data)
        
        assert success is True
        
        field = tensor_manager.get_tensor_field(field_id)
        assert np.allclose(field.data, 2.5)
        assert field.updated_at > field.created_at

    def test_tensor_field_update_shape_mismatch(self, tensor_manager):
        """Test tensor field update with wrong shape."""
        field_id = tensor_manager.create_tensor_field(
            name="shape_test",
            shape=(3, 3),
            dtype="float32"
        )
        
        # Try to update with wrong shape
        wrong_data = np.ones((2, 2), dtype=np.float32)
        success = tensor_manager.update_tensor_field(field_id, wrong_data)
        
        assert success is False

    def test_device_tensor_field_creation(self, tensor_manager):
        """Test device-specific tensor field creation."""
        field_id = tensor_manager.create_device_tensor_field(
            device_id="servo_motor_1",
            device_type="servo",
            channels=3,
            temporal_length=200
        )
        
        field = tensor_manager.get_tensor_field(field_id)
        assert field is not None
        assert field.name == "servo_servo_motor_1"
        assert field.shape == (3, 200)
        assert field.metadata["device_id"] == "servo_motor_1"
        assert field.metadata["device_type"] == "servo"
        assert field.metadata["field_type"] == "device_data"

    def test_sensor_tensor_field_creation(self, tensor_manager):
        """Test sensor-specific tensor field creation."""
        field_id = tensor_manager.create_sensor_tensor_field(
            sensor_id="imu_1",
            sensor_type="9dof_imu",
            channels=9,
            sampling_rate=1000.0,
            buffer_duration=0.5
        )
        
        field = tensor_manager.get_tensor_field(field_id)
        assert field is not None
        assert field.name == "sensor_9dof_imu_imu_1"
        assert field.shape == (9, 500)  # 1000 Hz * 0.5 seconds
        assert field.metadata["sensor_type"] == "9dof_imu"
        assert field.metadata["sampling_rate"] == 1000.0
        assert field.metadata["field_type"] == "sensor_data"

    def test_actuator_tensor_field_creation(self, tensor_manager):
        """Test actuator-specific tensor field creation."""
        field_id = tensor_manager.create_actuator_tensor_field(
            actuator_id="robotic_arm",
            actuator_type="6dof_arm",
            dof=6,
            control_history=100
        )
        
        field = tensor_manager.get_tensor_field(field_id)
        assert field is not None
        assert field.name == "actuator_6dof_arm_robotic_arm"
        assert field.shape == (6, 100)
        assert field.metadata["actuator_type"] == "6dof_arm"
        assert field.metadata["degrees_of_freedom"] == 6
        assert field.metadata["field_type"] == "actuator_control"

    def test_agent_state_tensor_field_creation(self, tensor_manager):
        """Test agent state tensor field creation."""
        field_id = tensor_manager.create_agent_state_tensor_field(
            agent_id="neural_agent_1",
            state_dimension=512,
            sequence_length=10
        )
        
        field = tensor_manager.get_tensor_field(field_id)
        assert field is not None
        assert field.name == "agent_state_neural_agent_1"
        assert field.shape == (10, 512)
        assert field.metadata["agent_id"] == "neural_agent_1"
        assert field.metadata["field_type"] == "agent_state"

    def test_tensor_field_groups(self, tensor_manager):
        """Test tensor field grouping."""
        # Create multiple fields
        field1 = tensor_manager.create_tensor_field("pos_x", (1, 100), "float32")
        field2 = tensor_manager.create_tensor_field("pos_y", (1, 100), "float32")
        field3 = tensor_manager.create_tensor_field("pos_z", (1, 100), "float32")
        field4 = tensor_manager.create_tensor_field("vel_x", (1, 100), "float32")
        
        # Create groups
        success1 = tensor_manager.create_field_group("position", [field1, field2, field3])
        success2 = tensor_manager.create_field_group("velocity", [field4])
        
        assert success1 is True
        assert success2 is True
        
        # Test group retrieval
        position_fields = tensor_manager.get_field_group("position")
        velocity_fields = tensor_manager.get_field_group("velocity")
        
        assert len(position_fields) == 3
        assert len(velocity_fields) == 1
        assert all(f.name.startswith("pos_") for f in position_fields)

    def test_tensor_statistics(self, tensor_manager):
        """Test tensor field statistics."""
        # Create field with known data
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        field_id = tensor_manager.create_tensor_field(
            name="stats_test",
            shape=(2, 3),
            dtype="float32",
            initial_data=data
        )
        
        stats = tensor_manager.get_tensor_statistics(field_id)
        assert stats is not None
        assert stats["shape"] == (2, 3)
        assert stats["element_count"] == 6
        assert stats["degrees_of_freedom"] == 2
        assert abs(stats["mean"] - 3.5) < 1e-6  # Mean of 1,2,3,4,5,6
        assert abs(stats["min"] - 1.0) < 1e-6
        assert abs(stats["max"] - 6.0) < 1e-6

    def test_all_tensor_statistics(self, tensor_manager):
        """Test getting statistics for all fields."""
        # Create multiple fields
        tensor_manager.create_tensor_field("field1", (2, 2), "float32", 
                                         initial_data=np.ones((2, 2)))
        tensor_manager.create_tensor_field("field2", (3, 1), "float32",
                                         initial_data=np.zeros((3, 1)))
        
        all_stats = tensor_manager.get_all_tensor_statistics()
        assert len(all_stats) == 2
        
        for field_id, stats in all_stats.items():
            assert "shape" in stats
            assert "mean" in stats
            assert "element_count" in stats

    def test_tensor_operations(self, tensor_manager):
        """Test tensor operations between fields."""
        # Create fields with known data
        data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data2 = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        
        field1 = tensor_manager.create_tensor_field("operand1", (2, 2), "float32", initial_data=data1)
        field2 = tensor_manager.create_tensor_field("operand2", (2, 2), "float32", initial_data=data2)
        
        # Test addition
        result_field = tensor_manager.compute_tensor_operations("add", [field1, field2])
        assert result_field is not None
        
        result_data = tensor_manager.get_tensor_field(result_field).data
        expected = data1 + data2
        assert np.allclose(result_data, expected)
        
        # Test multiplication
        result_field = tensor_manager.compute_tensor_operations("multiply", [field1, field2])
        assert result_field is not None
        
        result_data = tensor_manager.get_tensor_field(result_field).data
        expected = data1 * data2
        assert np.allclose(result_data, expected)

    def test_tensor_mean_operation(self, tensor_manager):
        """Test tensor mean operation."""
        # Create multiple fields
        data1 = np.ones((2, 3), dtype=np.float32) * 2.0
        data2 = np.ones((2, 3), dtype=np.float32) * 4.0
        data3 = np.ones((2, 3), dtype=np.float32) * 6.0
        
        field1 = tensor_manager.create_tensor_field("mean1", (2, 3), "float32", initial_data=data1)
        field2 = tensor_manager.create_tensor_field("mean2", (2, 3), "float32", initial_data=data2)
        field3 = tensor_manager.create_tensor_field("mean3", (2, 3), "float32", initial_data=data3)
        
        # Compute mean
        result_field = tensor_manager.compute_tensor_operations("mean", [field1, field2, field3])
        assert result_field is not None
        
        result_data = tensor_manager.get_tensor_field(result_field).data
        expected = np.ones((2, 3), dtype=np.float32) * 4.0  # Mean of 2, 4, 6
        assert np.allclose(result_data, expected)

    def test_tensor_concatenation(self, tensor_manager):
        """Test tensor concatenation operation."""
        data1 = np.ones((2, 2), dtype=np.float32)
        data2 = np.ones((2, 3), dtype=np.float32) * 2.0
        
        field1 = tensor_manager.create_tensor_field("concat1", (2, 2), "float32", initial_data=data1)
        field2 = tensor_manager.create_tensor_field("concat2", (2, 3), "float32", initial_data=data2)
        
        # Concatenate along last axis
        result_field = tensor_manager.compute_tensor_operations("concatenate", [field1, field2])
        assert result_field is not None
        
        result_data = tensor_manager.get_tensor_field(result_field).data
        assert result_data.shape == (2, 5)  # 2 + 3 columns
        assert np.allclose(result_data[:, :2], 1.0)
        assert np.allclose(result_data[:, 2:], 2.0)

    def test_visualization_data_update(self, tensor_manager):
        """Test visualization data updates."""
        # Create 1D field
        data_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        field_1d = tensor_manager.create_tensor_field("viz_1d", (4,), "float32", initial_data=data_1d)
        
        # Create 2D field
        data_2d = np.random.randn(3, 5).astype(np.float32)
        field_2d = tensor_manager.create_tensor_field("viz_2d", (3, 5), "float32", initial_data=data_2d)
        
        # Get visualization data
        viz_data = tensor_manager.get_visualization_data()
        
        assert field_1d in viz_data
        assert field_2d in viz_data
        
        # Check 1D data is stored fully
        viz_1d = viz_data[field_1d]
        assert np.array_equal(viz_1d["data"], data_1d)
        
        # Check 2D data has summary statistics
        viz_2d = viz_data[field_2d]
        assert "axis0_mean" in viz_2d["data"]
        assert "axis1_mean" in viz_2d["data"]
        assert "global_mean" in viz_2d["data"]

    def test_tensor_configuration_export(self, tensor_manager):
        """Test complete tensor configuration export."""
        # Create various tensor fields
        tensor_manager.create_device_tensor_field("device_1", "servo", channels=2)
        tensor_manager.create_sensor_tensor_field("sensor_1", "imu", channels=9, sampling_rate=1000)
        tensor_manager.create_actuator_tensor_field("actuator_1", "motor", dof=3)
        tensor_manager.create_agent_state_tensor_field("agent_1", state_dimension=128)
        
        # Create field groups
        device_fields = [f["field_id"] for f in tensor_manager.list_tensor_fields() 
                        if f["metadata"].get("field_type") == "device_data"]
        sensor_fields = [f["field_id"] for f in tensor_manager.list_tensor_fields()
                        if f["metadata"].get("field_type") == "sensor_data"]
        
        tensor_manager.create_field_group("devices", device_fields)
        tensor_manager.create_field_group("sensors", sensor_fields)
        
        # Export configuration
        config = tensor_manager.export_tensor_configuration()
        
        assert config["total_fields"] == 4
        assert config["total_elements"] > 0
        assert config["total_degrees_of_freedom"] > 0
        assert len(config["field_groups"]) == 2
        assert config["field_groups"]["devices"] == 1
        assert config["field_groups"]["sensors"] == 1
        
        # Check field details
        fields = config["fields"]
        assert len(fields) == 4
        
        for field_id, field_info in fields.items():
            assert "name" in field_info
            assert "shape" in field_info
            assert "dtype" in field_info
            assert "element_count" in field_info
            assert "degrees_of_freedom" in field_info
            assert "metadata" in field_info

    def test_tensor_field_removal(self, tensor_manager):
        """Test tensor field removal."""
        # Create fields and group
        field1 = tensor_manager.create_tensor_field("remove_test1", (2, 2), "float32")
        field2 = tensor_manager.create_tensor_field("remove_test2", (3, 3), "float32")
        tensor_manager.create_field_group("test_group", [field1, field2])
        
        # Remove field1
        success = tensor_manager.remove_tensor_field(field1)
        assert success is True
        
        # Verify removal
        assert tensor_manager.get_tensor_field(field1) is None
        assert field1 not in tensor_manager._visualization_data
        
        # Verify field2 still exists
        assert tensor_manager.get_tensor_field(field2) is not None
        
        # Verify group was updated
        group_fields = tensor_manager.get_field_group("test_group")
        assert len(group_fields) == 1
        assert group_fields[0].field_id == field2

    def test_clear_all_fields(self, tensor_manager):
        """Test clearing all tensor fields."""
        # Create some fields and groups
        tensor_manager.create_tensor_field("clear_test1", (1, 1), "float32")
        tensor_manager.create_tensor_field("clear_test2", (2, 2), "float32")
        tensor_manager.create_field_group("clear_group", ["clear_test1", "clear_test2"])
        
        # Verify non-empty
        assert len(tensor_manager._tensor_fields) > 0
        assert len(tensor_manager._field_groups) > 0
        
        # Clear all
        tensor_manager.clear_all_fields()
        
        # Verify empty
        assert len(tensor_manager._tensor_fields) == 0
        assert len(tensor_manager._field_groups) == 0
        assert len(tensor_manager._visualization_data) == 0

    def test_complex_robotics_tensor_scenario(self, tensor_manager):
        """Test complex robotics tensor field scenario."""
        # Create comprehensive robot state representation
        
        # Joint state tensors
        joint_positions = tensor_manager.create_actuator_tensor_field(
            "joints", "robot_joints", dof=12, control_history=1000
        )
        joint_velocities = tensor_manager.create_tensor_field(
            "joint_velocities", (12, 1000), "float32",
            metadata={"type": "joint_derivatives", "units": "rad/s"}
        )
        joint_torques = tensor_manager.create_tensor_field(
            "joint_torques", (12, 1000), "float32",
            metadata={"type": "joint_forces", "units": "Nm"}
        )
        
        # Sensor data tensors
        imu_data = tensor_manager.create_sensor_tensor_field(
            "body_imu", "9dof_imu", channels=9, sampling_rate=1000, buffer_duration=1.0
        )
        lidar_data = tensor_manager.create_sensor_tensor_field(
            "lidar", "360_lidar", channels=360, sampling_rate=40, buffer_duration=0.5
        )
        camera_data = tensor_manager.create_tensor_field(
            "rgb_camera", (480, 640, 3, 30), "uint8",  # Height x Width x Channels x Buffer
            metadata={"type": "visual_input", "fps": 30}
        )
        
        # Neural agent state tensors
        policy_hidden = tensor_manager.create_agent_state_tensor_field(
            "policy_agent", state_dimension=512, sequence_length=100
        )
        value_hidden = tensor_manager.create_agent_state_tensor_field(
            "value_agent", state_dimension=256, sequence_length=100
        )
        world_model = tensor_manager.create_tensor_field(
            "world_model_state", (1, 1024), "float32",
            metadata={"type": "world_representation", "compression": "vae"}
        )
        
        # Control outputs
        action_logits = tensor_manager.create_tensor_field(
            "action_logits", (12, 100), "float32",  # Actions for 12 joints over 100 timesteps
            metadata={"type": "policy_output", "action_space": "continuous"}
        )
        value_estimates = tensor_manager.create_tensor_field(
            "value_estimates", (1, 100), "float32",
            metadata={"type": "value_function", "bootstrap": True}
        )
        
        # Create logical groupings
        tensor_manager.create_field_group("joint_state", [joint_positions, joint_velocities, joint_torques])
        tensor_manager.create_field_group("sensory_input", [imu_data, lidar_data, camera_data])
        tensor_manager.create_field_group("neural_state", [policy_hidden, value_hidden, world_model])
        tensor_manager.create_field_group("control_output", [action_logits, value_estimates])
        
        # Populate with realistic data
        np.random.seed(42)  # For reproducible tests
        
        # Joint data - smooth trajectories
        t = np.linspace(0, 10, 1000)
        joint_data = np.sin(t[np.newaxis, :] + np.arange(12)[:, np.newaxis] * 0.5)
        tensor_manager.update_tensor_field(joint_positions, joint_data.astype(np.float32))
        
        # IMU data - simulated motion
        imu_sim = np.random.randn(9, 1000).astype(np.float32) * 0.1
        imu_sim[:3, :] += 9.81 * np.array([0, 0, 1])[:, np.newaxis]  # Add gravity
        tensor_manager.update_tensor_field(imu_data, imu_sim)
        
        # Neural states - typical network activations
        policy_state = np.random.randn(100, 512).astype(np.float32) * 0.5
        tensor_manager.update_tensor_field(policy_hidden, policy_state)
        
        # Test comprehensive statistics
        all_stats = tensor_manager.get_all_tensor_statistics()
        assert len(all_stats) >= 10
        
        # Test configuration export
        config = tensor_manager.export_tensor_configuration()
        
        # Verify scale and complexity
        assert config["total_fields"] >= 10
        assert config["total_elements"] > 1000000  # Camera data dominates
        assert config["total_degrees_of_freedom"] >= 20
        assert len(config["field_groups"]) == 4
        
        # Test visualization data
        viz_data = tensor_manager.get_visualization_data()
        assert len(viz_data) >= 10
        
        # Verify realistic value ranges
        joint_stats = tensor_manager.get_tensor_statistics(joint_positions)
        assert -1.5 <= joint_stats["min"] <= -0.5  # Sine wave range
        assert 0.5 <= joint_stats["max"] <= 1.5
        
        imu_stats = tensor_manager.get_tensor_statistics(imu_data)
        assert 8.0 <= imu_stats["mean"] <= 11.0  # Should be around gravity magnitude
        
        # Test tensor operations on robot data
        joint_velocity_computed = tensor_manager.compute_tensor_operations(
            "mean", [joint_positions, joint_velocities]
        )
        assert joint_velocity_computed is not None
        
        neural_fusion = tensor_manager.compute_tensor_operations(
            "concatenate", [policy_hidden, value_hidden]
        )
        assert neural_fusion is not None
        fusion_field = tensor_manager.get_tensor_field(neural_fusion)
        assert fusion_field.shape == (100, 768)  # 512 + 256