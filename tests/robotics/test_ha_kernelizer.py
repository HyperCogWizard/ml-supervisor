"""Test HomeAssistant kernelization functionality."""

import pytest
import numpy as np

from supervisor.robotics.ha_kernelizer import HomeAssistantKernelizer, AgenticAutomation


class TestHomeAssistantKernelizer:
    """Test HomeAssistant to agentic control loop conversion."""

    @pytest.fixture
    def kernelizer(self, coresys):
        """Return HA kernelizer."""
        return HomeAssistantKernelizer(coresys)

    def test_agentic_automation_creation(self):
        """Test agentic automation creation."""
        automation = AgenticAutomation(
            automation_id="test_auto",
            name="Test Automation",
            entity_ids=["light.living_room", "sensor.motion"],
            control_logic={
                "trigger": {"platform": "state", "entity_id": "sensor.motion", "to": "on"},
                "action": {"service": "light.turn_on", "entity_id": "light.living_room"}
            }
        )
        
        assert automation.automation_id == "test_auto"
        assert automation.name == "Test Automation"
        assert len(automation.entity_ids) == 2
        assert "light.living_room" in automation.entity_ids
        assert "sensor.motion" in automation.entity_ids

    def test_scheme_function_generation(self):
        """Test Scheme function generation from automation."""
        automation = AgenticAutomation(
            automation_id="motion_light",
            name="Motion Light",
            entity_ids=["sensor.motion", "light.kitchen"],
            control_logic={}
        )
        
        scheme_code = automation.as_scheme_function()
        
        assert "(define motion_light" in scheme_code
        assert "(lambda (entities state)" in scheme_code
        assert "get-triggers" in scheme_code
        assert "get-conditions" in scheme_code
        assert "get-actions" in scheme_code

    @pytest.mark.asyncio
    async def test_automation_kernelization(self, kernelizer):
        """Test automation kernelization process."""
        # Mock workbench for testing
        class MockWorkbench:
            def __init__(self):
                self._experiments = {"test_exp": {"device_nodes": [], "tensor_fields": []}}
                self.hypergraph_engine = MockHypergraphEngine()
                self.tensor_manager = MockTensorManager()
            
            async def create_agentic_control_loop(self, experiment_id, loop_id, name, config):
                return True
        
        class MockHypergraphEngine:
            def add_device_node(self, **kwargs):
                return f"device_{kwargs['device_id']}"
        
        class MockTensorManager:
            def create_tensor_field(self, **kwargs):
                return f"field_{kwargs['name']}"
        
        kernelizer.coresys.robotics_workbench = MockWorkbench()
        
        automation_config = {
            "id": "motion_light_auto",
            "alias": "Motion Light Automation",
            "trigger": {
                "platform": "state",
                "entity_id": "binary_sensor.motion_living_room",
                "from": "off",
                "to": "on"
            },
            "action": {
                "service": "light.turn_on",
                "entity_id": "light.living_room",
                "data": {"brightness": 255}
            }
        }
        
        automation_id = await kernelizer.kernelize_automation(automation_config, "test_exp")
        
        assert automation_id == "motion_light_auto"
        assert automation_id in kernelizer._agentic_automations
        
        automation = kernelizer._agentic_automations[automation_id]
        assert automation.name == "Motion Light Automation"
        assert "binary_sensor.motion_living_room" in automation.entity_ids
        assert "light.living_room" in automation.entity_ids

    def test_entity_tensor_config_light(self, kernelizer):
        """Test tensor configuration for light entities."""
        config = kernelizer._get_entity_tensor_config("light", "light.living_room")
        
        assert config["tensor_shape"] == (4, 100)  # brightness, r, g, b over time
        assert config["dtype"] == "float32"
        assert "brightness" in config["modalities"]
        assert "color" in config["modalities"]
        assert config["properties"]["controllable"] is True

    def test_entity_tensor_config_sensor(self, kernelizer):
        """Test tensor configuration for sensor entities."""
        config = kernelizer._get_entity_tensor_config("sensor", "sensor.temperature")
        
        assert config["tensor_shape"] == (1, 100)  # single value over time
        assert config["dtype"] == "float32"
        assert "measurement" in config["modalities"]
        assert config["properties"]["controllable"] is False

    def test_entity_tensor_config_climate(self, kernelizer):
        """Test tensor configuration for climate entities."""
        config = kernelizer._get_entity_tensor_config("climate", "climate.thermostat")
        
        assert config["tensor_shape"] == (3, 100)  # temp, target_temp, mode over time
        assert config["dtype"] == "float32"
        assert "temperature" in config["modalities"]
        assert "control" in config["modalities"]
        assert config["properties"]["hvac"] is True

    def test_symbolic_rules_extraction(self, kernelizer):
        """Test extraction of symbolic rules from automation."""
        automation_config = {
            "trigger": [
                {
                    "platform": "state",
                    "entity_id": "sensor.temperature",
                    "above": 25
                },
                {
                    "platform": "time",
                    "at": "18:00:00"
                }
            ],
            "condition": [
                {
                    "condition": "state",
                    "entity_id": "binary_sensor.presence",
                    "state": "on"
                },
                {
                    "condition": "numeric_state",
                    "entity_id": "sensor.humidity",
                    "below": 60
                }
            ]
        }
        
        rules = kernelizer._extract_symbolic_rules(automation_config)
        
        assert len(rules) >= 4
        assert any("value_above(sensor.temperature, 25)" in rule for rule in rules)
        assert any("time_trigger(18:00:00)" in rule for rule in rules)
        assert any("condition_state(binary_sensor.presence, on)" in rule for rule in rules)
        assert any("condition_below(sensor.humidity, 60)" in rule for rule in rules)

    def test_state_to_tensor_light(self, kernelizer):
        """Test converting light state to tensor."""
        state_data = {
            "state": "on",
            "attributes": {
                "brightness": 128,
                "rgb_color": [255, 128, 64]
            }
        }
        
        tensor_data = kernelizer._state_to_tensor("light.test", state_data, (4, 100))
        
        assert tensor_data is not None
        assert len(tensor_data) == 4
        assert abs(tensor_data[0] - 128.0/255.0) < 1e-6  # brightness
        assert abs(tensor_data[1] - 1.0) < 1e-6  # red
        assert abs(tensor_data[2] - 128.0/255.0) < 1e-6  # green
        assert abs(tensor_data[3] - 64.0/255.0) < 1e-6  # blue

    def test_state_to_tensor_switch(self, kernelizer):
        """Test converting switch state to tensor."""
        state_on = {"state": "on", "attributes": {}}
        state_off = {"state": "off", "attributes": {}}
        
        tensor_on = kernelizer._state_to_tensor("switch.test", state_on, (1, 100))
        tensor_off = kernelizer._state_to_tensor("switch.test", state_off, (1, 100))
        
        assert tensor_on is not None
        assert tensor_off is not None
        assert tensor_on[0] == 1.0
        assert tensor_off[0] == 0.0

    def test_state_to_tensor_sensor(self, kernelizer):
        """Test converting sensor state to tensor."""
        state_data = {
            "state": "23.5",
            "attributes": {
                "unit_of_measurement": "°C"
            }
        }
        
        tensor_data = kernelizer._state_to_tensor("sensor.temperature", state_data, (1, 100))
        
        assert tensor_data is not None
        assert len(tensor_data) == 1
        assert abs(tensor_data[0] - 23.5) < 1e-6

    def test_state_to_tensor_climate(self, kernelizer):
        """Test converting climate state to tensor."""
        state_data = {
            "state": "heat",
            "attributes": {
                "temperature": 22.5,
                "target_temp": 24.0,
                "hvac_mode": "heat"
            }
        }
        
        tensor_data = kernelizer._state_to_tensor("climate.thermostat", state_data, (3, 100))
        
        assert tensor_data is not None
        assert len(tensor_data) == 3
        assert abs(tensor_data[0] - 22.5) < 1e-6  # current temp
        assert abs(tensor_data[1] - 24.0) < 1e-6  # target temp
        assert tensor_data[2] == 1.0  # heat mode encoded as 1

    def test_ensure_list_utility(self, kernelizer):
        """Test ensure_list utility function."""
        # Single string
        result = kernelizer._ensure_list("entity.test")
        assert result == ["entity.test"]
        
        # List of strings
        result = kernelizer._ensure_list(["entity.one", "entity.two"])
        assert result == ["entity.one", "entity.two"]
        
        # Mixed list
        result = kernelizer._ensure_list([123, "entity.test", True])
        assert result == ["123", "entity.test", "True"]

    def test_kernelization_status(self, kernelizer):
        """Test kernelization status reporting."""
        # Add some test automations
        auto1 = AgenticAutomation("auto1", "Test 1", ["light.1"], {})
        auto2 = AgenticAutomation("auto2", "Test 2", ["switch.1"], {})
        auto1.active = True
        auto2.active = False
        auto1.execution_count = 5
        
        kernelizer._agentic_automations["auto1"] = auto1
        kernelizer._agentic_automations["auto2"] = auto2
        kernelizer._entity_tensor_mappings["light.1"] = "field_1"
        kernelizer._entity_tensor_mappings["switch.1"] = "field_2"
        
        status = kernelizer.get_kernelization_status()
        
        assert status["total_automations"] == 2
        assert status["active_automations"] == 1
        assert status["kernelized_entities"] == 2
        assert len(status["automations"]) == 2
        
        auto1_status = next(a for a in status["automations"] if a["id"] == "auto1")
        assert auto1_status["active"] is True
        assert auto1_status["execution_count"] == 5

    @pytest.mark.asyncio
    async def test_automation_activation(self, kernelizer):
        """Test automation activation/deactivation."""
        auto = AgenticAutomation("test_auto", "Test", [], {})
        kernelizer._agentic_automations["test_auto"] = auto
        
        # Test activation
        success = await kernelizer.activate_automation("test_auto")
        assert success is True
        assert auto.active is True
        
        # Test deactivation
        success = await kernelizer.deactivate_automation("test_auto")
        assert success is True
        assert auto.active is False
        
        # Test non-existent automation
        success = await kernelizer.activate_automation("non_existent")
        assert success is False

    def test_complex_automation_kernelization(self, kernelizer):
        """Test complex automation with multiple triggers and conditions."""
        complex_automation = {
            "id": "complex_automation",
            "alias": "Complex Security Automation",
            "trigger": [
                {
                    "platform": "state",
                    "entity_id": ["binary_sensor.door_contact", "binary_sensor.window_contact"],
                    "from": "off",
                    "to": "on"
                },
                {
                    "platform": "numeric_state",
                    "entity_id": "sensor.motion_confidence",
                    "above": 0.8
                }
            ],
            "condition": [
                {
                    "condition": "state",
                    "entity_id": "alarm_control_panel.security",
                    "state": "armed_away"
                },
                {
                    "condition": "time",
                    "after": "22:00:00",
                    "before": "06:00:00"
                }
            ],
            "action": [
                {
                    "service": "light.turn_on",
                    "target": {"entity_id": ["light.security_flood", "light.pathway"]},
                    "data": {"brightness": 255, "color_name": "red"}
                },
                {
                    "service": "notify.security_team",
                    "data": {"message": "Security breach detected"}
                }
            ]
        }
        
        # Extract entity IDs
        entity_ids = []
        for trigger in complex_automation["trigger"]:
            if "entity_id" in trigger:
                entity_ids.extend(kernelizer._ensure_list(trigger["entity_id"]))
        
        for condition in complex_automation["condition"]:
            if "entity_id" in condition:
                entity_ids.extend(kernelizer._ensure_list(condition["entity_id"]))
        
        for action in complex_automation["action"]:
            if "entity_id" in action:
                entity_ids.extend(kernelizer._ensure_list(action["entity_id"]))
            elif "target" in action and "entity_id" in action["target"]:
                entity_ids.extend(kernelizer._ensure_list(action["target"]["entity_id"]))
        
        entity_ids = list(set(entity_ids))
        
        # Verify comprehensive entity extraction
        expected_entities = [
            "binary_sensor.door_contact",
            "binary_sensor.window_contact", 
            "sensor.motion_confidence",
            "alarm_control_panel.security",
            "light.security_flood",
            "light.pathway"
        ]
        
        for entity in expected_entities:
            assert entity in entity_ids
        
        # Test symbolic rule extraction
        rules = kernelizer._extract_symbolic_rules(complex_automation)
        assert len(rules) >= 3
        
        # Verify rule types
        rule_text = " ".join(rules)
        assert "state_transition" in rule_text
        assert "value_above" in rule_text
        assert "condition_state" in rule_text