"""HomeAssistant agentic kernelization for neural-symbolic control loops."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..const import BusEvent
from ..coresys import CoreSysAttributes
from ..homeassistant.module import HomeAssistant

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AgenticAutomation:
    """Represents an agentic automation replacing static YAML automation."""
    
    def __init__(self, automation_id: str, name: str, entity_ids: List[str], 
                 control_logic: Dict[str, Any], agent_id: Optional[str] = None):
        """Initialize agentic automation."""
        self.automation_id = automation_id
        self.name = name
        self.entity_ids = entity_ids
        self.control_logic = control_logic
        self.agent_id = agent_id
        self.active = False
        self.execution_count = 0
        self.last_execution = None

    def as_scheme_function(self) -> str:
        """Convert automation to Scheme function representation."""
        # Generate a Scheme function that represents the automation logic
        scheme_code = f"""(define {self.automation_id}
  (lambda (entities state)
    (let ((triggers (get-triggers entities))
          (conditions (get-conditions entities state))
          (actions (get-actions entities state)))
      (if (and triggers conditions)
          (execute-actions actions)
          '()))))"""
        return scheme_code


class HomeAssistantKernelizer(CoreSysAttributes):
    """Converts HomeAssistant automations to agentic control loops."""

    def __init__(self, coresys):
        """Initialize kernelizer."""
        self.coresys = coresys
        self._agentic_automations: Dict[str, AgenticAutomation] = {}
        self._entity_tensor_mappings: Dict[str, str] = {}  # entity_id -> tensor_field_id
        self._active_control_loops: Dict[str, Dict[str, Any]] = {}
        self._homeassistant_entities: Dict[str, Dict[str, Any]] = {}

    async def kernelize_automation(self, automation_config: Dict[str, Any], 
                                  experiment_id: str) -> str:
        """Convert a HomeAssistant automation to agentic control loop."""
        automation_id = automation_config.get("id", f"auto_{len(self._agentic_automations)}")
        name = automation_config.get("alias", f"Agentic Automation {automation_id}")
        
        # Extract entity IDs from triggers and actions
        entity_ids = []
        
        # Process triggers
        triggers = automation_config.get("trigger", [])
        if not isinstance(triggers, list):
            triggers = [triggers]
        
        for trigger in triggers:
            if "entity_id" in trigger:
                entity_ids.extend(self._ensure_list(trigger["entity_id"]))
        
        # Process actions
        actions = automation_config.get("action", [])
        if not isinstance(actions, list):
            actions = [actions]
            
        for action in actions:
            if "entity_id" in action:
                entity_ids.extend(self._ensure_list(action["entity_id"]))
            elif "target" in action and "entity_id" in action["target"]:
                entity_ids.extend(self._ensure_list(action["target"]["entity_id"]))

        # Remove duplicates
        entity_ids = list(set(entity_ids))
        
        # Create agentic automation
        agentic_auto = AgenticAutomation(
            automation_id=automation_id,
            name=name,
            entity_ids=entity_ids,
            control_logic=automation_config
        )
        
        self._agentic_automations[automation_id] = agentic_auto
        
        # Create tensor nodes for each entity
        workbench = self.coresys.robotics_workbench
        for entity_id in entity_ids:
            await self._create_entity_tensor_node(entity_id, experiment_id)
        
        # Create neural-symbolic control loop
        control_loop_config = {
            "loop_type": "neural_symbolic",
            "update_rate": 1.0,  # 1 Hz default for HA automations
            "scheme_function": agentic_auto.as_scheme_function(),
            "entity_ids": entity_ids,
            "original_automation": automation_config,
            "neural_component": "transformer",
            "symbolic_rules": self._extract_symbolic_rules(automation_config)
        }
        
        loop_id = f"ha_kernelized_{automation_id}"
        await workbench.create_agentic_control_loop(
            experiment_id=experiment_id,
            loop_id=loop_id,
            name=f"Kernelized {name}",
            config=control_loop_config
        )
        
        self._active_control_loops[automation_id] = {
            "loop_id": loop_id,
            "experiment_id": experiment_id,
            "config": control_loop_config
        }
        
        _LOGGER.info("Kernelized automation %s with %d entities", automation_id, len(entity_ids))
        return automation_id

    async def _create_entity_tensor_node(self, entity_id: str, experiment_id: str) -> str:
        """Create tensor node for HomeAssistant entity."""
        if entity_id in self._entity_tensor_mappings:
            return self._entity_tensor_mappings[entity_id]
        
        workbench = self.coresys.robotics_workbench
        
        # Determine entity type and tensor configuration
        domain = entity_id.split(".")[0]
        entity_config = self._get_entity_tensor_config(domain, entity_id)
        
        # Create hypergraph node
        node_id = workbench.hypergraph_engine.add_device_node(
            device_id=entity_id,
            device_type=f"homeassistant_{domain}",
            name=f"HA Entity {entity_id}",
            tensor_dims=entity_config["tensor_dims"],
            modalities=entity_config["modalities"],
            properties={
                "entity_id": entity_id,
                "domain": domain,
                "homeassistant_entity": True,
                "agentic_control": True,
                **entity_config["properties"]
            }
        )
        
        # Create tensor field
        field_id = workbench.tensor_manager.create_tensor_field(
            name=f"ha_entity_{entity_id.replace('.', '_')}",
            shape=entity_config["tensor_shape"],
            dtype=entity_config["dtype"],
            metadata={
                "entity_id": entity_id,
                "domain": domain,
                "homeassistant_entity": True,
                "field_type": "homeassistant_state",
                **entity_config["metadata"]
            }
        )
        
        self._entity_tensor_mappings[entity_id] = field_id
        
        # Add to experiment
        experiment = workbench._experiments[experiment_id]
        experiment["device_nodes"].append(node_id)
        experiment["tensor_fields"].append(field_id)
        
        _LOGGER.info("Created tensor node for HA entity %s", entity_id)
        return field_id

    def _get_entity_tensor_config(self, domain: str, entity_id: str) -> Dict[str, Any]:
        """Get tensor configuration for entity based on domain."""
        configs = {
            "light": {
                "tensor_dims": [4, 100],  # [brightness, r, g, b] x history
                "tensor_shape": (4, 100),
                "dtype": "float32",
                "modalities": ["brightness", "color"],
                "properties": {"controllable": True, "dimmable": True},
                "metadata": {"state_attributes": ["brightness", "rgb_color"]}
            },
            "switch": {
                "tensor_dims": [1, 100],  # [on/off] x history
                "tensor_shape": (1, 100),
                "dtype": "float32",
                "modalities": ["binary_state"],
                "properties": {"controllable": True, "binary": True},
                "metadata": {"state_attributes": ["state"]}
            },
            "sensor": {
                "tensor_dims": [1, 100],  # [value] x history
                "tensor_shape": (1, 100),
                "dtype": "float32",
                "modalities": ["measurement"],
                "properties": {"controllable": False, "readable": True},
                "metadata": {"state_attributes": ["state", "unit_of_measurement"]}
            },
            "binary_sensor": {
                "tensor_dims": [1, 100],  # [on/off] x history
                "tensor_shape": (1, 100),
                "dtype": "float32",
                "modalities": ["binary_state"],
                "properties": {"controllable": False, "binary": True},
                "metadata": {"state_attributes": ["state"]}
            },
            "climate": {
                "tensor_dims": [3, 100],  # [temperature, target_temp, mode] x history
                "tensor_shape": (3, 100),
                "dtype": "float32",
                "modalities": ["temperature", "control"],
                "properties": {"controllable": True, "hvac": True},
                "metadata": {"state_attributes": ["temperature", "target_temp", "hvac_mode"]}
            },
            "cover": {
                "tensor_dims": [2, 100],  # [position, tilt] x history
                "tensor_shape": (2, 100),
                "dtype": "float32",
                "modalities": ["position"],
                "properties": {"controllable": True, "positional": True},
                "metadata": {"state_attributes": ["position", "tilt_position"]}
            },
            "media_player": {
                "tensor_dims": [3, 100],  # [volume, position, state] x history
                "tensor_shape": (3, 100),
                "dtype": "float32",
                "modalities": ["audio", "control"],
                "properties": {"controllable": True, "media": True},
                "metadata": {"state_attributes": ["volume_level", "media_position", "state"]}
            }
        }
        
        return configs.get(domain, {
            "tensor_dims": [1, 100],  # Default: single value with history
            "tensor_shape": (1, 100),
            "dtype": "float32",
            "modalities": ["generic"],
            "properties": {"controllable": False},
            "metadata": {"state_attributes": ["state"]}
        })

    def _extract_symbolic_rules(self, automation_config: Dict[str, Any]) -> List[str]:
        """Extract symbolic rules from automation configuration."""
        rules = []
        
        # Extract trigger conditions as rules
        triggers = automation_config.get("trigger", [])
        if not isinstance(triggers, list):
            triggers = [triggers]
        
        for trigger in triggers:
            trigger_type = trigger.get("platform", "unknown")
            if trigger_type == "state":
                entity_id = trigger.get("entity_id", "unknown")
                from_state = trigger.get("from")
                to_state = trigger.get("to")
                if from_state and to_state:
                    rules.append(f"state_transition({entity_id}, {from_state}, {to_state})")
                elif to_state:
                    rules.append(f"state_equals({entity_id}, {to_state})")
            elif trigger_type == "time":
                time_val = trigger.get("at")
                if time_val:
                    rules.append(f"time_trigger({time_val})")
            elif trigger_type == "numeric_state":
                entity_id = trigger.get("entity_id", "unknown")
                above = trigger.get("above")
                below = trigger.get("below")
                if above:
                    rules.append(f"value_above({entity_id}, {above})")
                if below:
                    rules.append(f"value_below({entity_id}, {below})")
        
        # Extract conditions as rules
        conditions = automation_config.get("condition", [])
        if not isinstance(conditions, list):
            conditions = [conditions]
        
        for condition in conditions:
            condition_type = condition.get("condition", "unknown")
            if condition_type == "state":
                entity_id = condition.get("entity_id", "unknown")
                state = condition.get("state")
                if state:
                    rules.append(f"condition_state({entity_id}, {state})")
            elif condition_type == "numeric_state":
                entity_id = condition.get("entity_id", "unknown")
                above = condition.get("above")
                below = condition.get("below")
                if above:
                    rules.append(f"condition_above({entity_id}, {above})")
                if below:
                    rules.append(f"condition_below({entity_id}, {below})")
        
        return rules

    def _ensure_list(self, value: Any) -> List[str]:
        """Ensure value is a list of strings."""
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return [str(v) for v in value]
        else:
            return [str(value)]

    async def update_entity_tensor(self, entity_id: str, state_data: Dict[str, Any]) -> bool:
        """Update tensor field with HomeAssistant entity state."""
        if entity_id not in self._entity_tensor_mappings:
            return False
        
        field_id = self._entity_tensor_mappings[entity_id]
        workbench = self.coresys.robotics_workbench
        field = workbench.tensor_manager.get_tensor_field(field_id)
        
        if not field:
            return False
        
        # Convert HA state to tensor data
        tensor_data = self._state_to_tensor(entity_id, state_data, field.shape)
        
        if tensor_data is not None:
            # Shift history and add new data
            if len(field.shape) == 2:  # Has history dimension
                new_data = field.data.copy()
                new_data[:, :-1] = new_data[:, 1:]  # Shift left
                new_data[:, -1] = tensor_data  # Add new data
            else:
                new_data = tensor_data
            
            success = workbench.tensor_manager.update_tensor_field(field_id, new_data)
            if success:
                # Trigger agentic evaluations
                await self._evaluate_affected_automations(entity_id)
            
            return success
        
        return False

    def _state_to_tensor(self, entity_id: str, state_data: Dict[str, Any], shape: tuple) -> Any:
        """Convert HomeAssistant state to tensor data."""
        import numpy as np
        
        domain = entity_id.split(".")[0]
        state = state_data.get("state", "unknown")
        attributes = state_data.get("attributes", {})
        
        try:
            if domain == "light":
                # [brightness, r, g, b]
                brightness = float(attributes.get("brightness", 0)) / 255.0
                rgb = attributes.get("rgb_color", [0, 0, 0])
                return np.array([brightness, rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0], dtype=np.float32)
            
            elif domain in ["switch", "binary_sensor"]:
                # [on/off]
                return np.array([1.0 if state == "on" else 0.0], dtype=np.float32)
            
            elif domain == "sensor":
                # [numeric_value]
                try:
                    value = float(state)
                    return np.array([value], dtype=np.float32)
                except (ValueError, TypeError):
                    return np.array([0.0], dtype=np.float32)
            
            elif domain == "climate":
                # [current_temp, target_temp, mode_encoded]
                current_temp = float(attributes.get("temperature", 0))
                target_temp = float(attributes.get("target_temp", 0))
                hvac_mode = attributes.get("hvac_mode", "off")
                mode_map = {"off": 0, "heat": 1, "cool": 2, "auto": 3}
                mode_encoded = mode_map.get(hvac_mode, 0)
                return np.array([current_temp, target_temp, mode_encoded], dtype=np.float32)
            
            elif domain == "cover":
                # [position, tilt]
                position = float(attributes.get("position", 0)) / 100.0
                tilt = float(attributes.get("tilt_position", 0)) / 100.0
                return np.array([position, tilt], dtype=np.float32)
            
            elif domain == "media_player":
                # [volume, position, state_encoded]
                volume = float(attributes.get("volume_level", 0))
                position = float(attributes.get("media_position", 0))
                state_map = {"off": 0, "idle": 1, "playing": 2, "paused": 3}
                state_encoded = state_map.get(state, 0)
                return np.array([volume, position, state_encoded], dtype=np.float32)
            
            else:
                # Generic: single numeric value or 0
                try:
                    value = float(state)
                    return np.array([value], dtype=np.float32)
                except (ValueError, TypeError):
                    return np.array([0.0], dtype=np.float32)
                    
        except Exception as e:
            _LOGGER.error("Error converting state to tensor for %s: %s", entity_id, e)
            return None

    async def _evaluate_affected_automations(self, entity_id: str) -> None:
        """Evaluate automations affected by entity state change."""
        for auto_id, automation in self._agentic_automations.items():
            if entity_id in automation.entity_ids and automation.active:
                await self._execute_agentic_automation(auto_id)

    async def _execute_agentic_automation(self, automation_id: str) -> bool:
        """Execute agentic automation logic."""
        automation = self._agentic_automations.get(automation_id)
        if not automation:
            return False
        
        try:
            # Get current entity states
            entity_states = {}
            workbench = self.coresys.robotics_workbench
            
            for entity_id in automation.entity_ids:
                if entity_id in self._entity_tensor_mappings:
                    field_id = self._entity_tensor_mappings[entity_id]
                    field = workbench.tensor_manager.get_tensor_field(field_id)
                    if field and field.data is not None:
                        entity_states[entity_id] = field.data
            
            # Execute neural-symbolic evaluation
            result = await self._neural_symbolic_evaluation(automation, entity_states)
            
            if result.get("should_execute", False):
                actions = result.get("actions", [])
                await self._execute_actions(actions)
                
                automation.execution_count += 1
                automation.last_execution = asyncio.get_event_loop().time()
                
                _LOGGER.info("Executed agentic automation %s", automation_id)
                return True
                
        except Exception as e:
            _LOGGER.error("Error executing agentic automation %s: %s", automation_id, e)
        
        return False

    async def _neural_symbolic_evaluation(self, automation: AgenticAutomation, 
                                        entity_states: Dict[str, Any]) -> Dict[str, Any]:
        """Perform neural-symbolic evaluation of automation logic."""
        # This is a simplified implementation
        # In a full implementation, this would involve:
        # 1. Neural component: transformer/neural network evaluation
        # 2. Symbolic component: rule-based logic evaluation  
        # 3. Fusion of neural and symbolic decisions
        
        control_logic = automation.control_logic
        
        # Evaluate triggers symbolically
        triggers_satisfied = self._evaluate_triggers(control_logic.get("trigger", []), entity_states)
        
        # Evaluate conditions symbolically
        conditions_satisfied = self._evaluate_conditions(control_logic.get("condition", []), entity_states)
        
        # Neural enhancement (placeholder - would be actual neural network)
        neural_confidence = 0.8  # Placeholder confidence score
        
        should_execute = triggers_satisfied and conditions_satisfied and neural_confidence > 0.5
        
        actions = control_logic.get("action", []) if should_execute else []
        
        return {
            "should_execute": should_execute,
            "actions": actions,
            "triggers_satisfied": triggers_satisfied,
            "conditions_satisfied": conditions_satisfied,
            "neural_confidence": neural_confidence
        }

    def _evaluate_triggers(self, triggers: List[Dict[str, Any]], 
                          entity_states: Dict[str, Any]) -> bool:
        """Evaluate automation triggers against entity states."""
        if not triggers:
            return True
        
        # For simplicity, just check if any trigger entity has recent changes
        for trigger in triggers:
            entity_id = trigger.get("entity_id")
            if entity_id and entity_id in entity_states:
                # Check if entity has changed recently (placeholder logic)
                return True
        
        return False

    def _evaluate_conditions(self, conditions: List[Dict[str, Any]], 
                           entity_states: Dict[str, Any]) -> bool:
        """Evaluate automation conditions against entity states."""
        if not conditions:
            return True
        
        # Simplified condition evaluation
        for condition in conditions:
            condition_type = condition.get("condition", "state")
            entity_id = condition.get("entity_id")
            
            if entity_id and entity_id in entity_states:
                # Placeholder condition checking
                return True
        
        return True

    async def _execute_actions(self, actions: List[Dict[str, Any]]) -> None:
        """Execute automation actions."""
        for action in actions:
            try:
                service = action.get("service", "")
                entity_id = action.get("entity_id") or action.get("target", {}).get("entity_id")
                
                if service and entity_id:
                    # This would integrate with HomeAssistant service calls
                    _LOGGER.info("Would execute action: %s on %s", service, entity_id)
                    
                    # Update tensor state to reflect action
                    if isinstance(entity_id, str) and entity_id in self._entity_tensor_mappings:
                        await self._update_tensor_from_action(entity_id, action)
                        
            except Exception as e:
                _LOGGER.error("Error executing action %s: %s", action, e)

    async def _update_tensor_from_action(self, entity_id: str, action: Dict[str, Any]) -> None:
        """Update tensor field based on executed action."""
        # This would update the tensor to reflect the action taken
        # For now, it's a placeholder
        _LOGGER.debug("Updating tensor for %s after action %s", entity_id, action.get("service"))

    def get_kernelization_status(self) -> Dict[str, Any]:
        """Get status of HomeAssistant kernelization."""
        return {
            "total_automations": len(self._agentic_automations),
            "active_automations": sum(1 for a in self._agentic_automations.values() if a.active),
            "kernelized_entities": len(self._entity_tensor_mappings),
            "active_control_loops": len(self._active_control_loops),
            "automations": [
                {
                    "id": auto_id,
                    "name": automation.name,
                    "entity_count": len(automation.entity_ids),
                    "execution_count": automation.execution_count,
                    "active": automation.active
                }
                for auto_id, automation in self._agentic_automations.items()
            ]
        }

    async def activate_automation(self, automation_id: str) -> bool:
        """Activate an agentic automation."""
        if automation_id in self._agentic_automations:
            self._agentic_automations[automation_id].active = True
            _LOGGER.info("Activated agentic automation %s", automation_id)
            return True
        return False

    async def deactivate_automation(self, automation_id: str) -> bool:
        """Deactivate an agentic automation."""
        if automation_id in self._agentic_automations:
            self._agentic_automations[automation_id].active = False
            _LOGGER.info("Deactivated agentic automation %s", automation_id)
            return True
        return False