"""Robotics workbench API endpoints."""

from aiohttp import web
import voluptuous as vol

from ..coresys import CoreSysAttributes
from .utils import api_process, api_process_raw, api_validate


class APIRobotics(CoreSysAttributes):
    """Handle RESTful API for Robotics Workbench."""

    @api_process
    async def info(self, request: web.Request) -> dict:
        """Return workbench information."""
        workbench = self.sys_coresys.robotics_workbench
        
        experiments = workbench.list_experiments()
        hypergraph_summary = workbench.hypergraph_engine.get_hypergraph_summary()
        tensor_config = workbench.tensor_manager.export_tensor_configuration()
        
        return {
            "version": "marduk-robotics-lab-1.0",
            "active_experiment": workbench._active_experiment,
            "monitoring_active": workbench._monitoring_active,
            "experiments": experiments,
            "hypergraph_summary": hypergraph_summary,
            "tensor_summary": {
                "total_fields": tensor_config["total_fields"],
                "total_elements": tensor_config["total_elements"],
                "total_dof": tensor_config["total_degrees_of_freedom"],
            },
            "agents": {
                "total_agents": len(workbench.gguf_manager._agent_states),
                "agent_ids": workbench.gguf_manager.list_agent_states(),
            }
        }

    @api_process
    async def experiments(self, request: web.Request) -> dict:
        """List all experiments."""
        workbench = self.sys_coresys.robotics_workbench
        return {"experiments": workbench.list_experiments()}

    @api_process
    async def create_experiment(self, request: web.Request) -> dict:
        """Create a new experiment."""
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("experiment_id"): str,
                vol.Required("name"): str,
                vol.Optional("description", default=""): str,
                vol.Optional("metadata", default={}): dict,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.create_experiment(
            experiment_id=body["experiment_id"],
            name=body["name"],
            description=body["description"],
            metadata=body["metadata"]
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def experiment_info(self, request: web.Request) -> dict:
        """Get experiment information."""
        experiment_id = request.match_info["experiment_id"]
        workbench = self.sys_coresys.robotics_workbench
        
        status = workbench.get_experiment_status(experiment_id)
        if not status:
            raise web.HTTPNotFound()
        
        return status

    @api_process
    async def start_experiment(self, request: web.Request) -> dict:
        """Start an experiment."""
        experiment_id = request.match_info["experiment_id"]
        workbench = self.sys_coresys.robotics_workbench
        
        success = await workbench.start_experiment(experiment_id)
        return {"result": "ok" if success else "error"}

    @api_process
    async def stop_experiment(self, request: web.Request) -> dict:
        """Stop an experiment."""
        experiment_id = request.match_info["experiment_id"]
        workbench = self.sys_coresys.robotics_workbench
        
        success = await workbench.stop_experiment(experiment_id)
        return {"result": "ok" if success else "error"}

    @api_process
    async def configure_device(self, request: web.Request) -> dict:
        """Configure a device for an experiment."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("device_id"): str,
                vol.Required("device_type"): str,
                vol.Required("name"): str,
                vol.Optional("tensor_dimensions", default=[1, 100]): [int],
                vol.Optional("modalities", default=[]): [str],
                vol.Optional("channels", default=1): int,
                vol.Optional("temporal_length", default=100): int,
                vol.Optional("properties", default={}): dict,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.configure_device(
            experiment_id=experiment_id,
            device_id=body["device_id"],
            device_type=body["device_type"],
            name=body["name"],
            config=body
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def configure_sensor(self, request: web.Request) -> dict:
        """Configure a sensor for an experiment."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("sensor_id"): str,
                vol.Required("sensor_type"): str,
                vol.Required("name"): str,
                vol.Optional("channels", default=1): int,
                vol.Optional("sampling_rate", default=100.0): float,
                vol.Optional("buffer_duration", default=1.0): float,
                vol.Optional("properties", default={}): dict,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.configure_sensor(
            experiment_id=experiment_id,
            sensor_id=body["sensor_id"],
            sensor_type=body["sensor_type"],
            name=body["name"],
            config=body
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def configure_actuator(self, request: web.Request) -> dict:
        """Configure an actuator for an experiment."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("actuator_id"): str,
                vol.Required("actuator_type"): str,
                vol.Required("name"): str,
                vol.Optional("degrees_of_freedom", default=1): int,
                vol.Optional("control_type", default="position"): str,
                vol.Optional("control_history", default=50): int,
                vol.Optional("properties", default={}): dict,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.configure_actuator(
            experiment_id=experiment_id,
            actuator_id=body["actuator_id"],
            actuator_type=body["actuator_type"],
            name=body["name"],
            config=body
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def create_control_loop(self, request: web.Request) -> dict:
        """Create an agentic control loop."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("loop_id"): str,
                vol.Required("name"): str,
                vol.Optional("loop_type", default="neural_symbolic"): str,
                vol.Optional("update_rate", default=10.0): float,
                vol.Optional("properties", default={}): dict,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.create_agentic_control_loop(
            experiment_id=experiment_id,
            loop_id=body["loop_id"],
            name=body["name"],
            config=body
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def connect_components(self, request: web.Request) -> dict:
        """Connect components in the hypergraph."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("source_ids"): [str],
                vol.Required("target_ids"): [str],
                vol.Optional("connection_type", default="data_flow"): str,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.connect_components(
            experiment_id=experiment_id,
            source_ids=body["source_ids"],
            target_ids=body["target_ids"],
            connection_type=body["connection_type"]
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def live_visualization(self, request: web.Request) -> dict:
        """Get live tensor field visualization data."""
        workbench = self.sys_coresys.robotics_workbench
        return workbench.get_live_visualization_data()

    @api_process
    async def hypergraph_structure(self, request: web.Request) -> dict:
        """Get complete hypergraph structure."""
        workbench = self.sys_coresys.robotics_workbench
        return workbench.hypergraph_engine.export_hypergraph_structure()

    @api_process
    async def tensor_fields(self, request: web.Request) -> dict:
        """List all tensor fields."""
        workbench = self.sys_coresys.robotics_workbench
        return {
            "tensor_fields": workbench.tensor_manager.list_tensor_fields(),
            "statistics": workbench.tensor_manager.get_all_tensor_statistics(),
        }

    @api_process
    async def tensor_field_stats(self, request: web.Request) -> dict:
        """Get tensor field statistics."""
        field_id = request.match_info["field_id"]
        workbench = self.sys_coresys.robotics_workbench
        
        stats = workbench.tensor_manager.get_tensor_statistics(field_id)
        if not stats:
            raise web.HTTPNotFound()
        
        return stats

    @api_process
    async def agents_list(self, request: web.Request) -> dict:
        """List all agent states."""
        workbench = self.sys_coresys.robotics_workbench
        agent_ids = workbench.gguf_manager.list_agent_states()
        
        agents_info = []
        for agent_id in agent_ids:
            agent_state = workbench.gguf_manager.get_agent_state(agent_id)
            if agent_state:
                agents_info.append({
                    "agent_id": agent_id,
                    "name": agent_state.name,
                    "metadata": agent_state.metadata,
                    "tensor_count": len(agent_state.tensors),
                    "device_config_count": len(agent_state.device_configs),
                    "control_loop_count": len(agent_state.control_loops),
                })
        
        return {"agents": agents_info}

    @api_process
    async def agent_info(self, request: web.Request) -> dict:
        """Get agent information and tensor schema."""
        agent_id = request.match_info["agent_id"]
        workbench = self.sys_coresys.robotics_workbench
        
        agent_state = workbench.gguf_manager.get_agent_state(agent_id)
        if not agent_state:
            raise web.HTTPNotFound()
        
        tensor_schema = workbench.gguf_manager.get_agent_tensor_schema(agent_id)
        
        return {
            "agent_id": agent_id,
            "name": agent_state.name,
            "metadata": agent_state.metadata,
            "tensors": agent_state.tensors,
            "device_configs": agent_state.device_configs,
            "control_loops": agent_state.control_loops,
            "tensor_schema": tensor_schema,
        }

    @api_process
    async def export_experiment(self, request: web.Request) -> dict:
        """Export experiment as GGUF/P-System schema."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Optional("export_path", default="/tmp/robotics_export"): str,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        from pathlib import Path
        export_path = Path(body["export_path"])
        
        success = await workbench.export_experiment(experiment_id, export_path)
        
        return {
            "result": "ok" if success else "error",
            "export_path": str(export_path) if success else None,
        }

    @api_process
    async def kernelize_homeassistant(self, request: web.Request) -> dict:
        """Kernelize HomeAssistant entity into agentic control."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("entity_id"): str,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.kernelize_homeassistant_entity(
            entity_id=body["entity_id"],
            experiment_id=experiment_id
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def kernelize_homeassistant_automation(self, request: web.Request) -> dict:
        """Kernelize HomeAssistant automation into agentic control loop."""
        experiment_id = request.match_info["experiment_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("automation_config"): dict,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        automation_id = await workbench.kernelize_homeassistant_automation(
            automation_config=body["automation_config"],
            experiment_id=experiment_id
        )
        
        return {"result": "ok", "automation_id": automation_id}

    @api_process
    async def homeassistant_status(self, request: web.Request) -> dict:
        """Get HomeAssistant kernelization status."""
        workbench = self.sys_coresys.robotics_workbench
        return workbench.get_kernelization_status()

    @api_process
    async def control_ha_automation(self, request: web.Request) -> dict:
        """Control kernelized HomeAssistant automation."""
        automation_id = request.match_info["automation_id"]
        
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("action"): vol.In(["activate", "deactivate"]),
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        
        if body["action"] == "activate":
            success = await workbench.activate_ha_automation(automation_id)
        else:
            success = await workbench.deactivate_ha_automation(automation_id)
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def update_ha_entity_state(self, request: web.Request) -> dict:
        """Update HomeAssistant entity state in tensor field."""
        body = await api_validate(
            request,
            vol.Schema({
                vol.Required("entity_id"): str,
                vol.Required("state_data"): dict,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        success = await workbench.update_homeassistant_entity_state(
            entity_id=body["entity_id"],
            state_data=body["state_data"]
        )
        
        return {"result": "ok" if success else "error"}

    @api_process
    async def p_system_schema(self, request: web.Request) -> dict:
        """Get complete P-System compatible schema export."""
        workbench = self.sys_coresys.robotics_workbench
        
        # Get latest meta-cognitive report
        meta_report = workbench.get_realtime_meta_cognitive_report()
        
        if not meta_report:
            # Generate on-demand if no report available
            await workbench.meta_cognitive._generate_realtime_report()
            meta_report = workbench.get_realtime_meta_cognitive_report()
        
        # Create comprehensive meta-cognitive schema
        schema = {
            "p_system_version": "3.0",
            "marduk_robotics_lab_version": "1.0", 
            "export_timestamp": asyncio.get_event_loop().time(),
            "meta_cognitive_report": meta_report,
            "self_descriptive": True,
            "distributed_cognition_ready": True,
            "neural_symbolic_integration": True,
            "real_time_adaptation": True,
            "system_health": workbench.get_system_health_metrics()
        }
        
        return schema

    @api_process
    async def meta_cognitive_report(self, request: web.Request) -> dict:
        """Get latest meta-cognitive report."""
        workbench = self.sys_coresys.robotics_workbench
        
        report = workbench.get_realtime_meta_cognitive_report()
        if not report:
            return {"error": "No meta-cognitive report available"}
        
        return {"report": report}

    @api_process
    async def meta_cognitive_history(self, request: web.Request) -> dict:
        """Get meta-cognitive report history."""
        limit = int(request.query.get("limit", 100))
        workbench = self.sys_coresys.robotics_workbench
        
        history = workbench.get_meta_cognitive_history(limit)
        return {"history": history, "count": len(history)}

    @api_process
    async def system_health(self, request: web.Request) -> dict:
        """Get comprehensive system health metrics."""
        workbench = self.sys_coresys.robotics_workbench
        return workbench.get_system_health_metrics()

    @api_process
    async def export_psystem_schema(self, request: web.Request) -> dict:
        """Export complete P-System schema to file."""
        body = await api_validate(
            request,
            vol.Schema({
                vol.Optional("export_path", default="/tmp/psystem_export"): str,
            })
        )
        
        workbench = self.sys_coresys.robotics_workbench
        from pathlib import Path
        export_path = Path(body["export_path"])
        
        success = await workbench.export_psystem_schema(export_path)
        
        return {
            "result": "ok" if success else "error",
            "export_path": str(export_path) if success else None,
        }