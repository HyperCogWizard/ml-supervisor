"""Hypergraph engine for modular robotics workbench components."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Set
from uuid import uuid4

from ..coresys import CoreSysAttributes

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass
class TensorDimensionSpec:
    """Structured specification for tensor dimensions."""

    degrees_of_freedom: int
    channels: int
    modalities: List[str]
    temporal_length: int | None = None
    batch_size: int = 1
    additional_dims: List[int] = field(default_factory=list)

    def get_tensor_shape(self) -> List[int]:
        """Get the complete tensor shape."""
        shape = [self.batch_size, self.degrees_of_freedom, self.channels]
        if self.temporal_length:
            shape.append(self.temporal_length)
        shape.extend(self.additional_dims)
        return shape

    def get_total_elements(self) -> int:
        """Get total number of tensor elements."""
        shape = self.get_tensor_shape()
        return int(sum(shape)) if shape else 0


@dataclass
class MiddlewareInterface:
    """Standard interface specification for middleware components."""

    interface_type: str  # "sensor_input", "actuator_output", "control_signal", "data_stream"
    data_format: str  # "tensor", "json", "binary", "event"
    communication_protocol: str  # "direct", "message_queue", "shared_memory", "network"
    update_frequency: float | None = None


@dataclass
class HypergraphNode:
    """A node in the robotics hypergraph representing a middleware component."""

    node_id: str
    node_type: str  # "device", "sensor", "actuator", "agent", "control_loop", "middleware"
    middleware_type: str  # "hardware_interface", "data_processor", "controller", "ai_agent"
    name: str
    properties: Dict[str, Any]
    tensor_spec: TensorDimensionSpec | None = None
    component_interface: MiddlewareInterface | None = None

    # Legacy support
    tensor_dimensions: List[int] | None = None
    degrees_of_freedom: int = 0
    modalities: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize computed properties and handle legacy compatibility."""
        # Legacy compatibility
        if self.tensor_dimensions and not self.tensor_spec:
            # Convert legacy tensor_dimensions to TensorDimensionSpec
            dof = len(self.tensor_dimensions) if len(self.tensor_dimensions) > 1 else self.tensor_dimensions[0] if self.tensor_dimensions else 1
            channels = self.tensor_dimensions[0] if self.tensor_dimensions else 1
            self.tensor_spec = TensorDimensionSpec(
                degrees_of_freedom=dof,
                channels=channels,
                modalities=self.modalities,
                additional_dims=self.tensor_dimensions[2:] if len(self.tensor_dimensions) > 2 else []
            )

        # Update legacy fields from tensor_spec
        if self.tensor_spec:
            self.degrees_of_freedom = self.tensor_spec.degrees_of_freedom
            if not self.modalities:
                self.modalities = self.tensor_spec.modalities
            if not self.tensor_dimensions:
                self.tensor_dimensions = self.tensor_spec.get_tensor_shape()

        # Set default middleware_type if not specified
        if not hasattr(self, 'middleware_type') or not self.middleware_type:
            self.middleware_type = self._infer_middleware_type()

    def _infer_middleware_type(self) -> str:
        """Infer middleware type from node_type."""
        type_mapping = {
            "sensor": "hardware_interface",
            "actuator": "hardware_interface",
            "device": "hardware_interface",
            "agent": "ai_agent",
            "control_loop": "controller"
        }
        return type_mapping.get(self.node_type, "data_processor")


@dataclass
class HypergraphEdge:
    """A hyperedge connecting multiple nodes."""

    edge_id: str
    edge_type: str  # "data_flow", "control", "dependency", "neural_connection"
    nodes: Set[str]  # Node IDs
    properties: Dict[str, Any]
    weight: float = 1.0
    directional: bool = False


class HypergraphEngine(CoreSysAttributes):
    """Engine for managing hypergraph-encoded workbench components."""

    def __init__(self, coresys):
        """Initialize hypergraph engine."""
        self.coresys = coresys
        self._nodes: Dict[str, HypergraphNode] = {}
        self._edges: Dict[str, HypergraphEdge] = {}
        self._node_edges: Dict[str, Set[str]] = {}  # Node ID -> Edge IDs

    def add_device_node(self, device_id: str, device_type: str, name: str,
                       tensor_dims: List[int] | None = None,
                       modalities: List[str] | None = None,
                       properties: Dict[str, Any] | None = None,
                       degrees_of_freedom: int | None = None,
                       channels: int | None = None) -> str:
        """Add a device node to the hypergraph."""
        node_id = f"device_{device_id}"

        # Create tensor specification
        tensor_spec = None
        if degrees_of_freedom or channels or tensor_dims:
            dof = degrees_of_freedom or (len(tensor_dims) if tensor_dims else 1)
            ch = channels or (tensor_dims[0] if tensor_dims else 1)
            tensor_spec = TensorDimensionSpec(
                degrees_of_freedom=dof,
                channels=ch,
                modalities=modalities or [device_type],
                additional_dims=tensor_dims[2:] if tensor_dims and len(tensor_dims) > 2 else []
            )

        # Create component interface
        interface = MiddlewareInterface(
            interface_type="device_interface",
            data_format="tensor",
            communication_protocol="direct"
        )

        node = HypergraphNode(
            node_id=node_id,
            node_type="device",
            middleware_type="hardware_interface",
            name=name,
            properties=properties or {"device_type": device_type, "device_id": device_id},
            tensor_spec=tensor_spec,
            component_interface=interface,
            tensor_dimensions=tensor_dims,  # Legacy support
            modalities=modalities or []
        )

        self._nodes[node_id] = node
        self._node_edges[node_id] = set()

        _LOGGER.info("Added device node: %s (%s)", node_id, name)
        return node_id

    def add_sensor_node(self, sensor_id: str, sensor_type: str, name: str,
                       channels: int = 1, sampling_rate: float | None = None,
                       tensor_dims: List[int] | None = None,
                       properties: Dict[str, Any] | None = None,
                       temporal_length: int | None = None) -> str:
        """Add a sensor node to the hypergraph."""
        node_id = f"sensor_{sensor_id}"

        props = properties or {}
        props.update({
            "sensor_type": sensor_type,
            "sensor_id": sensor_id,
            "channels": channels,
        })
        if sampling_rate:
            props["sampling_rate"] = sampling_rate

        # Create structured tensor specification
        tensor_spec = TensorDimensionSpec(
            degrees_of_freedom=1,  # Sensors typically have 1 DoF (measurement)
            channels=channels,
            modalities=[sensor_type],
            temporal_length=temporal_length or (tensor_dims[1] if tensor_dims and len(tensor_dims) > 1 else 100)
        )

        # Create component interface
        interface = MiddlewareInterface(
            interface_type="sensor_input",
            data_format="tensor",
            communication_protocol="direct",
            update_frequency=sampling_rate
        )

        node = HypergraphNode(
            node_id=node_id,
            node_type="sensor",
            middleware_type="hardware_interface",
            name=name,
            properties=props,
            tensor_spec=tensor_spec,
            component_interface=interface,
            tensor_dimensions=tensor_dims or [channels, temporal_length or 100],  # Legacy support
            modalities=[sensor_type]
        )

        self._nodes[node_id] = node
        self._node_edges[node_id] = set()

        _LOGGER.info("Added sensor node: %s (%s)", node_id, name)
        return node_id

    def add_actuator_node(self, actuator_id: str, actuator_type: str, name: str,
                         dof: int = 1, control_type: str = "position",
                         tensor_dims: List[int] | None = None,
                         properties: Dict[str, Any] | None = None,
                         channels: int | None = None) -> str:
        """Add an actuator node to the hypergraph."""
        node_id = f"actuator_{actuator_id}"

        props = properties or {}
        props.update({
            "actuator_type": actuator_type,
            "actuator_id": actuator_id,
            "degrees_of_freedom": dof,
            "control_type": control_type,
        })

        # Create structured tensor specification
        actuator_channels = channels or dof  # Default channels = DoF
        tensor_spec = TensorDimensionSpec(
            degrees_of_freedom=dof,
            channels=actuator_channels,
            modalities=[control_type, actuator_type],
            temporal_length=tensor_dims[1] if tensor_dims and len(tensor_dims) > 1 else 50
        )

        # Create component interface
        interface = MiddlewareInterface(
            interface_type="actuator_output",
            data_format="tensor",
            communication_protocol="direct"
        )

        node = HypergraphNode(
            node_id=node_id,
            node_type="actuator",
            middleware_type="hardware_interface",
            name=name,
            properties=props,
            tensor_spec=tensor_spec,
            component_interface=interface,
            tensor_dimensions=tensor_dims or [dof, 50],  # Legacy support
            degrees_of_freedom=dof
        )

        self._nodes[node_id] = node
        self._node_edges[node_id] = set()

        _LOGGER.info("Added actuator node: %s (%s)", node_id, name)
        return node_id

    def add_agent_node(self, agent_id: str, name: str, agent_type: str = "autonomous",
                      state_dims: List[int] | None = None,
                      properties: Dict[str, Any] | None = None,
                      hidden_size: int = 256) -> str:
        """Add an agent node to the hypergraph."""
        node_id = f"agent_{agent_id}"

        props = properties or {}
        props.update({
            "agent_type": agent_type,
            "agent_id": agent_id,
        })

        # Create structured tensor specification for agent state
        tensor_spec = TensorDimensionSpec(
            degrees_of_freedom=1,  # Agents have complex state, DoF=1 for simplicity
            channels=hidden_size,
            modalities=["cognitive", "neural", "symbolic"],
            temporal_length=state_dims[1] if state_dims and len(state_dims) > 1 else None
        )

        # Create component interface
        interface = MiddlewareInterface(
            interface_type="control_signal",
            data_format="tensor",
            communication_protocol="message_queue"
        )

        node = HypergraphNode(
            node_id=node_id,
            node_type="agent",
            middleware_type="ai_agent",
            name=name,
            properties=props,
            tensor_spec=tensor_spec,
            component_interface=interface,
            tensor_dimensions=state_dims or [1, hidden_size],  # Legacy support
            modalities=["cognitive", "neural", "symbolic"]
        )

        self._nodes[node_id] = node
        self._node_edges[node_id] = set()

        _LOGGER.info("Added agent node: %s (%s)", node_id, name)
        return node_id

    def add_control_loop_node(self, loop_id: str, name: str, loop_type: str = "pid",
                             update_rate: float = 10.0,
                             properties: Dict[str, Any] | None = None) -> str:
        """Add a control loop node to the hypergraph."""
        node_id = f"control_{loop_id}"

        props = properties or {}
        props.update({
            "loop_type": loop_type,
            "loop_id": loop_id,
            "update_rate": update_rate,
        })

        # Create structured tensor specification
        tensor_spec = TensorDimensionSpec(
            degrees_of_freedom=1,  # Control loops have 1 DoF (control signal)
            channels=1,  # Single control channel
            modalities=[loop_type, "control"],
            temporal_length=100  # Control history buffer
        )

        # Create component interface
        interface = MiddlewareInterface(
            interface_type="control_signal",
            data_format="tensor",
            communication_protocol="direct",
            update_frequency=update_rate
        )

        node = HypergraphNode(
            node_id=node_id,
            node_type="control_loop",
            middleware_type="controller",
            name=name,
            properties=props,
            tensor_spec=tensor_spec,
            component_interface=interface,
            tensor_dimensions=[1, 1],  # Legacy support
        )

        self._nodes[node_id] = node
        self._node_edges[node_id] = set()

        _LOGGER.info("Added control loop node: %s (%s)", node_id, name)
        return node_id

    def add_middleware_component(self, component_id: str, middleware_type: str, name: str,
                                tensor_spec: TensorDimensionSpec,
                                interface: MiddlewareInterface,
                                properties: Dict[str, Any] | None = None) -> str:
        """Add a general middleware component node to the hypergraph."""
        node_id = f"middleware_{component_id}"

        node = HypergraphNode(
            node_id=node_id,
            node_type="middleware",
            middleware_type=middleware_type,
            name=name,
            properties=properties or {"component_id": component_id},
            tensor_spec=tensor_spec,
            component_interface=interface
        )

        self._nodes[node_id] = node
        self._node_edges[node_id] = set()

        _LOGGER.info("Added middleware component: %s (%s)", node_id, name)
        return node_id

    def connect_nodes(self, node_ids: List[str], edge_type: str = "data_flow",
                     name: str = "", properties: Dict[str, Any] | None = None,
                     weight: float = 1.0, directional: bool = False) -> str:
        """Create a hyperedge connecting multiple nodes."""
        edge_id = str(uuid4())

        # Validate nodes exist
        for node_id in node_ids:
            if node_id not in self._nodes:
                raise ValueError(f"Node {node_id} not found")

        edge = HypergraphEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            nodes=set(node_ids),
            properties=properties or {"name": name},
            weight=weight,
            directional=directional
        )

        self._edges[edge_id] = edge

        # Update node-edge mappings
        for node_id in node_ids:
            self._node_edges[node_id].add(edge_id)

        _LOGGER.info("Connected nodes %s with edge %s (%s)", node_ids, edge_id, edge_type)
        return edge_id

    def get_node(self, node_id: str) -> HypergraphNode | None:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> HypergraphEdge | None:
        """Get edge by ID."""
        return self._edges.get(edge_id)

    def get_connected_nodes(self, node_id: str) -> List[HypergraphNode]:
        """Get all nodes connected to the given node."""
        if node_id not in self._node_edges:
            return []

        connected_nodes = []
        for edge_id in self._node_edges[node_id]:
            edge = self._edges[edge_id]
            for connected_node_id in edge.nodes:
                if connected_node_id != node_id:
                    connected_nodes.append(self._nodes[connected_node_id])

        return connected_nodes

    def get_nodes_by_type(self, node_type: str) -> List[HypergraphNode]:
        """Get all nodes of a specific type."""
        return [node for node in self._nodes.values() if node.node_type == node_type]

    def get_hypergraph_summary(self) -> Dict[str, Any]:
        """Get summary of the hypergraph structure."""
        node_types = {}
        middleware_types = {}
        for node in self._nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
            if hasattr(node, 'middleware_type') and node.middleware_type:
                middleware_types[node.middleware_type] = middleware_types.get(node.middleware_type, 0) + 1

        edge_types = {}
        for edge in self._edges.values():
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        total_dof = sum(node.degrees_of_freedom for node in self._nodes.values())
        total_channels = sum(node.tensor_spec.channels if node.tensor_spec else 0 for node in self._nodes.values())

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": node_types,
            "middleware_types": middleware_types,
            "edge_types": edge_types,
            "total_degrees_of_freedom": total_dof,
            "total_channels": total_channels,
            "complexity_metric": len(self._nodes) * len(self._edges) + total_dof + total_channels,
        }

    def export_hypergraph_structure(self) -> Dict[str, Any]:
        """Export the complete hypergraph structure."""
        nodes_data = {}
        for node_id, node in self._nodes.items():
            node_data = {
                "node_type": node.node_type,
                "middleware_type": getattr(node, 'middleware_type', ''),
                "name": node.name,
                "properties": node.properties,
                "degrees_of_freedom": node.degrees_of_freedom,
                "modalities": node.modalities,
                # Legacy support
                "tensor_dimensions": node.tensor_dimensions,
            }

            # Add tensor specification if available
            if hasattr(node, 'tensor_spec') and node.tensor_spec:
                node_data["tensor_spec"] = {
                    "degrees_of_freedom": node.tensor_spec.degrees_of_freedom,
                    "channels": node.tensor_spec.channels,
                    "modalities": node.tensor_spec.modalities,
                    "temporal_length": node.tensor_spec.temporal_length,
                    "batch_size": node.tensor_spec.batch_size,
                    "additional_dims": node.tensor_spec.additional_dims,
                    "tensor_shape": node.tensor_spec.get_tensor_shape(),
                }

            # Add component interface if available
            if hasattr(node, 'component_interface') and node.component_interface:
                node_data["component_interface"] = {
                    "interface_type": node.component_interface.interface_type,
                    "data_format": node.component_interface.data_format,
                    "communication_protocol": node.component_interface.communication_protocol,
                    "update_frequency": node.component_interface.update_frequency,
                }

            nodes_data[node_id] = node_data

        edges_data = {}
        for edge_id, edge in self._edges.items():
            edges_data[edge_id] = {
                "edge_type": edge.edge_type,
                "nodes": list(edge.nodes),
                "properties": edge.properties,
                "weight": edge.weight,
                "directional": edge.directional,
            }

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "summary": self.get_hypergraph_summary(),
        }

    def clear_hypergraph(self) -> None:
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._node_edges.clear()
        _LOGGER.info("Cleared hypergraph structure")

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connected edges."""
        if node_id not in self._nodes:
            return False

        # Remove all edges connected to this node
        edges_to_remove = list(self._node_edges.get(node_id, set()))
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove the node
        del self._nodes[node_id]
        if node_id in self._node_edges:
            del self._node_edges[node_id]

        _LOGGER.info("Removed node: %s", node_id)
        return True

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        if edge_id not in self._edges:
            return False

        edge = self._edges[edge_id]

        # Remove edge from node mappings
        for node_id in edge.nodes:
            if node_id in self._node_edges:
                self._node_edges[node_id].discard(edge_id)

        # Remove the edge
        del self._edges[edge_id]

        _LOGGER.info("Removed edge: %s", edge_id)
        return True
