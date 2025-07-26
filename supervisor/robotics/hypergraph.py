"""Hypergraph engine for modular robotics workbench components."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from ..coresys import CoreSysAttributes

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass
class HypergraphNode:
    """A node in the robotics hypergraph."""
    
    node_id: str
    node_type: str  # "device", "sensor", "actuator", "agent", "control_loop"
    name: str
    properties: Dict[str, Any]
    tensor_dimensions: Optional[List[int]] = None
    degrees_of_freedom: int = 0
    modalities: List[str] = None

    def __post_init__(self):
        """Initialize computed properties."""
        if self.modalities is None:
            self.modalities = []
        if self.tensor_dimensions:
            self.degrees_of_freedom = len(self.tensor_dimensions)


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
                       tensor_dims: Optional[List[int]] = None, 
                       modalities: Optional[List[str]] = None,
                       properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a device node to the hypergraph."""
        node_id = f"device_{device_id}"
        
        node = HypergraphNode(
            node_id=node_id,
            node_type="device",
            name=name,
            properties=properties or {"device_type": device_type, "device_id": device_id},
            tensor_dimensions=tensor_dims,
            modalities=modalities or []
        )
        
        self._nodes[node_id] = node
        self._node_edges[node_id] = set()
        
        _LOGGER.info("Added device node: %s (%s)", node_id, name)
        return node_id

    def add_sensor_node(self, sensor_id: str, sensor_type: str, name: str,
                       channels: int = 1, sampling_rate: Optional[float] = None,
                       tensor_dims: Optional[List[int]] = None,
                       properties: Optional[Dict[str, Any]] = None) -> str:
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
            
        # Default tensor dimensions for sensors
        if not tensor_dims:
            tensor_dims = [channels, 1]  # [channels, time_samples]
        
        node = HypergraphNode(
            node_id=node_id,
            node_type="sensor",
            name=name,
            properties=props,
            tensor_dimensions=tensor_dims,
            modalities=[sensor_type]
        )
        
        self._nodes[node_id] = node
        self._node_edges[node_id] = set()
        
        _LOGGER.info("Added sensor node: %s (%s)", node_id, name)
        return node_id

    def add_actuator_node(self, actuator_id: str, actuator_type: str, name: str,
                         dof: int = 1, control_type: str = "position",
                         tensor_dims: Optional[List[int]] = None,
                         properties: Optional[Dict[str, Any]] = None) -> str:
        """Add an actuator node to the hypergraph."""
        node_id = f"actuator_{actuator_id}"
        
        props = properties or {}
        props.update({
            "actuator_type": actuator_type,
            "actuator_id": actuator_id,
            "degrees_of_freedom": dof,
            "control_type": control_type,
        })
        
        # Default tensor dimensions for actuators
        if not tensor_dims:
            tensor_dims = [dof, 1]  # [dof, control_values]
        
        node = HypergraphNode(
            node_id=node_id,
            node_type="actuator", 
            name=name,
            properties=props,
            tensor_dimensions=tensor_dims,
            degrees_of_freedom=dof
        )
        
        self._nodes[node_id] = node
        self._node_edges[node_id] = set()
        
        _LOGGER.info("Added actuator node: %s (%s)", node_id, name)
        return node_id

    def add_agent_node(self, agent_id: str, name: str, agent_type: str = "autonomous",
                      state_dims: Optional[List[int]] = None,
                      properties: Optional[Dict[str, Any]] = None) -> str:
        """Add an agent node to the hypergraph."""
        node_id = f"agent_{agent_id}"
        
        props = properties or {}
        props.update({
            "agent_type": agent_type,
            "agent_id": agent_id,
        })
        
        # Default state tensor dimensions
        if not state_dims:
            state_dims = [1, 256]  # [batch, hidden_state]
        
        node = HypergraphNode(
            node_id=node_id,
            node_type="agent",
            name=name,
            properties=props,
            tensor_dimensions=state_dims,
            modalities=["cognitive", "neural"]
        )
        
        self._nodes[node_id] = node
        self._node_edges[node_id] = set()
        
        _LOGGER.info("Added agent node: %s (%s)", node_id, name)
        return node_id

    def add_control_loop_node(self, loop_id: str, name: str, loop_type: str = "pid",
                             update_rate: float = 10.0,
                             properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a control loop node to the hypergraph."""
        node_id = f"control_{loop_id}"
        
        props = properties or {}
        props.update({
            "loop_type": loop_type,
            "loop_id": loop_id,
            "update_rate": update_rate,
        })
        
        node = HypergraphNode(
            node_id=node_id,
            node_type="control_loop",
            name=name,
            properties=props,
            tensor_dimensions=[1, 1],  # [input, output]
        )
        
        self._nodes[node_id] = node
        self._node_edges[node_id] = set()
        
        _LOGGER.info("Added control loop node: %s (%s)", node_id, name)
        return node_id

    def connect_nodes(self, node_ids: List[str], edge_type: str = "data_flow", 
                     name: str = "", properties: Optional[Dict[str, Any]] = None,
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

    def get_node(self, node_id: str) -> Optional[HypergraphNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[HypergraphEdge]:
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
        for node in self._nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        edge_types = {}
        for edge in self._edges.values():
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        
        total_dof = sum(node.degrees_of_freedom for node in self._nodes.values())
        
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "total_degrees_of_freedom": total_dof,
            "complexity_metric": len(self._nodes) * len(self._edges) + total_dof,
        }

    def export_hypergraph_structure(self) -> Dict[str, Any]:
        """Export the complete hypergraph structure."""
        nodes_data = {}
        for node_id, node in self._nodes.items():
            nodes_data[node_id] = {
                "node_type": node.node_type,
                "name": node.name,
                "properties": node.properties,
                "tensor_dimensions": node.tensor_dimensions,
                "degrees_of_freedom": node.degrees_of_freedom,
                "modalities": node.modalities,
            }
        
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