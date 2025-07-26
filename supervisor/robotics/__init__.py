"""Robotics Engineering Workbench for Marduk's Robotics Lab."""

from .workbench import RoboticsWorkbench
from .gguf_integration import GGUFManager
from .hypergraph import HypergraphEngine, TensorDimensionSpec, MiddlewareInterface
from .tensor_manager import TensorManager
from .ha_kernelizer import HomeAssistantKernelizer
from .meta_cognitive import MetaCognitiveReporter
from .middleware import (
    MiddlewareComponent,
    HardwareInterfaceComponent, 
    DataProcessorComponent,
    ControllerComponent,
    AIAgentComponent,
    ComponentRegistry
)

__all__ = [
    "RoboticsWorkbench",
    "GGUFManager", 
    "HypergraphEngine",
    "TensorManager",
    "HomeAssistantKernelizer",
    "MetaCognitiveReporter",
    "TensorDimensionSpec",
    "MiddlewareInterface",
    "MiddlewareComponent",
    "HardwareInterfaceComponent",
    "DataProcessorComponent", 
    "ControllerComponent",
    "AIAgentComponent",
    "ComponentRegistry",
]