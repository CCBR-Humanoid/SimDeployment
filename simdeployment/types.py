"""Types for interacting with the simulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ActuatorType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"

@dataclass
class ActuatorMetadata:
    """Metadata for a single joint in the simulation."""
    joint_name: str
    actuator_type: ActuatorType
    kp: float
    kd: float
    max_torque: float

@dataclass
class ModelMetadata:
    """Metadata about the simulation model."""
    joint_name_to_actuator_metadata: dict[str, ActuatorMetadata]
    start_height: float
    initial_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    freejoint: bool = False


@dataclass
class MuJoCoMetadata:
    """Metadata for the MuJoCo simulation."""
    dt: float = 0.001 # simulation timestep (1000Hz)
    control_frequency: float = 50.0 # policy frequency (50Hz)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    integrator: str = "implicitfast"
    solver: str = "cg"

@dataclass
class ActuatorState:
    """State of a single actuator in the simulation."""
    position: float
    velocity: float
    torque: float

@dataclass
class ActuatorCommand:
    position: float | None
    velocity: float | None
    torque: float | None

@dataclass
class ConfigureActuatorRequest:
    kp: int
    kd: int
    max_torque: float
