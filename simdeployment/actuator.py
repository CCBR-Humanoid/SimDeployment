from abc import ABC, abstractmethod

import numpy as np

from simdeployment.types import ActuatorCommand
import logging

logger = logging.getLogger(__name__)

class Actuator(ABC):
    """Abstract base class for actuators that return a torque for a single joint."""

    @abstractmethod
    def get_torque(
        self,
        cmd: ActuatorCommand,
        *,
        qpos: float,
        qvel: float,
    ) -> float:
        """Get the torque for the actuator given the command and current state.

        Args:
            cmd: The command for the actuator.
            qpos: The current position of the joint.
            qvel: The current velocity of the joint.

        Returns:
            The torque to apply to the joint.
        """
        pass

    def configure(self, **kwargs: float) -> None:
        """Update actuator configuration parameters."""
        pass


class PositionActuator(Actuator):
    def __init__(self, kp: float, kd: float, max_torque: float) -> None:
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque

    def get_torque(
        self,
        cmd: ActuatorCommand,
    *,
        qpos: float,
        qvel: float,
    ) -> float:
        if cmd.velocity is not None or cmd.torque is not None:
            logger.warning("PositionActuator received command with velocity or torque setpoint. Ignoring these values.")

        if cmd.position is None:
            logger.warning("PositionActuator received command with no position setpoint. Returning 0 torque.")
            return 0.0

        position_error = cmd.position - qpos
        velocity_error = -qvel
        torque = self.kp * position_error + self.kd * velocity_error
        return float(np.clip(torque, -self.max_torque, self.max_torque))

    def configure(self, **kwargs: float) -> None:
        if "kp" in kwargs:
            self.kp = kwargs["kp"]
        if "kd" in kwargs:
            self.kd = kwargs["kd"]
        if "max_torque" in kwargs:
            self.max_torque = kwargs["max_torque"]


class VelocityActuator(Actuator):
    def __init__(self, kp: float, kd: float, max_torque: float) -> None:
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque

    def get_torque(
        self,
        cmd: ActuatorCommand,
        *,
        qpos: float,
        qvel: float,
    ) -> float:
        if cmd.torque is not None:
            logger.warning("VelocityActuator received command with torque setpoint. Ignoring this value.")

        if cmd.position is None:
            logger.warning("VelocityActuator received command with no position setpoint. Returning 0 torque.")
            return 0.0

        if cmd.velocity is None:
            logger.warning("VelocityActuator received command with no velocity setpoint. Returning 0 torque.")
            return 0.0

        position_error = cmd.position - qpos
        velocity_error = cmd.velocity - qvel
        torque = self.kp * position_error + self.kd * velocity_error
        return float(np.clip(torque, -self.max_torque, self.max_torque))

    def configure(self, **kwargs: float) -> None:
        if "kp" in kwargs:
            self.kp = kwargs["kp"]
        if "kd" in kwargs:
            self.kd = kwargs["kd"]
        if "max_torque" in kwargs:
            self.max_torque = kwargs["max_torque"]
