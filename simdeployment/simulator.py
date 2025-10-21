"""Wrapper around MuJoCo simulator."""

import asyncio
import logging
import time
from pathlib import Path

import mujoco
import numpy as np
from kmv.app.viewer import QtViewer

from simdeployment.actuator import Actuator, PositionActuator, VelocityActuator
from simdeployment.types import (
    ActuatorCommand,
    ActuatorState,
    ActuatorType,
    ConfigureActuatorRequest,
    ModelMetadata,
    MuJoCoMetadata,
)

logger = logging.getLogger(__name__)

"""
What this does:
* Configures our simulation + model (so gravity, integrator, other sim params)
    * probably in our class __init__
* Configures how our actuators produce torque (so PD control, position or velocity targets)
* Define getters for sensor data (so joint positions, velocities, IMU data)
* Define setter for actuator commands (target position, velocity, or whatever)
* Step the simulation forward in time
* Render a live view of the simulation
* Reset the simulation to initial conditions


"""


def get_integrator(integrator: str) -> mujoco.mjtIntegrator:
    match integrator.lower():
        case "euler":
            return mujoco.mjtIntegrator.mjINT_EULER
        case "implicit":
            return mujoco.mjtIntegrator.mjINT_IMPLICIT
        case "implicitfast":
            return mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        case "rk4":
            return mujoco.mjtIntegrator.mjINT_RK4
        case _:
            raise ValueError(f"Invalid integrator: {integrator}")


def get_solver(solver: str) -> mujoco.mjtSolver:
    match solver.lower():
        case "cg":
            return mujoco.mjtSolver.mjSOL_CG
        case "newton":
            return mujoco.mjtSolver.mjSOL_NEWTON
        case _:
            raise ValueError(f"Invalid solver: {solver}")


class MujocoSimulator:
    def __init__(self, model_path: Path, model_metadata: ModelMetadata, simulation_metadata: MuJoCoMetadata) -> None:
        self._model_path = model_path

        # load model from string path
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))

        # Configure the simulation parameters
        self._model.opt.timestep = simulation_metadata.dt
        self._model.opt.gravity = simulation_metadata.gravity
        self._model.opt.integrator = get_integrator(simulation_metadata.integrator)
        self._model.opt.solver = get_solver(simulation_metadata.solver)

        # make the mujoco data
        self._data = mujoco.MjData(self._model)

        # If we have a freejoint in the model, qpos has an addition 7dim for xyz and wxyz (quaternion) of the base
        self._freejoint = model_metadata.freejoint
        self._start_height = model_metadata.start_height
        self._initial_quat = model_metadata.initial_quat
        self._dt = simulation_metadata.dt
        self._control_frequency = simulation_metadata.control_frequency

        # control_frequency = 50hz
        # dt = 0.001
        # steps_per_pd_update = (1/0.001) / 50 = 20
        self._steps_per_pd_update = (1.0 / self._dt) / self._control_frequency
        logger.info(
            "Simulation timestep: %.4f s, Control frequency: %.2f Hz, Steps per control update: %d",
        )

        # might not need this
        self._model_metadata = model_metadata

        # Reset the simulation
        if self._freejoint:
            self._data.qpos[:3] = np.array([0.0, 0.0, self._start_height])
            self._data.qpos[3:7] = np.array(self._initial_quat)
            self._data.qpos[7:] = np.zeros_like(self._data.qpos[7:])
        else:
            self._data.qpos = np.zeros_like(self._data.qpos)

        self._data.qvel = np.zeros_like(self._data.qvel)
        self._data.qacc = np.zeros_like(self._data.qacc)

        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_step(self._model, self._data)

        # get our viewer
        self._viewer = QtViewer(
            self._model,
            width=512,
            height=512,
            shadow=False,
            reflection=False,
            contact_force=False,
            contact_point=False,
            inertia=False,
            enable_plots=True,
            camera_distance=3.5,
            camera_azimuth=90.0,
            camera_elevation=-10.0,
            camera_lookat=(0.0, 0.0, 0.5),
            window_title="Mujoco Viewer"
        )

        self._sensor_name_to_id = {self._model.sensor(i).name: i for i in range(self._model.nsensor)}

        # initialize actuator logic
        self._joint_name_to_id = {self._model.actuator(i).name: i for i in range(self._model.nu)}
        self._joint_id_to_name = {v: k for k, v in self._joint_name_to_id.items()}

        self._joint_name_to_actuator_name = {
            joint_name: f"{joint_name}_ctrl" for joint_name in self._joint_name_to_id.keys()
        }

        # this is a mapping from joint_id to actuator callable
        self._actuators: dict[int, Actuator] = {}

        for joint_name, joint_metadata in model_metadata.joint_name_to_actuator_metadata.items():
            joint_id = self._joint_name_to_id[joint_name]
            actuator_type = joint_metadata.actuator_type

            actuator: Actuator

            match actuator_type:
                case ActuatorType.POSITION:
                    actuator = PositionActuator(joint_metadata.kp, joint_metadata.kd, joint_metadata.max_torque)
                case ActuatorType.VELOCITY:
                    actuator = VelocityActuator(joint_metadata.kp, joint_metadata.kd, joint_metadata.max_torque)
                case _:
                    logger.warning("Actuator type %s not implemented yet.", actuator_type)
                    raise NotImplementedError

            self._actuators[joint_id] = actuator

        self._current_commands: dict[str, ActuatorCommand] = {}
        self._next_commands: dict[str, tuple[ActuatorCommand, float]] = {}

        self._target_time = time.time()
        self._num_steps = 0

    async def step(self) -> None:
        """Step the simulation forward by one timestep."""
        self._target_time += self._dt
        self._num_steps += 1

        to_remove = []
        for name, (target_command, application_time) in self._next_commands.items():
            if application_time <= self._target_time:
                self._current_commands[name] = target_command
                to_remove.append(name)

        for name in to_remove:
            del self._next_commands[name]

        # We need to compute torque from actuator commands
        prev_torque = self._data.ctrl[:]
        for joint_name, command in self._current_commands.items():
            joint_id = self._joint_name_to_id[joint_name]
            actuator = self._actuators[joint_id]

            torque = prev_torque[joint_id]

            if self._num_steps % self._steps_per_pd_update == 0:
                joint_qpos_lookup = joint_id
                joint_qvel_lookup = joint_id
                if self._freejoint:
                    joint_qpos_lookup += 7  # skip the freejoint's 7 dof (xyz, wxyz)
                    joint_qvel_lookup += 6  # skip the freejoint's 6 dof (xyz, angular xyz)
                qpos = self._data.qpos[joint_qpos_lookup]
                qvel = self._data.qvel[joint_qvel_lookup]

                # compute torque
                torque = actuator.get_torque(
                    cmd=command,
                    qpos=qpos,
                    qvel=qvel,
                )

            # lookup corresponding actuator id from joint_name
            actuator_name = self._joint_name_to_actuator_name[joint_name]
            actuator_id = self._model.actuator(actuator_name).id

            # set torque
            self._data.ctrl[actuator_id] = torque

        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_step(self._model, self._data)

        self._viewer.push_state(
            self._data.qpos,
            self._data.qvel,
            sim_time=float(self._data.time),
        )

        xfrc = self._viewer.drain_control_pipe()
        if xfrc is not None:
            self._data.xfrc_applied[:] = xfrc

    async def get_actuator_state(self, joint_id: int) -> ActuatorState:
        if joint_id not in self._joint_id_to_name:
            raise KeyError(f"Joint ID {joint_id} not found in the model.")

        joint_name = self._joint_id_to_name[joint_id]
        joint_data = self._data.joint(joint_name)
        return ActuatorState(
            position=joint_data.qpos,
            velocity=joint_data.qvel,
            torque=None,
        )

    async def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        if sensor_name not in self._sensor_name_to_id:
            raise KeyError(f"Sensor name {sensor_name} not found in the model.")
        sensor_id = self._sensor_name_to_id[sensor_name]
        return self._data.sensor(sensor_id).data.copy()

    async def set_actuator_command(self, joint_name: str, actuator_command: ActuatorCommand) -> None:
        """Set the target setpoint for a specific actuator.

        The setpoint type depends on the type of actuator configured for the joint.
        E.g. if joint_id 1 corresponds to a position_actuator, target_setpoint is of unit radians.
        """
        self._next_commands[joint_name] = (actuator_command, self._target_time)

    async def command_actuators(self, commands: dict[str, ActuatorCommand]) -> None:
        """Set the target setpoints for multiple actuators at once.

        The setpoint types depend on the types of actuators configured for the joints.
        E.g. if joint_id 1 corresponds to a position_actuator, target_setpoint is of unit radians.
        """
        commands = [self.set_actuator_command(joint_name, command) for joint_name, command in commands.items()]
        await asyncio.gather(*commands)

    async def configure_actuator(self, joint_id: int, config: ConfigureActuatorRequest) -> None:
        if joint_id not in self._actuators:
            raise KeyError(f"Joint ID {joint_id} not found in the model.")
        actuator = self._actuators[joint_id]

        cfg: dict[str, float] = {k: float(v) for k, v in config.items() if isinstance(v, (int, float))}

        actuator.configure(**cfg)

    async def reset(self) -> None:
        self._next_commands.clear()
        self._current_commands.clear()

        mujoco.mj_resetData(self._model, self._data)

        self._data.ctrl[:] = 0.0
        self._data.qfrc_applied[:] = 0.0
        self._data.qfrc_bias[:] = 0.0
        self._data.actuator_force[:] = 0.0

        qpos = np.zeros_like(self._data.qpos)

        if self._freejoint:
            qpos[3:] = np.array([0.0, 0.0, self._start_height])
            qpos[3:7] = np.array(self._initial_quat)
            qpos[7:] = np.zeros_like(qpos[7:])
        else:
            qpos[:] = np.zeros_like(qpos)

        self._data.qpos[:] = qpos
        self._data.qvel[:] = np.zeros_like(self._data.qvel)
        self._data.qacc[:] = np.zeros_like(self._data.qacc)

        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_step(self._model, self._data)

    async def close(self) -> None:
        try:
            self._viewer.close()
        except Exception as e:
            logger.error("Error closing viewer: %s", e)

    @property
    def dt(self) -> float:
        return self._dt
