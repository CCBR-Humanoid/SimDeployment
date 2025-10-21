"""Serve the simulation."""

import asyncio
import json
from pathlib import Path

from simdeployment.simulator import MujocoSimulator
from simdeployment.types import ActuatorMetadata, ActuatorType, ModelMetadata, MuJoCoMetadata


def load_joint_metadata(metadata_path: Path) -> ModelMetadata:
    """Load joint metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    # Convert joint metadata from dict to ActuatorMetadata objects
    joint_name_to_actuator_metadata = {}
    for joint_name, metadata in data["joint_name_to_actuator_metadata"].items():
        joint_name_to_actuator_metadata[joint_name] = ActuatorMetadata(
            joint_name=metadata["joint_name"],
            actuator_type=ActuatorType(metadata["actuator_type"]),
            kp=metadata["kp"],
            kd=metadata["kd"],
            max_torque=metadata["max_torque"]
        )

    return ModelMetadata(
        joint_name_to_actuator_metadata=joint_name_to_actuator_metadata,
        start_height=data["start_height"],
        initial_quat=tuple(data["initial_quat"]),
        freejoint=data["freejoint"]
    )


async def serve(simulator: MujocoSimulator) -> None:
    while True:
        await asyncio.sleep(simulator.dt)
        await simulator.step()


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "robots" / "dummy" / "dummy.mjcf"
    metadata_path = model_path.parent / "joint_metadata.json"

    # Load joint metadata from JSON file
    model_metadata = load_joint_metadata(metadata_path)

    asyncio.run(serve(MujocoSimulator(
        model_path=model_path,
        model_metadata=model_metadata,
        simulation_metadata=MuJoCoMetadata(
            dt=0.001,
            control_frequency=50.0,
            gravity=(0.0, 0.0, -9.81),
            integrator="implicitfast",
            solver="cg",
        ),
    )))
