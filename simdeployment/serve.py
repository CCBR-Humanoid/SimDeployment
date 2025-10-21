import asyncio
from pathlib import Path

from simdeployment.simulator import MujocoSimulator
from simdeployment.types import ModelMetadata, MuJoCoMetadata


async def serve(simulator: MujocoSimulator) -> None:
    while True:
        await asyncio.sleep(simulator.dt)
        await simulator.step()


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "robots" / "dummy" / "dummy.mjcf"
    asyncio.run(serve(MujocoSimulator(
        model_path=model_path,
        model_metadata=ModelMetadata(
            joint_name_to_actuator_metadata={},
            start_height=1.0,
            initial_quat=(1.0, 0.0, 0.0, 0.0),
            freejoint=True,
        ),
        simulation_metadata=MuJoCoMetadata(
            dt=0.001,
            control_frequency=50.0,
            gravity=(0.0, 0.0, -9.81),
            integrator="implicitfast",
            solver="cg",
        ),
    )))
