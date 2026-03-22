import numpy as np
import os
import random
from deeplense.lens import DeepLens
from agent_module.models import ModelIParams, ModelIIParams, ModelIIIParams, SimulationResult


def run_simulation(params: ModelIParams | ModelIIParams | ModelIIIParams) -> SimulationResult:
    """Runs DeepLens simulation based on validated parameters"""

    os.makedirs("outputs", exist_ok=True)
    images = []

    for i in range(params.num_images):
        # step 1: create lens object
        if params.substructure == "axion":
            lens = DeepLens(axion_mass=params.axion_mass)
        else:
            lens = DeepLens()

        # step 2: make halo
        lens.make_single_halo(params.halo_mass)

        # step 3: add substructure
        if params.substructure == "no_sub":
            lens.make_no_sub()
        elif params.substructure == "cdm":
            lens.make_old_cdm()
        elif params.substructure == "axion":
            lens.make_vortex(3e10)

        # step 4: set instrument if Model II or III
        if params.model == "Model_II":
            lens.set_instrument('Euclid')
        elif params.model == "Model_III":
            lens.set_instrument('hst')

        # step 5: source light and simulate
        if params.model == "Model_I":
            lens.make_source_light()
            lens.simple_sim()
        else:
            lens.make_source_light_mag()
            lens.simple_sim_2()

        images.append(lens.image_real)

    # save images
    output_path = f"outputs/{params.model}_{params.substructure}_{random.getrandbits(64)}.npy"
    np.save(output_path, np.array(images))

    return SimulationResult(
        model=params.model,
        substructure=params.substructure,
        num_images=params.num_images,
        image_shape=tuple(images[0].shape),
        output_path=output_path,
        metadata={"halo_mass": params.halo_mass, "axion_mass": params.axion_mass}
    )