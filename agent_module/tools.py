from agent_module.models import ModelIParams, ModelIIParams, ModelIIIParams
from agent_module.simulator import run_simulation
from agent_module.visualizer import visualize_images
import matplotlib
matplotlib.use("Agg")


def simulate(
    model: str,
    substructure: str,
    num_images: int,
    axion_mass: float = None,
    halo_mass: float = 1e12,
) -> str:
    """Run a DeepLenseSim simulation with the given parameters."""
    try:
        if model == "Model_I":
            params = ModelIParams(
                substructure=substructure,
                num_images=num_images,
                axion_mass=axion_mass,
                halo_mass=halo_mass,
            )
        elif model == "Model_II":
            params = ModelIIParams(
                substructure=substructure,
                num_images=num_images,
                axion_mass=axion_mass,
                halo_mass=halo_mass,
            )
        elif model == "Model_III":
            params = ModelIIIParams(
                substructure=substructure,
                num_images=num_images,
                axion_mass=axion_mass,
                halo_mass=halo_mass,
            )
        else:
            return f"Unknown model: {model}"

        result = run_simulation(params)
        return (
            f"Simulation complete!\n"
            f"Model: {result.model}\n"
            f"Substructure: {result.substructure}\n"
            f"Images generated: {result.num_images}\n"
            f"Image shape: {result.image_shape}\n"
            f"Saved to: {result.output_path}\n"
            f"Metadata: {result.metadata}"
        )
    except Exception as e:
        return f"Simulation failed: {str(e)}"


def list_models() -> str:
    """List all available simulation models and their descriptions."""
    return """
Available DeepLenseSim Models:

Model_I - Generic Telescope
  - Single channel, 150x150 pixels
  - Simple Gaussian PSF
  - Background noise for SNR ~25
  - Best for: quick simulations, baseline comparisons

Model_II - Euclid Space Telescope
  - Single channel, 150x150 pixels
  - Realistic Euclid instrument settings
  - More realistic noise and PSF
  - Best for: simulating ESA Euclid observations

Model_III - Hubble Space Telescope (HST)
  - Single channel, 150x150 pixels
  - Realistic HST instrument settings
  - High resolution PSF
  - Best for: simulating HST observations

Available Substructures:
  - no_sub: No dark matter substructure
  - cdm: Cold dark matter (point mass subhalos)
  - axion: Axion/vortex substructure (requires axion_mass between 1e-24 and 1e-22)
"""


def compare_models(
    substructure: str,
    axion_mass: float = None,
    num_images: int = 1,
) -> str:
    """Generate the same substructure across all 3 models for comparison."""
    results = []
    output_paths = []

    for model in ["Model_I", "Model_II", "Model_III"]:
        try:
            if model == "Model_I":
                params = ModelIParams(
                    substructure=substructure,
                    num_images=num_images,
                    axion_mass=axion_mass,
                )
            elif model == "Model_II":
                params = ModelIIParams(
                    substructure=substructure,
                    num_images=num_images,
                    axion_mass=axion_mass,
                )
            elif model == "Model_III":
                params = ModelIIIParams(
                    substructure=substructure,
                    num_images=num_images,
                    axion_mass=axion_mass,
                )

            result = run_simulation(params)
            results.append(f"{model}: saved to {result.output_path}")
            output_paths.append(result.output_path)

        except Exception as e:
            results.append(f"{model}: failed - {str(e)}")

    # visualize all 3 side by side
    if output_paths:
        import numpy as np
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(output_paths), figsize=(5*len(output_paths), 5))
        if len(output_paths) == 1:
            axes = [axes]

        model_names = ["Model_I\n(Generic)", "Model_II\n(Euclid)", "Model_III\n(HST)"]
        for i, path in enumerate(output_paths):
            images = np.load(path, allow_pickle=True)
            axes[i].imshow(images[0], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(model_names[i], fontsize=12)

        plt.suptitle(f"Model Comparison - {substructure} substructure", fontsize=14)
        plt.tight_layout()
        compare_path = f"outputs/comparison_{substructure}.png"
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.show()
        results.append(f"Comparison plot saved to: {compare_path}")

    return "\n".join(results)