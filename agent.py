from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from deeplense.lens import DeepLens
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel




class ModelIParams(BaseModel):
    model: Literal["Model_I"] = "Model_I"
    substructure: Literal["no_sub", "cdm", "axion"]
    num_images: int = Field(default=10, ge=1, le=1000)
    halo_mass: float = Field(default=1e12, gt=0)
    axion_mass: Optional[float] = Field(default=None)

    @model_validator(mode='after')
    def check_axion_mass(self):
        if self.substructure == "axion" and self.axion_mass is None:
            raise ValueError("axion_mass is required when substructure is axion")
        return self

class ModelIIParams(BaseModel):
    model: Literal["Model_II"] = "Model_II"
    substructure: Literal["no_sub", "cdm", "axion"]
    num_images: int = Field(default=10, ge=1, le=1000)
    halo_mass: float = Field(default=1e12, gt=0)
    axion_mass: Optional[float] = Field(default=None)
    instrument: Literal["Euclid"] = "Euclid"

    @model_validator(mode='after')
    def check_axion_mass(self):
        if self.substructure == "axion" and self.axion_mass is None:
            raise ValueError("axion_mass is required when substructure is axion")
        return self

class ModelIIIParams(BaseModel):
    model: Literal["Model_III"] = "Model_III"
    substructure: Literal["no_sub", "cdm", "axion"]
    num_images: int = Field(default=10, ge=1, le=1000)
    halo_mass: float = Field(default=1e12, gt=0)
    axion_mass: Optional[float] = Field(default=None)
    instrument: Literal["hst"] = "hst"

    @model_validator(mode='after')
    def check_axion_mass(self):
        if self.substructure == "axion" and self.axion_mass is None:
            raise ValueError("axion_mass is required when substructure is axion")
        return self

class SimulationResult(BaseModel):
    model: str
    substructure: str
    num_images: int
    image_shape: tuple
    output_path: str
    metadata: dict

import numpy as np
import os
import random
from deeplense.lens import DeepLens

def run_simulation(params: ModelIParams | ModelIIParams | ModelIIIParams) -> SimulationResult:
    """Tool function the agent calls to run a simulation"""
    
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
    
#definition of the agent
simulation_agent = Agent(
    model=AnthropicModel("claude-opus-4-5"),
    system_prompt="""
    You are a gravitational lensing simulation assistant.
    You help researchers generate strong lensing images using DeepLenseSim.
    
    When a user asks for simulations:
    1. Identify which model they want (Model_I, Model_II, Model_III)
    2. Identify substructure type (no_sub, cdm, axion)
    3. Identify number of images
    4. If substructure is axion, ask for axion_mass
    5. If ANYTHING is missing, ask the user before running
    6. Always confirm parameters with user before running simulation
    
    Available models:
    - Model_I: generic telescope, 150x150
    - Model_II: Euclid space telescope, 150x150
    - Model_III: Hubble Space Telescope, 150x150
    
    Available substructures:
    - no_sub: no dark matter substructure
    - cdm: cold dark matter subhalos
    - axion: axion substructure (requires axion_mass between 1e-24 and 1e-22)
    """,
)
@simulation_agent.tool_plain
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
    
def visualize_images(output_path: str, num_images: int = None):
    """Load and visualize generated lensing images"""
    
    images = np.load(output_path, allow_pickle=True)
    
    # how many to show
    n = min(len(images), num_images or len(images))
    
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    # handle single image case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        axes[i].imshow(images[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    # get model and substructure from filename
    filename = os.path.basename(output_path)
    parts = filename.split('_')
    title = f"{parts[0]}_{parts[1]} - {parts[2]} substructure"
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # save the plot
    plot_path = output_path.replace('.npy', '_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {plot_path}")
    
    return plot_path
def chat():
    print("DeepLenseSim Agent ready!")
    print("Type 'quit' to exit\n")
    
    message_history = []
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue
            
        result = simulation_agent.run_sync(
            user_input,
            message_history=message_history
        )
        
        response = result.output
        print(f"\nAgent: {response}\n")
        
        # use pydantic-ai's own message history
        message_history = result.all_messages()
if __name__ == "__main__":
    chat()