from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


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
