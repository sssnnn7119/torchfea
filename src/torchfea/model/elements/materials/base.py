from ....interfaces import Serializable


class Materials_Base(Serializable):
    def __init__(self) -> None:
        super().__init__()

    def material_Constitutive_C3(self, F, J, Jneg, invF, I1):
        pass

    def strain_energy_density_C3(self, F):
        pass
