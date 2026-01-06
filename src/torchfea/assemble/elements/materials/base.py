import torch


class Materials_Base():

    def __init__(self) -> None:
        self.type: int
        """
        0: LinearElastic
        1: NeoHookean
        """

    def material_Constitutive_C3(self, F, J, Jneg, invF, I1):
        pass

    def strain_energy_density_C3(self, F):
        pass

    def __I1dF(self, F, I1, J, invF):
        return torch.einsum('geij,ge->geij', F, J**(-2 / 3)) - \
               torch.einsum('ge,geji->geij', I1, invF) * 2 / 3

    def __I1dF2(self, F, I1, J, invF):
        pass

    def __JdF(self, invF, J):
        return torch.einsum('ge,geji->geij', J, invF)
