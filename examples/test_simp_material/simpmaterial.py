from typing import Optional


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

import torchfea

class SIMPElement(torchfea.elements.Element_3D):

    def __init__(self, elems_index, elems, penalfactor: torch.Tensor):
        super().__init__(elems_index, elems)

        self.penalfactor = penalfactor
        """the penalization factor for SIMP material"""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._dN2W = torch.einsum('geija,ge->geija', self.shape_function_d2_gaussian, self.gaussian_weight)

        EmdUgrad2_2 = torch.zeros([1, 1, 3, 3, 3, 3, 3, 3])
        for I0 in range(3):
            for i0 in range(3):
                for j0 in range(3):
                    EmdUgrad2_2[..., I0, i0, j0, I0, i0, j0] = self.penalfactor * 2
        
        self._EmdUe_2 = torch.einsum('geija, geklb,geIijJkl->aIbJe', self._dN2W, self.shape_function_d2_gaussian, EmdUgrad2_2)

    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)
        
        Ea = super().potential_Energy(RGC, rotation_matrix)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        Er = self.penalfactor * torch.einsum('geIij,geIij,ge->', Ugrad2, Ugrad2, self.gaussian_weight)


        return Ea + Er
    
    def _get_EpdUe_EpdUe2(self, U, if_onlyforce = False):
        result0 = super()._get_EpdUe_EpdUe2(U, if_onlyforce)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        EmdUgrad2 = 2 * self.penalfactor * Ugrad2

        EmdUe = torch.einsum('geIij,geija->aIe', EmdUgrad2,
                                self._dN2W)
        
        if if_onlyforce:
            return EmdUe + result0

        return EmdUe + result0[0], self._EmdUe_2 + result0[1]

class SIMPElementType2(torchfea.elements.Element_3D):

    def __init__(self, elems_index, elems, penalfactor: torch.Tensor):
        super().__init__(elems_index, elems)

        self.penalfactor = penalfactor
        """the penalization factor for SIMP material"""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._dN2W = torch.einsum('geija,ge->geija', self.shape_function_d2_gaussian, self.gaussian_weight)

        EmdUgrad2_2 = torch.zeros([1, 1, 3, 3, 3, 3, 3, 3])
        for I0 in range(3):
            for i0 in range(3):
                for j0 in range(3):
                    EmdUgrad2_2[..., I0, i0, j0, I0, i0, j0] += self.penalfactor * 4
                    EmdUgrad2_2[..., I0, i0, j0, i0, I0, j0] += -self.penalfactor * 4

        self._EmdUe_2 = torch.einsum('geija, geklb,geIijJkl->aIbJe', self._dN2W, self.shape_function_d2_gaussian, EmdUgrad2_2)

    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)
        
        Ea = super().potential_Energy(RGC, rotation_matrix)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
        
        Fskew = Ugrad2 - Ugrad2.transpose(2, 3)

        Er = self.penalfactor * torch.einsum('geIij,geIij,ge->', Fskew, Fskew, self.gaussian_weight)


        return Ea + Er
    
    def _get_EpdUe_EpdUe2(self, U, if_onlyforce = False):
        result0 = super()._get_EpdUe_EpdUe2(U, if_onlyforce)

        Ue = U[self._elems]

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         Ue[:, i])
        
        # Fskew_geijk = Ugrad2_geijk - Ugrad2_geikj
        Fskew = Ugrad2 - Ugrad2.transpose(2, 3) 

        # E = p Fskew_geijk Fskew_geijk w_ge
        Er = self.penalfactor * torch.einsum('geIij,geIij,ge->', Fskew, Fskew, self.gaussian_weight)


        EmdUgrad2 = 4 * self.penalfactor * (Ugrad2 - Ugrad2.transpose(2, 3))

        EmdUe = torch.einsum('geIij,geija->aIe', EmdUgrad2,
                                self._dN2W)
        
        if if_onlyforce:
            return EmdUe + result0

        return EmdUe + result0[0], self._EmdUe_2 + result0[1]

class SIMPElementFgrad(torchfea.elements.Element_3D):

    def __init__(self, elems_index, elems, penalfactor: torch.Tensor):
        super().__init__(elems_index, elems)

        self.penalfactor = penalfactor
        """the penalization factor for SIMP material"""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        if self.penalfactor.dim() == 0 or self.penalfactor.shape == (1,):
            self.penalfactor = self.penalfactor.reshape(1, 1)

        self._dN2WP = torch.einsum('geija,ge->geija', self.shape_function_d2_gaussian, self.gaussian_weight * self.penalfactor)


        self._EmdUe_2 = torch.zeros([self.num_nodes_per_elem, 3, self.num_nodes_per_elem, 3, self._elems.shape[0]])

        for I0 in range(3):
            for i0 in range(3):
                for j0 in range(3):

                    I = I0
                    i = i0
                    j = j0
                    J = I0
                    k = i0
                    l = j0

                    self._EmdUe_2[:, I, :, J, :] += torch.einsum('gea, geb->abe', self._dN2WP[:, :, i, j, :], self.shape_function_d2_gaussian[:, :, k, l, :]) * 2


    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)
        
        Ea = super().potential_Energy(RGC, rotation_matrix)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        Er = torch.einsum('geIij,geIij,ge->', Ugrad2, Ugrad2, self.gaussian_weight * self.penalfactor)


        return Ea + Er
    
    def _get_EpdUe_EpdUe2(self, U, if_onlyforce = False):
        result0 = super()._get_EpdUe_EpdUe2(U, if_onlyforce)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        EmdUgrad2 = 2 * Ugrad2

        EmdUe = torch.einsum('geIij,geija->aIe', EmdUgrad2,
                                self._dN2WP)
        
        if if_onlyforce:
            return EmdUe + result0

        return EmdUe + result0[0], self._EmdUe_2 + result0[1]

class SIMPElementHuHu_LuLu(torchfea.elements.Element_3D):

    def __init__(self, elems_index, elems, penalfactor: torch.Tensor):
        super().__init__(elems_index, elems)

        self.penalfactor = penalfactor
        """the penalization factor for SIMP material"""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        if self.penalfactor.dim() == 0 or self.penalfactor.shape == (1,):
            self.penalfactor = self.penalfactor.reshape(1, 1)

        self._dN2WP = torch.einsum('geija,ge->geija', self.shape_function_d2_gaussian, self.gaussian_weight * self.penalfactor)


        self._EmdUe_2 = torch.zeros([self.num_nodes_per_elem, 3, self.num_nodes_per_elem, 3, self._elems.shape[0]])

        for I0 in range(3):
            for i0 in range(3):

                I = I0
                i = i0
                j = i0
                J = I0
                k = i0
                l = i0

                self._EmdUe_2[:, I, :, J, :] -= torch.einsum('gea, geb->abe', self._dN2WP[:, :, i, j, :], self.shape_function_d2_gaussian[:, :, k, l, :]) / 3

                for j0 in range(3):

                    I = I0
                    i = i0
                    j = j0
                    J = I0
                    k = i0
                    l = j0

                    self._EmdUe_2[:, I, :, J, :] += torch.einsum('gea, geb->abe', self._dN2WP[:, :, i, j, :], self.shape_function_d2_gaussian[:, :, k, l, :])

        # self._EmdUe_2 = torch.einsum('geija, geklb,geIijJkl->aIbJe', self._dN2W, self.shape_function_d2_gaussian, EmdUgrad2_2)


    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)
        
        Ea = super().potential_Energy(RGC, rotation_matrix)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        Er = 0.5 * (Ugrad2**2).sum([2, 3, 4])
        for i in range(3):
            Er -= 0.5 * (Ugrad2[:, :, :, i, i]**2).sum([-1]) / 3

        Er = (Er * self.gaussian_weight * self.penalfactor).sum()

        return Ea + Er
    
    def _get_EpdUe_EpdUe2(self, U, if_onlyforce = False):
        result0 = super()._get_EpdUe_EpdUe2(U, if_onlyforce)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
        
        Er = self.potential_Energy(U)

        EmdUgrad2 = Ugrad2
        for i in range(3):
            EmdUgrad2[:, :, :, i, i] -= Ugrad2[:, :, :, i, i] / 3

        EmdUe = torch.einsum('geIij,geija->aIe', EmdUgrad2,
                                self._dN2WP)
        
        if if_onlyforce:
            return EmdUe + result0

        return EmdUe + result0[0], self._EmdUe_2 + result0[1]


class SIMPElementFsrew(torchfea.elements.Element_3D):

    _serialized_attributes: list[str] = ['_elems_index', '_elems', '_density', 'materials', 'penalfactor']

    def __init__(self, elems_index, elems, penalfactor: torch.Tensor):
        super().__init__(elems_index, elems)

        self.penalfactor = penalfactor
        """the penalization factor for SIMP material"""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        if self.penalfactor.dim() == 0 or self.penalfactor.shape == (1,):
            self.penalfactor = self.penalfactor.reshape(1, 1)

        self._dN2WP = torch.einsum('geija,ge->geija', self.shape_function_d2_gaussian, self.gaussian_weight * self.penalfactor)

        self._EmdUe_2 = torch.zeros([self.num_nodes_per_elem, 3, self.num_nodes_per_elem, 3, self._elems.shape[0]])

        for I0 in range(3):
            for i0 in range(3):
                for j0 in range(3):

                    I = I0
                    i = i0
                    j = j0
                    J = I0
                    k = i0
                    l = j0

                    self._EmdUe_2[:, I, :, J, :] += torch.einsum('gea, geb->abe', self._dN2WP[:, :, i, j, :], self.shape_function_d2_gaussian[:, :, k, l, :]) * 4

                    I = I0
                    i = i0
                    j = j0
                    J = i0
                    k = I0
                    l = j0

                    self._EmdUe_2[:, I, :, J, :] += torch.einsum('gea, geb->abe', self._dN2WP[:, :, i, j, :], self.shape_function_d2_gaussian[:, :, k, l, :]) * -4

    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)
        
        Ea = super().potential_Energy(RGC, rotation_matrix)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                        self.shape_function_d2_gaussian[..., i],
                                        U[self._elems[:, i]])
        
        Fskew = Ugrad2 - Ugrad2.transpose(2, 3)

        Er = torch.einsum('geIij,ge->', Fskew**2, self.gaussian_weight * self.penalfactor)


        return Ea + Er
    
    def _get_EpdUe_EpdUe2(self, U, if_onlyforce = False):
        result0 = super()._get_EpdUe_EpdUe2(U, if_onlyforce)

        Ue = U[self._elems]

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                        self.shape_function_d2_gaussian[..., i],
                                        Ue[:, i])
        
        # Fskew_geijk = Ugrad2_geijk - Ugrad2_geikj
        # Fskew = Ugrad2 - Ugrad2.transpose(2, 3) 

        # E = p Fskew_geijk Fskew_geijk w_ge
        # Er = torch.einsum('geIij,geIij,ge->', Fskew, Fskew, self.gaussian_weight * self.penalfactor)


        EmdUgrad2 = 4 * (Ugrad2 - Ugrad2.transpose(2, 3))

        EmdUe = torch.einsum('geIij,geija->aIe', EmdUgrad2,
                                self._dN2WP)
        
        if if_onlyforce:
            return EmdUe + result0

        return EmdUe + result0[0], self._EmdUe_2 + result0[1]


class SIMPElementC3D10(torchfea.elements.C3D10, SIMPElementHuHu_LuLu):
    pass

def get_fe_C3D4Less():
    fem = torchfea.FEA_INP()
    # fem.Read_INP(
    #     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
    # )

    # fem.Read_INP(
    #     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
    # )
    import pathlib
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))

    path0 = pathlib.Path(current_path).parent / 'models' / 'C3D4Less.inp'

    fem.read_inp(path0)

    fe = torchfea.from_inp(fem)
    fe.solver = torchfea.solver.StaticImplicitSolver()
    # elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
    # fe.elems['element-0'].set_materials(elems)

    # torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

    fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                    name='pressure_1')

    rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

    fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))
    
    fe.assembly.get_part('final_model').convert_linear_to_quadratic_elements(['C3D4'], ['C3D4'])

    return fe


def test_autograd():

    fe = torchfea.load_model('tests/test_simp_material/simpmodel.npz')

    part = fe.assembly.get_part('final_model')

    elem_c3d4 = part.elems['C3D4']
    elem_c3d4_simp = SIMPElementC3D10(elem_c3d4._elems_index[:1], elem_c3d4._elems[:1], penalfactor=torch.tensor(1e1))
    elem_c3d4_simp.materials.clear()

    elem_c3d4_simp.initialize(nodes=part.nodes)
    RGC = torch.randn([part.nodes.shape[0], 3], requires_grad=True)

    elem_c3d4_simp._get_EpdUe_EpdUe2(RGC)

def test_solve():
    fe = torchfea.load_model('tests/test_simp_material/simpmodel.npz')

    fe.assembly.get_load('pressure_1').pressure = 0.06

    fe.initialize()

    part = fe.assembly.get_part('final_model')

    elem_c3d4 = part.elems['C3D4']
    elem_c3d4_simp = SIMPElementC3D10(elem_c3d4._elems_index, elem_c3d4._elems, penalfactor=torch.tensor(1e-3))
    elem_c3d4_simp.materials = elem_c3d4.materials

    part.elems['C3D4'] = elem_c3d4_simp

    resultsimp = fe.solve()


    result = torchfea.solver.StaticResult.load('tests/test_simp_material/result.npz')

    print('without penalty GC:', result.GC[-6:])
    print('with penalty GC:', resultsimp.GC[-6:])

    fe.assembly.show_all(resultsimp.GC)

if __name__ == '__main__':
    


    test_autograd()

    test_solve()

    raise False