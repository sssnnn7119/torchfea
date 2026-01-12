from math import e
import numpy as np
import torch
from .base import BaseConstraint

class Couple(BaseConstraint):

    def __init__(self, instance_name: str, set_nodes_name: str, rp_name: str) -> None:
        super().__init__()
        self.instance_name = instance_name
        self.set_nodes_name = set_nodes_name
        self.rp_name = rp_name

        
        self._ref_location: torch.Tensor

        self._couple_index: int
        self._rp_index: int
        self._instance_RGC_index: int
        self._indexNodes: np.ndarray

    def initialize(self, assembly):
        super().initialize(assembly)
        self._indexNodes = self._assembly.get_instance(self.instance_name).set_nodes[self.set_nodes_name]
        self._rp_index = self._assembly.get_reference_point(self.rp_name)._RGC_index
        self._couple_index = self._assembly.get_instance(self.instance_name)._RGC_index

        instance = self._assembly.get_instance(self.instance_name)
        self._instance_RGC_index = instance._RGC_index

        index_global = instance.nodes[self._indexNodes]
        self._ref_location = index_global - self._assembly.get_reference_point(self.rp_name).node

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply the couple constraint to the displacement vector
        """
        RGC[self._couple_index][self._indexNodes] = RGC[self._rp_index][:3] + self._rotation3d(
            RGC[self._rp_index][3:], self._ref_location) - self._ref_location

        return RGC
    
    def modify_mass_matrix(self, mass_indices, mass_values, RGC: list[torch.Tensor]):
        v = RGC[self._rp_index][:3]
        z = RGC[self._rp_index][3:]

        theta = z.norm() + 1e-20
        w = (z / theta)

        epsilon = torch.zeros([3, 3, 3])
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[1, 0, 2] = epsilon[2, 1, 0] = -1

        y = v - self._ref_location + \
            self._ref_location * torch.cos(theta) + \
            torch.einsum('ijk, j, pk->pi', epsilon, w, self._ref_location) * torch.sin(theta) + \
            torch.einsum('i,j,pj->pi', w, w, self._ref_location) * (1 - torch.cos(theta))

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._couple_index][self._indexNodes] = False
        RGC_remain_index[self._rp_index][:] = True
        return RGC_remain_index

    def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
                   K_indices: torch.Tensor = None, K_values: torch.Tensor = None, if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify the R and K

        Args:
            indexStart (list[int]): The starting indices for each node.
            U (list[torch.Tensor]): The displacement vector for each node.
            R (torch.Tensor): The global force vector.
            K (torch.Tensor): The global stiffness matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The modified R and K tensors.
        """

        if not if_onlyforce and (K_indices is None or K_values is None):
            raise ValueError("K_indices and K_values must be provided when if_onlyforce is False")

        R_now = R0[self._assembly.RGC_list_indexStart[self._instance_RGC_index]:self._assembly.RGC_list_indexStart[self._instance_RGC_index+1]].view(-1, 3)
        Ydot, Ydot2 = self._calculate_Ydotz(RGC)

        # R
        # region
        Rrest = R_now[self._indexNodes]

        Edotv = Rrest.sum(dim=0)
        Edotz = torch.einsum('bj,bjp->p', Rrest, Ydot)

        R = torch.zeros(self._assembly.RGC_list_indexStart[-1])
        start_idx = self._assembly.RGC_list_indexStart[self._rp_index]
        R[start_idx:start_idx+3] += Edotv
        R[start_idx+3:start_idx+6] += Edotz
        # endregion

        if if_onlyforce:
            return R

        # K
        # region
        ## first, get the K of the rest part in index1

        # initial select the instance indices

        indice_max = self._assembly.RGC_list_indexStart[-1]
        indice_start = self._assembly.RGC_list_indexStart[self._couple_index]

        index = torch.where(
            torch.isin(((K_indices[1] - indice_start) // 3),
                    torch.tensor(self._indexNodes.tolist())))

        sort_index = torch.argsort((K_indices[1][index] - indice_start) // 3)

        index = index[0][sort_index]
        indice1 = K_indices[0][index]
        indice30 = K_indices[1][index] // 3
        indice3 = torch.unique_consecutive(indice30, return_inverse=True)[1]
        indice4 = K_indices[1][index] % 3

        Rdotv_indices = torch.stack([indice1, indice4], dim=0)
        Rdotv_indices_flatten = Rdotv_indices[0] * 3 + Rdotv_indices[1]
        Rdotv_values = K_values[index]
        Rdotv = torch.zeros([indice_max * 3]).scatter_add_(
            0, Rdotv_indices_flatten,
            Rdotv_values).reshape(indice_max, 3)

        Rdotz_indices = torch.stack([
            indice1.reshape(-1, 1).repeat(1, 3),
            torch.tensor([0, 1, 2]).reshape([1, 3]).repeat(
                indice1.shape[0], 1)
        ],
                                    dim=0).reshape([2, -1])
        Rdotz_indices_flatten = Rdotz_indices[0] * 3 + Rdotz_indices[1]
        Rdotz_values = (K_values[index].unsqueeze(-1) *
                        Ydot.view(-1, 3)[indice4 + indice3 * 3]).flatten()
        Rdotz = torch.zeros([indice_max * 3]).scatter_add_(
            0, Rdotz_indices_flatten,
            Rdotz_values).reshape(indice_max, 3)
        
        index_remain_dim0 = np.vstack([indice_start + self._indexNodes * 3,
                                    indice_start + self._indexNodes * 3 + 1,
                                    indice_start + self._indexNodes * 3 + 2]).T

        Edotvv = Rdotv[index_remain_dim0].sum(dim=0)
        Edotzv = torch.einsum('biq,bip->pq', Rdotv[index_remain_dim0], Ydot)

        Edotzz = torch.einsum('biq,bip->pq', Rdotz[index_remain_dim0],
                            Ydot) + torch.einsum('ai,aipq->pq', Rrest, Ydot2)
        # combine the indices and values
        indices = []
        values = []

        ## for Rv
        indice_Rv = Rdotv_indices
        index1 = indice_Rv[0]
        index2 = self._assembly.RGC_list_indexStart[self._rp_index] + indice_Rv[1]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(Rdotv_values)
        indices.append(torch.stack([index2, index1], dim=0))
        values.append(Rdotv_values)
        ## for Rz
        indice_Rz = Rdotz_indices
        index1 = indice_Rz[0]
        index2 = self._assembly.RGC_list_indexStart[self._rp_index] + indice_Rz[1] + 3
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(Rdotz_values)
        indices.append(torch.stack([index2, index1], dim=0))
        values.append(Rdotz_values)
        ## for Edot2
        mat66 = torch.zeros([6, 6])
        mat66[:3, :3] = Edotvv
        mat66[3:, 3:] = Edotzz
        mat66[3:, :3] = Edotzv
        mat66[:3, 3:] = Edotzv.transpose(0, 1)

        indice_Edot2 = [
            torch.tensor([0, 1, 2, 3, 4, 5]).reshape(-1,
                                                    1).repeat(1, 6).flatten(),
            torch.tensor([0, 1, 2, 3, 4,
                        5]).reshape(1, -1).repeat(6, 1).flatten()
        ]
        index1 = indice_Edot2[0] + self._assembly.RGC_list_indexStart[self._rp_index]
        index2 = indice_Edot2[1] + self._assembly.RGC_list_indexStart[self._rp_index]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(mat66.flatten())

        # combine the indices and values
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values, dim=0)
        #endregion

        return R, indices, values

    def _calculate_Ydotz(self, RGC: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        v = RGC[self._rp_index][:3]
        z = RGC[self._rp_index][3:]
        theta = z.norm() + 1e-20
        w = (z / theta)

        epsilon_indices = [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                        [2, 1, 2, 0, 1, 0]]
        epsilon_values = [1, -1, -1, 1, 1, -1]


        # basic derivatives
        y = self._ref_location
        Y = v + self._rotation3d(z, y) - y

        der_theta = -y * torch.sin(theta) + w.view(
            1, 3) * (w.view(1, 3) * y).sum(dim=1).reshape(-1, 1) * torch.sin(
                theta) + torch.cross(w.view(1, 3), y, dim=1) * torch.cos(theta)

        der_theta2 = -y * torch.cos(theta) + w.view(
            1, 3) * (w.view(1, 3) * y).sum(dim=1).reshape(
                -1, 1) * (torch.cos(theta)) - torch.cross(
                    w.view(1, 3), y, dim=1) * torch.sin(theta)

        der_w = (torch.einsum('al,i->ail', y, w)) * (1 - torch.cos(theta))
        temp = (1 - torch.cos(theta)) * (w.view(1, 3) * y).sum(dim=1).flatten()
        for i in range(3):
            der_w[:, i, i] += temp
        for i in range(6):
            der_w[:, epsilon_indices[0][i],
                epsilon_indices[2][i]] -= epsilon_values[i] * torch.sin(
                    theta) * y[:, epsilon_indices[1][i]]

        der_w2 = torch.zeros([y.shape[0], 3, 3, 3])
        temp = (1 - torch.cos(theta)) * y
        for i in range(3):
            der_w2[:, i, i, :] += temp
            der_w2[:, i, :, i] += temp

        der_w_theta = (torch.einsum('al,i->ail', y, w)) * torch.sin(theta)
        temp = torch.sin(theta) * (w.view(1, 3) * y).sum(dim=1).flatten()
        for i in range(3):
            der_w_theta[:, i, i] += temp
        for i in range(6):
            der_w_theta[:, epsilon_indices[0][i],
                        epsilon_indices[2][i]] -= epsilon_values[
                            i] * torch.cos(theta) * y[:, epsilon_indices[1][i]]

        wdot = -torch.einsum('i,p->ip', z, z) / theta**3 + torch.eye(3) / theta
        thetadot = w
        wdot2 = 3 * torch.einsum('i,p,q->ipq', z, z, z) / theta**5
        temp = z / theta**3
        for i in range(3):
            wdot2[i, i, :] -= temp
            wdot2[i, :, i] -= temp
            wdot2[:, i, i] -= temp
        thetadot2 = wdot

        Ydot = torch.einsum('bjl,lp->bjp', der_w, wdot) + torch.einsum(
            'bj,p->bjp', der_theta, thetadot)

        Ydot2 = (
            torch.einsum('ai,pq->aipq', der_theta, thetadot2) +
            torch.einsum('ai, p, q->aipq', der_theta2, thetadot, thetadot) +
            torch.einsum('ail,lq,p->aipq', der_w_theta, wdot, thetadot))

        Ydot2 += (torch.einsum('ailm,lp,mq->aipq', der_w2, wdot, wdot) +
                torch.einsum('ail,lp,q->aipq', der_w_theta, wdot, thetadot) +
                torch.einsum('ail,lpq->aipq', der_w, wdot2))
        
        return Ydot, Ydot2

    def _rotation3d(self, rotation_vector: torch.Tensor,
                    vector0: torch.Tensor):
        """
        Rotate a 3D vector by a rotation vector
        :param rotation_vector: rotation vector (3,)
        :param vector0: 3D vector (n, 3)
        :return: 3D vector (n, 3)
        """
        vector0 = vector0.view(-1, 3)
        theta = torch.norm(rotation_vector) + 1e-20
        if theta == 0:
            return vector0
        else:
            rotation_vector = rotation_vector / theta
            rotation_vector = rotation_vector.view(1, 3)
            vector1 = vector0 * torch.cos(theta) + torch.cross(
                rotation_vector, vector0, dim=1) * torch.sin(
                    theta) + rotation_vector * (rotation_vector * vector0).sum(
                        dim=1).unsqueeze(-1) * (1 - torch.cos(theta))
        return vector1

