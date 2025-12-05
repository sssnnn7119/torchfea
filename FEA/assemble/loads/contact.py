

import numpy as np
import scipy.spatial
import torch
from .base import BaseLoad
from ..elements import BaseSurface

class ContactBase(BaseLoad):
    def __init__(self,
                 penalty_distance_g: float = 1e-5,
                 penalty_factor_g: float = 40.0,
                 penalty_degree: int = 9,
                 penalty_threshold_h: float = 1.5,
                 penalty_ratio_h: float = 0.9,
                 penalty_start_f: float = -0.6,
                 penalty_end_f: float = -0.7):
        """
        Initialize the base contact load with common parameters.

        Args:
            penalty_distance_g (float): The penalty distance for contact. When the distance between nodes is less than this value, a penalty is applied.
            penalty_factor_g (float): The penalty factor g for contact.
            penalty_degree (int): The penalty degree for contact. The degree of the penalty function.
            penalty_threshold_h (float): The penalty threshold for contact.
            penalty_ratio_h (float): The penalty ratio for contact.
            penalty_start_f (float): The penalty degree for the angle factor f. The degree of the penalty function.
            penalty_end_f (float): The penalty threshold for the angle factor f.
        """
        super().__init__()

        self._penalty_distance_g = penalty_distance_g
        """The penalty distance for contact. When the distance between nodes is less than this value, a penalty is applied."""

        self._penalty_factor_g = penalty_factor_g
        """The penalty factor g for contact."""

        self._penalty_threshold_h = penalty_threshold_h
        """The penalty threshold for contact."""

        self._penalty_ratio_h = penalty_ratio_h
        """The penalty ratio for contact."""

        self._penalty_start_f = penalty_start_f
        """The penalty degree for the angle factor f. The degree of the penalty function."""

        self._penalty_end_f = penalty_end_f
        """The penalty threshold for the angle factor f."""

        self._penalty_degree = penalty_degree
        """The penalty degree for contact. The degree of the penalty function."""

        self._point_pairs: torch.Tensor
        """The point pairs that need to be considered for self-contact."""

        self.is_self_contact: bool
        """Whether this is self-contact (True) or two-surface contact (False)."""

        self.surface_element1: BaseSurface
        """The first surface element for contact."""

        self.surface_element2: BaseSurface
        """The second surface element for contact."""

        self.surface_name1: str
        """The name of the first surface to apply the load on."""

        self.surface_name2: str
        """The name of the second surface to apply the load on."""

        self.instance_name1: str
        """The name of the first instance to apply the load on."""

        self.instance_name2: str
        """The name of the second instance to apply the load on."""

    def _overlap_check(self, dy: torch.Tensor, dn: torch.Tensor):
        distance = dy.norm(dim=-1)
        T = -(dy*dn).sum(-1)

        check = ((T<0) & (distance<0.4)).sum()

        return check>0

    def _filter_point_pairs(self, surface_element1: BaseSurface, surface_element2: BaseSurface, nodes1: torch.Tensor, nodes2: torch.Tensor, max_search_length_ratio: float = 2.0):
        """
        Filter point pairs between surfaces for contact detection.
        
        Args:
            surface_element1: First surface element
            surface_element2: Second surface element (same as first for self-contact)
            nodes: Node positions
            is_self_contact: Whether this is self-contact (affects diagonal filtering)
            
        Returns:
            tuple: (point_pairs, ratio_d) for contact detection
        """

        # Get Gaussian points for both surfaces
        elems_gaussian1 = surface_element1.gaussian_points_position(nodes1)
        elems_gaussian2 = surface_element2.gaussian_points_position(nodes2)
        
        # Calculate midpoints for initial distance filtering
        elems_mid1 = elems_gaussian1.mean(dim=0).cpu()
        elems_mid2 = elems_gaussian2.mean(dim=0).cpu()
        
        # Calculate distances between surface midpoints
        if not self.is_self_contact:
            points = torch.cat([elems_mid1, elems_mid2], dim=0).detach().cpu().numpy()
        else:
            points = elems_mid1.detach().cpu().numpy()
        kdtree = scipy.spatial.cKDTree(points)
        pairs = torch.from_numpy(kdtree.query_pairs(max_search_length_ratio * self._penalty_threshold_h, output_type='ndarray')).to(nodes1.device).T
        index_revert = torch.where(pairs[0] >= pairs[1])[0]
        pairs[:, index_revert] = pairs[:, index_revert][[1, 0]]
        if not self.is_self_contact:
            pairs = pairs[:, pairs[0] < elems_mid1.shape[0]]
            pairs = pairs[:, pairs[1] >= elems_mid1.shape[0]]
            pairs[1] -= elems_mid1.shape[0]
        
        self._point_pairs = pairs

    def _calculate_positions_and_normals(self, RGC: list[torch.Tensor], surface_element1: BaseSurface, surface_element2: BaseSurface=None):
        """
        Calculate positions and normals for contact surfaces.
        
        Args:
            RGC: Current configuration
            surface_element1: First surface element
            surface_element2: Second surface element (None for self-contact)
            
        Returns:
            tuple: (y1, n1, y2, n2) where y2=y1 and n2=n1 for self-contact
        """
        U = RGC[0]
        Y = self._assembly.nodes + U
        
        y1 = surface_element1.gaussian_points_position(Y)
        N1 = surface_element1.get_gaussian_normal(Y)
        
        if surface_element2 is None:
            # Self-contact: reuse same calculations
            return y1, N1, y1, N1
        else:
            # Two-surface contact
            y2 = surface_element2.gaussian_points_position(Y)
            N2 = surface_element2.get_gaussian_normal(Y)
            return y1, N1, y2, N2

    def show_contact_pairs(self, ind: int, RGC: list[torch.Tensor]):
        a = self
        instance1 = a._assembly.get_instance(a.instance_name1)
        instance2 = a._assembly.get_instance(a.instance_name2)
        a._filter_point_pairs(a.surface_element1, a.surface_element2, 
                                 instance1.nodes + RGC[instance1._RGC_index], 
                                 instance2.nodes + RGC[instance2._RGC_index])

        # U = U.clone().detach().requires_grad_(True)
        Y1 = instance1.nodes + RGC[instance1._RGC_index]
        Y2 = instance2.nodes + RGC[instance2._RGC_index]

        num_g1 = a.surface_element1._num_gaussian
        num_g2 = a.surface_element2._num_gaussian
        num_e1 = a.surface_element1._elems.shape[0]
        num_e2 = a.surface_element2._elems.shape[0]
        num_n1 = a.surface_element1.num_nodes_per_elem
        num_n2 = a.surface_element2.num_nodes_per_elem

        # Calculate positions and normals for both surfaces
        Ye1 = Y1[a.surface_element1._elems]
        Ye2 = Y2[a.surface_element2._elems]

        y1 = torch.einsum('eai, ga->gei', Ye1, a.surface_element1.shape_function_gaussian[0])
        y2 = torch.einsum('eai, ga->gei', Ye2, a.surface_element2.shape_function_gaussian[0])

        NR1 = torch.einsum('gma, eai->gemi', a.surface_element1.shape_function_gaussian[1], Ye1)
        NR2 = torch.einsum('gma, eai->gemi', a.surface_element2.shape_function_gaussian[1], Ye2)
        
        N1 = torch.cross(NR1[:, :, 0, :], NR1[:, :, 1, :], dim=-1)
        N2 = torch.cross(NR2[:, :, 0, :], NR2[:, :, 1, :], dim=-1)

        nnorm1 = N1.norm(dim=-1)
        nnorm2 = N2.norm(dim=-1)
        n1 = N1 / nnorm1[:, :, None]
        n2 = N2 / nnorm2[:, :, None]

        num_p = a._point_pairs.shape[1]
        
        # Create extended tensor for two surfaces
        E1 = torch.zeros([num_g1, num_p, 2, 3], device=Y1.device)
        E1[:, :, 0] = y1[:, a._point_pairs[0]]
        E1[:, :, 1] = n1[:, a._point_pairs[0]]

        E2 = torch.zeros([num_g2, num_p, 2, 3], device=Y2.device)
        E2[:, :, 0] = y2[:, a._point_pairs[1]]
        E2[:, :, 1] = n2[:, a._point_pairs[1]]

        dy = E1[:, None, :, 0, :] - E2[None, :, :, 0, :]
        dn = E1[:, None, :, 1, :] - E2[None, :, :, 1, :]

        M = (E1[:, None, :, 1, :] * E2[None, :, :, 1, :]).sum(dim=-1)
        MM = (a._penalty_start_f - M) / (a._penalty_start_f - a._penalty_end_f)
        MM = MM.clamp(0, 1)
        f = MM**3 * (6*MM**2 - 15*MM + 10)

        D = a._penalty_distance_g + (dn * dy).sum(dim=-1) / 2
        D[D < 0] = 0
        g = (D / a._penalty_factor_g) ** a._penalty_degree
        
        L = dy.norm(dim=-1)
        T = (a._penalty_threshold_h - L) / (a._penalty_ratio_h * a._penalty_threshold_h)
        T = T.clamp(0, 1)
        h = T**3 * (6*T**2 - 15*T + 10)

        penalty = g * f * h

        # Filter zero penalty pairs
        index_remain = torch.where(penalty.sum([0,1]) > 0)[0]

        point_pairs = a._point_pairs[:, index_remain]
    
        from mayavi import mlab

        ind_now = torch.where(point_pairs[0]==point_pairs[0].unique()[ind])[0]

        point_pairs_show = point_pairs[:, ind_now]
        mlab.figure()
        mlab.triangular_mesh((instance1.nodes+RGC[instance1._RGC_index]).cpu()[:, 0], (instance1.nodes+RGC[instance1._RGC_index]).cpu()[:, 1], (instance1.nodes+RGC[instance1._RGC_index]).cpu()[:, 2], a.surface_element1._elems.cpu().numpy(), color=(0.5, 0.5, 0.5), opacity=0.5)
        mlab.triangular_mesh((instance2.nodes+RGC[instance2._RGC_index]).cpu()[:, 0], (instance2.nodes+RGC[instance2._RGC_index]).cpu()[:, 1], (instance2.nodes+RGC[instance2._RGC_index]).cpu()[:, 2], a.surface_element2._elems[:, [0,1,2]].cpu().numpy(), color=(0.5, 0.5, 0.5), opacity=0.5)
        mlab.triangular_mesh((instance2.nodes+RGC[instance2._RGC_index]).cpu()[:, 0], (instance2.nodes+RGC[instance2._RGC_index]).cpu()[:, 1], (instance2.nodes+RGC[instance2._RGC_index]).cpu()[:, 2], a.surface_element2._elems[:, [0,2,3]].cpu().numpy(), color=(0.5, 0.5, 0.5), opacity=0.5)

        for gind in range(y1.shape[0]):
            mlab.points3d((y1[gind])[point_pairs_show[0], 0].cpu(), (y1[gind])[point_pairs_show[0], 1].cpu(), (y1[gind])[point_pairs_show[0], 2].cpu(), color=(1, 0, 0), scale_factor = 0.2)
            mlab.quiver3d((y1[gind])[point_pairs_show[0], 0].cpu(), (y1[gind])[point_pairs_show[0], 1].cpu(), (y1[gind])[point_pairs_show[0], 2].cpu(), (n1[gind])[point_pairs_show[0], 0].cpu(), (n1[gind])[point_pairs_show[0], 1].cpu(), (n1[gind])[point_pairs_show[0], 2].cpu(), color=(1, 0, 0), scale_factor = 1.0)

        for gind in range(y2.shape[0]):
            mlab.points3d((y2[gind])[point_pairs_show[1], 0].cpu(), (y2[gind])[point_pairs_show[1], 1].cpu(), (y2[gind])[point_pairs_show[1], 2].cpu(), color=(0, 0, 1), scale_factor = 0.2)
            mlab.quiver3d((y2[gind])[point_pairs_show[1], 0].cpu(), (y2[gind])[point_pairs_show[1], 1].cpu(), (y2[gind])[point_pairs_show[1], 2].cpu(), (n2[gind])[point_pairs_show[1], 0].cpu(), (n2[gind])[point_pairs_show[1], 1].cpu(), (n2[gind])[point_pairs_show[1], 2].cpu(), color=(0, 0, 1), scale_factor = 1.0)

        mlab.show()

class ContactSelf(ContactBase):
    """
    Class representing self-contact loads in the finite element model.
    """

    def __init__(self, instance_name: str, surface_name: str,
                 ignore_min_normal: float = -0.5,
                 ignore_max_normal: float = 0.0, 
                 initial_detact_ratio: float = 2.0, **kwargs):
        """
        Initialize the self-contact load.

        Args:
            surface_name (str): The name of the surface to apply the load on.
            **kwargs: Additional parameters passed to ContactBase.
        """

        super().__init__(**kwargs)

        
        self._ignore_min_normal = ignore_min_normal
        """The minimum initial normal distance to ignore for contact."""
        self._ignore_max_normal = ignore_max_normal
        """The maximum initial normal distance to ignore for contact."""

        self.surface_name = surface_name
        """The name of the surface to apply the load on."""

        self.instance_name = instance_name
        """The name of the instance to apply the load on."""

        self.surface_name1 = self.surface_name2 = surface_name
        self.instance_name1 = self.instance_name2 = instance_name

        self.surface_element: BaseSurface
        """The list of surface elements for self-contact."""

        self.is_self_contact = True

        self._ratio: torch.Tensor
        """The ratio to avoid the intersection of surfaces."""

        self._initial_detact_ratio = initial_detact_ratio
        """The initial detach ratio to avoid the initial intersection of surfaces."""
    
    def initialize(self, assembly):
        
        super().initialize(assembly)

        # filter the point pairs
        self.surface_element = assembly.get_instance(self.instance_name).surfaces.get_elements(self.surface_name)[0]

        self.surface_element1 = self.surface_element2 = self.surface_element

        instance = assembly.get_instance(self.instance_name)

        self._filter_point_pairs(
            self.surface_element, self.surface_element, instance.nodes)

    def _filter_point_pairs(self, surface_element1: BaseSurface, surface_element2: BaseSurface, nodes: torch.Tensor):
        super()._filter_point_pairs(surface_element1, surface_element2, nodes1=nodes, nodes2=nodes, max_search_length_ratio=self._initial_detact_ratio)
        
        def _ratio_d_func(dx: torch.Tensor, dm: torch.Tensor):
            """
            Calculate the ratio for self-contact to avoid the calculation of the nearest distance.

            Args:
                dx (torch.Tensor): The normalized distance vector between points.
                dm (torch.Tensor): The difference in normal vectors between points.

            Returns:
                torch.Tensor: The ratio for self-contact.
            """
            dx = dx / dx.norm(dim=-1, keepdim=True)

            T = - (dm * dx).sum(-1)
            T = (T - self._ignore_min_normal) / (self._ignore_max_normal - self._ignore_min_normal)
            T = T.clamp(0, 1)
            return 6 * T**5 - 15 * T**4 + 10 * T**3

        instance = self._assembly.get_instance(self.instance_name)

        # Get surface normals
        normal1 = surface_element1.get_gaussian_normal(instance.nodes)
        normal2 = surface_element2.get_gaussian_normal(instance.nodes)
        elems_gaussian1 = surface_element1.gaussian_points_position(instance.nodes)
        elems_gaussian2 = surface_element2.gaussian_points_position(instance.nodes)
        normal1 = normal1 / normal1.norm(dim=-1, keepdim=True)
        normal2 = normal2 / normal2.norm(dim=-1, keepdim=True)

        # Calculate normal differences and position differences
        dm = normal1[:, None, self._point_pairs[0], :] - normal2[None, :, self._point_pairs[1], :]
        dy = elems_gaussian1[:, None, self._point_pairs[0], :] - elems_gaussian2[None, :, self._point_pairs[1], :]
        dr = dy / dy.norm(dim=-1, keepdim=True)

        # Calculate ratio based on normal alignment
        ratio_d = _ratio_d_func(dx=dr, dm=dm)
        index_remain = (ratio_d.sum([0, 1]) > 0)

        self._point_pairs = self._point_pairs[:, index_remain]
        self._ratio = ratio_d[:, :, index_remain]

    def get_potential_energy(self, RGC):

        instance = self._assembly.get_instance(self.instance_name)
        self._filter_point_pairs(self.surface_element, self.surface_element, instance.nodes + RGC[instance._RGC_index])

        weight = torch.einsum('gp, g, Gp, G->gGp', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        U = RGC[instance._RGC_index]
        # U = U.detach().clone().requires_grad_()
        Y = instance.nodes + U

        num_g = self.surface_element._num_gaussian
        
        num_e = self.surface_element._elems.shape[0]
        num_n = self.surface_element.num_nodes_per_elem

        Ye = Y[self.surface_element._elems]

        y = torch.einsum('eai, ga->gei', Ye, self.surface_element.shape_function_gaussian[0])
        NR = torch.einsum('gma, eai->gemi', self.surface_element.shape_function_gaussian[1], Ye)
        N = torch.cross(NR[:, :, 0, :], NR[:, :, 1, :], dim=-1)

        nnorm = N.norm(dim=-1)
        n = N / nnorm[:, :, None]

        num_p = self._point_pairs.shape[1]
        E = torch.zeros([num_g, num_p, 2, 2, 3], device=U.device) # e1/e2, y/n, 0/1/2

        E[:, :, 0, 0] = y[:, self._point_pairs[0]]
        E[:, :, 1, 0] = y[:, self._point_pairs[1]]
        E[:, :, 0, 1] = n[:, self._point_pairs[0]]
        E[:, :, 1, 1] = n[:, self._point_pairs[1]]

        dy = E[:, None, :, 0, 0, :] - E[None, :, :, 1, 0, :]
        dn = E[:, None, :, 0, 1, :] - E[None, :, :, 1, 1, :]

        M = (E[:, None, :, 0, 1, :] * E[None, :, :, 1, 1, :]).sum(dim=-1)
        MM = (self._penalty_start_f - M) / (self._penalty_start_f-self._penalty_end_f)
        MM = MM.clamp(0, 1)
        f = MM**3 * (6*MM**2 - 15*MM + 10)

        D = (dn * dy).sum(dim=-1) / 2
        g = torch.exp(D * self._penalty_factor_g) * self._penalty_distance_g


        L = dy.norm(dim=-1)
        T = (self._penalty_threshold_h - L) / (self._penalty_ratio_h * self._penalty_threshold_h)
        T = T.clamp(0, 1)
        h = T**3 * (6*T**2 - 15*T + 10)

        penalty = self._ratio * g * f * h * weight
        potential_energy = penalty.sum()
        return -potential_energy


    def get_stiffness(self, RGC, if_onlyforce: bool = False, *args, **kwargs):
        

        instance = self._assembly.get_instance(self.instance_name)
        self._filter_point_pairs(self.surface_element, self.surface_element, instance.nodes + RGC[instance._RGC_index])

        weight0 = torch.einsum('gp, g, Gp, G->gGp', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        U = RGC[instance._RGC_index]
        # U = U.detach().clone().requires_grad_()
        Y = instance.nodes + U

        num_g = self.surface_element._num_gaussian
        
        num_e = self.surface_element._elems.shape[0]
        num_n = self.surface_element.num_nodes_per_elem

        Ye = Y[self.surface_element._elems]

        y = torch.einsum('eai, ga->gei', Ye, self.surface_element.shape_function_gaussian[0])
        NR = torch.einsum('gma, eai->gemi', self.surface_element.shape_function_gaussian[1], Ye)
        N = torch.cross(NR[:, :, 0, :], NR[:, :, 1, :], dim=-1)

        nnorm = N.norm(dim=-1)
        n = N / nnorm[:, :, None]

        num_p = self._point_pairs.shape[1]
        E0 = torch.zeros([num_g, num_p, 2, 2, 3], device=U.device) # e1/e2, y/n, 0/1/2

        E0[:, :, 0, 0] = y[:, self._point_pairs[0]]
        E0[:, :, 1, 0] = y[:, self._point_pairs[1]]
        E0[:, :, 0, 1] = n[:, self._point_pairs[0]]
        E0[:, :, 1, 1] = n[:, self._point_pairs[1]]

        dy0 = E0[:, None, :, 0, 0, :] - E0[None, :, :, 1, 0, :]
        dn0 = E0[:, None, :, 0, 1, :] - E0[None, :, :, 1, 1, :]

        M0 = (E0[:, None, :, 0, 1, :] * E0[None, :, :, 1, 1, :]).sum(dim=-1)
        MM0 = (self._penalty_start_f - M0) / (self._penalty_start_f-self._penalty_end_f)
        MM0 = MM0.clamp(0, 1)
        f0 = MM0**3 * (6*MM0**2 - 15*MM0 + 10)

        D0 = (dn0 * dy0).sum(dim=-1) / 2
        g0 = torch.exp(D0 * self._penalty_factor_g) * self._penalty_distance_g


        L0 = dy0.norm(dim=-1)
        T0 = (self._penalty_threshold_h - L0) / (self._penalty_ratio_h * self._penalty_threshold_h)
        T0 = T0.clamp(0, 1)
        h0 = T0**3 * (6*T0**2 - 15*T0 + 10)

        penalty = self._ratio * g0 * f0 * h0 * weight0

        # filter the zero penalty pairs
        index_remain_total = torch.where(penalty.sum([0,1])>1e-12)[0]

        
        num_p = index_remain_total.shape[0]

        if num_p == 0:
            # No active contact pairs
            return torch.tensor([], dtype=torch.int64), torch.tensor([]), torch.tensor([[], []], dtype=torch.int64), torch.tensor([])
        

        pdU_indices_total = [] 
        pdU_values_total = []
        pdU_2_indices_total = []
        pdU_2_values_total = []

        index_now = 0
        batch_size = 5000
        while True:

            index_remain = index_remain_total[index_now:index_now+batch_size]
            num_p = index_remain.shape[0]
            if index_remain.shape[0] == 0:
                break

            point_pairs = self._point_pairs[:, index_remain]
            D = D0[:, :, index_remain]
            M = M0[:, :, index_remain]
            MM = MM0[:, :, index_remain]
            E = E0[:, index_remain]
            T = T0[:, :, index_remain]
            L = L0[:, :, index_remain]
            dy = dy0[:, :, index_remain]
            dn = dn0[:, :, index_remain]
            f = f0[:, :, index_remain]
            g = g0[:, :, index_remain]
            h = h0[:, :, index_remain]
            ratio = self._ratio[:, :, index_remain]
            weight = weight0[:, :, index_remain]
            
            # if index_remain.shape[0] > 0:
                # print('  Contact pairs: ', index_remain.shape[0], '\t surface name: ', self.surface_name)
                # from mayavi import mlab
                # ind = 0
                # point_pairs_show = point_pairs[:, [ind]]
                # mlab.figure()
                # mlab.triangular_mesh((self._fea.nodes+RGC[0]).cpu()[:, 0], (self._fea.nodes+RGC[0]).cpu()[:, 1], (self._fea.nodes+RGC[0]).cpu()[:, 2], self.surface_element._elems.cpu().numpy(), color=(0.5, 0.5, 0.5))
                # mlab.points3d(y[0][point_pairs_show[0], 0].cpu(), y[0][point_pairs_show[0], 1].cpu(), y[0][point_pairs_show[0], 2].cpu(), color=(1, 0, 0), scale_factor = 0.2)
                # mlab.points3d(y[0][point_pairs_show[1], 0].cpu(), y[0][point_pairs_show[1], 1].cpu(), y[0][point_pairs_show[1], 2].cpu(), color=(0, 0, 1), scale_factor = 0.2)

                # mlab.quiver3d(y[0][point_pairs_show[0], 0].cpu(), y[0][point_pairs_show[0], 1].cpu(), y[0][point_pairs_show[0], 2].cpu(), n[0][point_pairs_show[0], 0].cpu(), n[0][point_pairs_show[0], 1].cpu(), n[0][point_pairs_show[0], 2].cpu(), scale_factor=10.)
                # mlab.quiver3d(y[0][point_pairs_show[1], 0].cpu(), y[0][point_pairs_show[1], 1].cpu(), y[0][point_pairs_show[1], 2].cpu(), n[0][point_pairs_show[1], 0].cpu(), n[0][point_pairs_show[1], 1].cpu(), n[0][point_pairs_show[1], 2].cpu(), scale_factor=10.)
                # mlab.show()

            # # Compute the potential energy
            # potential_energy = penalty.sum()

            ndN = torch.einsum('ij, ge->geij', torch.eye(3), 1/nnorm) + \
                torch.einsum('gei, gej, ge->geij', n, n, -1/nnorm)
            ndN_2 = torch.einsum('ij, gek, ge->geijk', torch.eye(3), n, -1/nnorm**2) + \
                torch.einsum('geik, gej, ge->geijk', ndN, n, -1/nnorm) + \
                torch.einsum('gei, gejk, ge->geijk', n, ndN, -1/nnorm) + \
                torch.einsum('gei, gej, gek, ge->geijk', n, n, n, 1/nnorm**2)
            
            ydUe = self.surface_element.shape_function_gaussian[0]

            epsilon = torch.zeros([3, 3, 3])
            epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
            epsilon[1, 0, 2] = epsilon[2, 1, 0] = epsilon[0, 2, 1] = -1

            NdUe = torch.einsum('ijl, geja->geial', 
                                epsilon, 
                                torch.einsum('gei, ga->geia', NR[:, :, 0], self.surface_element.shape_function_gaussian[1][:, 1]) - 
                                torch.einsum('gei, ga->geia', NR[:, :, 1], self.surface_element.shape_function_gaussian[1][:, 0]))
            NdUe_2 = torch.einsum('ipl, gab->gialbp', epsilon, 
                                torch.einsum('gb,ga->gab', self.surface_element.shape_function_gaussian[1][:, 0], self.surface_element.shape_function_gaussian[1][:, 1])-
                                torch.einsum('gb,ga->gab', self.surface_element.shape_function_gaussian[1][:, 1], self.surface_element.shape_function_gaussian[1][:, 0]))

            ndUe = torch.einsum('geij, geial->gejal', ndN, NdUe)

            ndUe_2 = torch.einsum('geijk, geial, gekbp->gejalbp', ndN_2, NdUe, NdUe) + \
                    torch.einsum('geij, gialbp->gejalbp', ndN, NdUe_2)

            edUe = torch.zeros([num_g, num_e, 2, 3, num_n, 3])
            edUe[:, :, 1] = ndUe
            edUe[:, :, 0, 0, :, 0] = ydUe[:, None, :]
            edUe[:, :, 0, 1, :, 1] = ydUe[:, None, :]
            edUe[:, :, 0, 2, :, 2] = ydUe[:, None, :]

            edUe_2 = torch.zeros([num_g, num_e, 2, 3, num_n, 3, num_n, 3])
            edUe_2[:, :, 1] = ndUe_2

            # g = torch.exp(D * self._penalty_factor_g) * self._penalty_distance_g
            gdD = (self._penalty_factor_g) * g
            gdD_2 = (self._penalty_factor_g**2) * g

            gdE = torch.zeros([num_g, num_g, num_p, 2, 2, 3])
            gdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dn)
            gdE[:, :, :, 1, 0, :] = -gdE[:, :, :, 0, 0, :]
            gdE[:, :, :, 0, 1, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dy)
            gdE[:, :, :, 1, 1, :] = -gdE[:, :, :, 0, 1, :]

            gdE_2 = torch.zeros([num_g, num_g, num_p, 2, 2, 3, 2, 2, 3])
            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dn)
            gdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dy) + \
                    torch.einsum('gGp, ij->gGpij', gdD / 2, torch.eye(3))
            gdE_2[:, :, :, 0, 0, :, 0, 1, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 1, :] = tmp

            tmp = tmp.permute([0, 1, 2, 4, 3])
            gdE_2[:, :, :, 0, 1, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 1, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 1, 0, :] = tmp

            temp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dy, dy)
            gdE_2[:, :, :, 0, 1, :, 0, 1, :] = temp
            gdE_2[:, :, :, 1, 1, :, 0, 1, :] = -temp
            gdE_2[:, :, :, 0, 1, :, 1, 1, :] = -temp
            gdE_2[:, :, :, 1, 1, :, 1, 1, :] = temp

            # MM = (self._penalty_start_f - M) / (self._penalty_start_f-self._penalty_end_f)
            # MM = MM.clamp(0, 1)
            # f = MM**3 * (6*MM**2 - 15*MM + 10)
            fdM = -30*MM**2*(MM-1)**2 / (self._penalty_start_f-self._penalty_end_f)
            fdM_2 = 60*MM*(MM-1)*(2*MM-1) / (self._penalty_start_f-self._penalty_end_f)**2
            fdM[MM>=1] = 0 
            fdM[MM<=0] = 0
            fdM_2[MM>=1] = 0 
            fdM_2[MM<=0] = 0
            # M = (E[:, None, :, 0, 1, :] * E[None, :, :, 1, 1, :]).sum(dim=-1)

            fdE = torch.zeros([num_g, num_g, num_p, 2, 2, 3])
            fdE[:, :, :, 0, 1, :] = torch.einsum('gGp, gGpi->gGpi', fdM, E[:, None, :, 1, 1, :])
            fdE[:, :, :, 1, 1, :] = torch.einsum('gGp, gGpi->gGpi', fdM, E[None, :, :, 0, 1, :])

            fdE_2 = torch.zeros([num_g, num_g, num_p, 2, 2, 3, 2, 2, 3])
            fdE_2[:, :, :, 0, 1, :, 0, 1, :] = torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 1, 1, :], E[:, None, :, 1, 1, :])
            fdE_2[:, :, :, 0, 1, :, 1, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
                                                torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 1, 1, :], E[:, None, :, 0, 1, :])
            fdE_2[:, :, :, 1, 1, :, 1, 1, :] = torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 0, 1, :], E[:, None, :, 0, 1, :])
            fdE_2[:, :, :, 1, 1, :, 0, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
            torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 0, 1, :], E[:, None, :, 1, 1, :])

            hdE = torch.zeros([num_g, num_g, num_p, 2, 2, 3])

            # L = dy.norm(dim=-1)
            # T = (self._penalty_distance - L) / (0.5 * self._penalty_distance)
            # T = T.clamp(0, 1)
            # h = T**3 * (6*T**2 - 15*T + 10)
            Lddy = torch.einsum('gGpi, gGp->gGpi', dy, 1/L)
            Lddy_2 = torch.einsum('ij, gGp->gGpij', torch.eye(3), 1/L) + torch.einsum('gGpi, gGpj, gGp->gGpij', dy, Lddy, -1/L**2)
            hdL = -30*T**2*(T-1)**2 / (self._penalty_ratio_h * self._penalty_threshold_h)
            hdL_2 = 60*T*(T-1)*(2*T-1) / (self._penalty_ratio_h * self._penalty_threshold_h)**2
            hdL[T>=1] = 0
            hdL[T<=0] = 0
            hdL_2[T>=1] = 0
            hdL_2[T<=0] = 0
            hdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', hdL, Lddy)
            hdE[:, :, :, 1, 0, :] = -hdE[:, :, :, 0, 0, :]

            hdE_2 = torch.zeros([num_g, num_g, num_p, 2, 2, 3, 2, 2, 3])
            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', hdL_2, Lddy, Lddy) + \
                    torch.einsum('gGp, gGpij->gGpij', hdL, Lddy_2)
            hdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
            hdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
            hdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
            hdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

            pdE = torch.einsum('gGpmxi, gGp, gGp->gGpmxi', fdE, g, h) + \
                torch.einsum('gGp, gGpmxi, gGp->gGpmxi', f, gdE, h) + \
                torch.einsum('gGp, gGp, gGpmxi->gGpmxi', f, g, hdE)

            pdE = pdE * ratio[:, :, :, None, None, None] * weight[:, :, :, None, None, None]

            pdE_2 = torch.einsum('gGpmxinyj, gGp, gGp->gGpmxinyj', fdE_2, g, h) + \
                    torch.einsum('gGpmxi, gGpnyj, gGp->gGpmxinyj', fdE, gdE, h) + \
                    torch.einsum('gGpmxi, gGp, gGpnyj->gGpmxinyj', fdE, g, hdE) + \
                    \
                    torch.einsum('gGpnyj, gGpmxi, gGp->gGpmxinyj', fdE, gdE, h) +\
                    torch.einsum('gGp, gGpmxinyj, gGp->gGpmxinyj', f, gdE_2, h) +\
                    torch.einsum('gGp, gGpmxi, gGpnyj->gGpmxinyj', f, gdE, hdE) +\
                    \
                    torch.einsum('gGpnyj, gGp, gGpmxi->gGpmxinyj', fdE, g, hdE)+\
                    torch.einsum('gGp, gGpnyj, gGpmxi->gGpmxinyj', f, gdE, hdE)+\
                    torch.einsum('gGp, gGp, gGpmxinyj->gGpmxinyj', f, g, hdE_2)
            
            pdE_2 = torch.einsum('gGpmxinyj, gGp, gGp->gGpmxinyj', pdE_2, ratio, weight)
    
            # pdUe = torch.zeros([num_e, num_n, 3])
            pdEsum0 = pdE.sum(0)
            pdEsum1 = pdE.sum(1)
            pdUe_values0 = torch.einsum('gpxi, gpxial->pal', pdEsum1[:, :, 0], edUe[:, point_pairs[0]])
            pdUe_values1 = torch.einsum('Gpxi, Gpxial->pal', pdEsum0[:, :, 1], edUe[:, point_pairs[1]])

            # for i in range(point_pairs.shape[1]):
            #     pdUe[point_pairs[0, i]] += pdUe_values0[i]
            #     pdUe[point_pairs[1, i]] += pdUe_values1[i]

            pdU_values = torch.stack([pdUe_values0, pdUe_values1], dim=0)
            tri_ind = point_pairs.cpu()

            pdU_indices = self.surface_element._elems[tri_ind].to(torch.int64)
            pdU_indices = torch.stack([pdU_indices*3, pdU_indices*3+1, pdU_indices*3+2], dim=-1)
            pdU_indices = pdU_indices.to(self._assembly.device)

            if if_onlyforce:
                return pdU_indices.reshape([2, -1]), pdU_values.flatten(), torch.tensor([[], []], dtype=torch.int64), torch.tensor([])

            # pdU = torch.zeros_like(Y).flatten().scatter_add_(0, pdU_indices.flatten(), pdU_values.flatten()).reshape([-1, 3])

            pdUe_2_values00 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(1)[:, :, 0, :, :, 0], edUe[:, point_pairs[0]], edUe[:, point_pairs[0]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum1[:, :, 0], edUe_2[:, point_pairs[0]])
            
            pdUe_2_values01 = torch.einsum('gGpxiyj, gpxial, GpyjbL->palbL', pdE_2[:, :, :, 0, :, :, 1], edUe[:, point_pairs[0]], edUe[:, point_pairs[1]])

            pdUe_2_values10 = torch.einsum('gGpxiyj, Gpxial, gpyjbL->palbL', pdE_2[:, :, :, 1, :, :, 0], edUe[:, point_pairs[1]], edUe[:, point_pairs[0]])
            
            pdUe_2_values11 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(0)[:, :, 1, :, :, 1], edUe[:, point_pairs[1]], edUe[:, point_pairs[1]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum0[:, :, 1], edUe_2[:, point_pairs[1]])

            pdU_2_values = torch.stack([pdUe_2_values00, pdUe_2_values01, pdUe_2_values10, pdUe_2_values11], dim=0)


            # pdU_2 = torch.sparse_coo_tensor(pdU_2_indices_.reshape([4, -1]), pdU_2_values.flatten(), size=Y.shape*2)


            pdU_2_indices00 = torch.stack([
                self.surface_element._elems[tri_ind[0]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
                self.surface_element._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
            ])

            pdU_2_indices01 = torch.stack([
                self.surface_element._elems[tri_ind[0]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
                self.surface_element._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
            ])

            pdU_2_indices10 = torch.stack([
                self.surface_element._elems[tri_ind[1]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
                self.surface_element._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
            ])

            pdU_2_indices11 = torch.stack([
                self.surface_element._elems[tri_ind[1]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
                self.surface_element._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
            ])

            pdU_2_indices_ = torch.stack([pdU_2_indices00, pdU_2_indices01, pdU_2_indices10, pdU_2_indices11], dim=1)
            pdU_2_indices = torch.stack([pdU_2_indices_[0]*3+pdU_2_indices_[1], pdU_2_indices_[2]*3+pdU_2_indices_[3]], dim=0).to(self._assembly.device)

            pdU_indices_total.append(pdU_indices.flatten())
            pdU_values_total.append(pdU_values.flatten())
            pdU_2_indices_total.append(pdU_2_indices.reshape([2, -1]))
            pdU_2_values_total.append(pdU_2_values.flatten())

            index_now += batch_size

        pdU_indices = torch.cat(pdU_indices_total, dim=0)
        pdU_values = torch.cat(pdU_values_total, dim=0)
        pdU_2_indices = torch.cat(pdU_2_indices_total, dim=1)
        pdU_2_values = torch.cat(pdU_2_values_total, dim=0)

        index_start = self._assembly.RGC_list_indexStart[instance._RGC_index]
        return pdU_indices + index_start, -pdU_values, pdU_2_indices + index_start, -pdU_2_values

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._assembly.get_instance(self.instance_name)._RGC_index][self.surface_element._elems.flatten().unique().cpu()] = True
        return RGC_remain_index
    
class Contact(ContactBase):
    """
    Contact between two surfaces.

    Args:
        surface_name1 (str): The name of the first surface element.
        surface_name2 (str): The name of the second surface element.
        **kwargs: Additional parameters passed to ContactBase.
    """

    def __init__(self,
            instance_name1: str,
            instance_name2: str,
            surface_name1: str,
            surface_name2: str,
            **kwargs):
        
        super().__init__(**kwargs)

        self.surface_name1 = surface_name1
        """The name of the first surface to apply the load on."""

        self.surface_name2 = surface_name2
        """The name of the second surface to apply the load on."""

        self.surface_element1: BaseSurface
        """The first surface element for contact."""

        self.surface_element2: BaseSurface
        """The second surface element for contact."""

        self.instance_name1 = instance_name1
        """The name of the first instance containing the surface."""

        self.instance_name2 = instance_name2
        """The name of the second instance containing the surface."""

        self.is_self_contact = False

    def initialize(self, assembly):
        super().initialize(assembly)

        # Get surface elements from FEA model
        self.surface_element1 = assembly.get_instance(self.instance_name1).surfaces.get_elements(self.surface_name1)[0]
        self.surface_element2 = assembly.get_instance(self.instance_name2).surfaces.get_elements(self.surface_name2)[0]

        # Filter point pairs between the two surfaces
        self._filter_point_pairs(
            self.surface_element1, self.surface_element2, assembly.get_instance(self.instance_name1).nodes, assembly.get_instance(self.instance_name2).nodes)
        
    def get_potential_energy(self, RGC):
        
        instance1 = self._assembly.get_instance(self.instance_name1)
        instance2 = self._assembly.get_instance(self.instance_name2)
        self._filter_point_pairs(self.surface_element1, self.surface_element2, 
                                 instance1.nodes + RGC[instance1._RGC_index], 
                                 instance2.nodes + RGC[instance2._RGC_index])

        weight = torch.einsum('ge, g, Ge, G->gGe', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        # U = U.clone().detach().requires_grad_(True)
        Y1 = instance1.nodes + RGC[instance1._RGC_index]
        Y2 = instance2.nodes + RGC[instance2._RGC_index]

        num_g1 = self.surface_element1._num_gaussian
        num_g2 = self.surface_element2._num_gaussian
        num_e1 = self.surface_element1._elems.shape[0]
        num_e2 = self.surface_element2._elems.shape[0]
        num_n1 = self.surface_element1.num_nodes_per_elem
        num_n2 = self.surface_element2.num_nodes_per_elem

        # Calculate positions and normals for both surfaces
        Ye1 = Y1[self.surface_element1._elems]
        Ye2 = Y2[self.surface_element2._elems]

        y1 = torch.einsum('eai, ga->gei', Ye1, self.surface_element1.shape_function_gaussian[0])
        y2 = torch.einsum('eai, ga->gei', Ye2, self.surface_element2.shape_function_gaussian[0])

        NR1 = torch.einsum('gma, eai->gemi', self.surface_element1.shape_function_gaussian[1], Ye1)
        NR2 = torch.einsum('gma, eai->gemi', self.surface_element2.shape_function_gaussian[1], Ye2)
        
        N1 = torch.cross(NR1[:, :, 0, :], NR1[:, :, 1, :], dim=-1)
        N2 = torch.cross(NR2[:, :, 0, :], NR2[:, :, 1, :], dim=-1)

        nnorm1 = N1.norm(dim=-1)
        nnorm2 = N2.norm(dim=-1)
        n1 = N1 / nnorm1[:, :, None]
        n2 = N2 / nnorm2[:, :, None]

        num_p = self._point_pairs.shape[1]
        
        # Create extended tensor for two surfaces
        E1 = torch.zeros([num_g1, num_p, 2, 3], device=Y1.device)
        E1[:, :, 0] = y1[:, self._point_pairs[0]]
        E1[:, :, 1] = n1[:, self._point_pairs[0]]

        E2 = torch.zeros([num_g2, num_p, 2, 3], device=Y2.device)
        E2[:, :, 0] = y2[:, self._point_pairs[1]]
        E2[:, :, 1] = n2[:, self._point_pairs[1]]

        dy = E1[:, None, :, 0, :] - E2[None, :, :, 0, :]
        dn = E1[:, None, :, 1, :] - E2[None, :, :, 1, :]

        M = (E1[:, None, :, 1, :] * E2[None, :, :, 1, :]).sum(dim=-1)
        MM = (self._penalty_start_f - M) / (self._penalty_start_f - self._penalty_end_f)
        MM = MM.clamp(0, 1)
        f = MM**3 * (6*MM**2 - 15*MM + 10)

        D = (dn * dy).sum(dim=-1) / 2
        g = torch.exp(D * self._penalty_factor_g) * self._penalty_distance_g
        
        L = dy.norm(dim=-1)
        T = (self._penalty_threshold_h - L) / (self._penalty_ratio_h * self._penalty_threshold_h)
        T = T.clamp(0, 1)
        h = T**3 * (6*T**2 - 15*T + 10)

        penalty = g * f * h * weight
        
        # Compute the potential energy
        potential_energy = penalty.sum()
        return -potential_energy

    def get_stiffness(self, RGC, if_onlyforce=False, *args, **kwargs):
        instance1 = self._assembly.get_instance(self.instance_name1)
        instance2 = self._assembly.get_instance(self.instance_name2)
        self._filter_point_pairs(self.surface_element1, self.surface_element2, 
                                 instance1.nodes + RGC[instance1._RGC_index], 
                                 instance2.nodes + RGC[instance2._RGC_index])

        weight0 = torch.einsum('gp, g, Gp, G->gGp', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        # U = U.clone().detach().requires_grad_(True)
        Y1 = instance1.nodes + RGC[instance1._RGC_index]
        Y2 = instance2.nodes + RGC[instance2._RGC_index]

        num_g1 = self.surface_element1._num_gaussian
        num_g2 = self.surface_element2._num_gaussian
        num_e1 = self.surface_element1._elems.shape[0]
        num_e2 = self.surface_element2._elems.shape[0]
        num_n1 = self.surface_element1.num_nodes_per_elem
        num_n2 = self.surface_element2.num_nodes_per_elem

        # Calculate positions and normals for both surfaces
        Ye1 = Y1[self.surface_element1._elems]
        Ye2 = Y2[self.surface_element2._elems]

        y1 = torch.einsum('eai, ga->gei', Ye1, self.surface_element1.shape_function_gaussian[0])
        y2 = torch.einsum('eai, ga->gei', Ye2, self.surface_element2.shape_function_gaussian[0])

        NR1 = torch.einsum('gma, eai->gemi', self.surface_element1.shape_function_gaussian[1], Ye1)
        NR2 = torch.einsum('gma, eai->gemi', self.surface_element2.shape_function_gaussian[1], Ye2)
        
        N1 = torch.cross(NR1[:, :, 0, :], NR1[:, :, 1, :], dim=-1)
        N2 = torch.cross(NR2[:, :, 0, :], NR2[:, :, 1, :], dim=-1)

        nnorm1 = N1.norm(dim=-1)
        nnorm2 = N2.norm(dim=-1)
        n1 = N1 / nnorm1[:, :, None]
        n2 = N2 / nnorm2[:, :, None]

        num_p = self._point_pairs.shape[1]
        
        # Create extended tensor for two surfaces
        E10 = torch.zeros([num_g1, num_p, 2, 3], device=Y1.device)
        E10[:, :, 0] = y1[:, self._point_pairs[0]]
        E10[:, :, 1] = n1[:, self._point_pairs[0]]

        E20 = torch.zeros([num_g2, num_p, 2, 3], device=Y2.device)
        E20[:, :, 0] = y2[:, self._point_pairs[1]]
        E20[:, :, 1] = n2[:, self._point_pairs[1]]
        dy0 = E10[:, None, :, 0, :] - E20[None, :, :, 0, :]
        dn0 = E10[:, None, :, 1, :] - E20[None, :, :, 1, :]

        M0 = (E10[:, None, :, 1, :] * E20[None, :, :, 1, :]).sum(dim=-1)
        MM0 = (self._penalty_start_f - M0) / (self._penalty_start_f - self._penalty_end_f)
        MM0 = MM0.clamp(0, 1)
        f0 = MM0**3 * (6*MM0**2 - 15*MM0 + 10)

        D0 = (dn0 * dy0).sum(dim=-1) / 2
        g0 = torch.exp(D0 * self._penalty_factor_g) * self._penalty_distance_g
        
        L0 = dy0.norm(dim=-1)
        T0 = (self._penalty_threshold_h - L0) / (self._penalty_ratio_h * self._penalty_threshold_h)
        T0 = T0.clamp(0, 1)
        h0 = T0**3 * (6*T0**2 - 15*T0 + 10)

        penalty = g0 * f0 * h0 * weight0

        # Filter zero penalty pairs
        index_remain_total = torch.where(penalty.sum([0,1]) > 0)[0]

        if index_remain_total.shape[0] == 0:
            # No active contact pairs
            return torch.tensor([], dtype=torch.int64), torch.tensor([]), torch.tensor([[], []], dtype=torch.int64), torch.tensor([])

        pdU_indices_total = [] 
        pdU_values_total = []
        pdU_2_indices_total = []
        pdU_2_values_total = []

        index_now = 0
        batch_size = 10000
        while True:
            index_remain = index_remain_total[index_now:index_now+batch_size]
            if index_remain.shape[0] == 0:
                break

            point_pairs = self._point_pairs[:, index_remain]
            num_p = index_remain.shape[0]

            if index_remain.shape[0] > 0:
                print('  Contact pairs: ', index_remain.shape[0])

            # Filter all variables
            MM = MM0[:, :, index_remain]
            E1 = E10[:, index_remain]
            E2 = E20[:, index_remain]
            T = T0[:, :, index_remain]
            L = L0[:, :, index_remain]
            dy = dy0[:, :, index_remain]
            dn = dn0[:, :, index_remain]
            f = f0[:, :, index_remain]
            g = g0[:, :, index_remain]
            h = h0[:, :, index_remain]
            weight = weight0[:, :, index_remain]

            # Calculate derivatives for both surfaces
            # Surface 1 derivatives
            n1dN1 = torch.einsum('ij, ge->geij', torch.eye(3), 1/nnorm1) + \
                torch.einsum('gei, gej, ge->geij', n1, n1, -1/nnorm1)
            n1dN1_2 = torch.einsum('ij, gek, ge->geijk', torch.eye(3), n1, -1/nnorm1**2) + \
                torch.einsum('geik, gej, ge->geijk', n1dN1, n1, -1/nnorm1) + \
                torch.einsum('gei, gejk, ge->geijk', n1, n1dN1, -1/nnorm1) + \
                torch.einsum('gei, gej, gek, ge->geijk', n1, n1, n1, 1/nnorm1**2)
            
            y1dUe = self.surface_element1.shape_function_gaussian[0]
            
            epsilon = torch.zeros([3, 3, 3])
            epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
            epsilon[1, 0, 2] = epsilon[2, 1, 0] = epsilon[0, 2, 1] = -1

            N1dUe = torch.einsum('ijl, geja->geial', epsilon, 
                                torch.einsum('gei, ga->geia', NR1[:, :, 0], 
                                            self.surface_element1.shape_function_gaussian[1][:, 1]) - 
                                torch.einsum('gei, ga->geia', NR1[:, :, 1], 
                                            self.surface_element1.shape_function_gaussian[1][:, 0]))
            
            N1dUe_2 = torch.einsum('ipl, gab->gialbp', epsilon, 
                                torch.einsum('gb,ga->gab', self.surface_element1.shape_function_gaussian[1][:, 0], self.surface_element1.shape_function_gaussian[1][:, 1])-
                                torch.einsum('gb,ga->gab', self.surface_element1.shape_function_gaussian[1][:, 1], self.surface_element1.shape_function_gaussian[1][:, 0]))

            n1dUe = torch.einsum('geij, geial->gejal', n1dN1, N1dUe)
            n1dUe_2 = torch.einsum('geijk, geial, gekbp->gejalbp', n1dN1_2, N1dUe, N1dUe) + \
                    torch.einsum('geij, gialbp->gejalbp', n1dN1, N1dUe_2)

            e1dUe = torch.zeros([num_g1, num_e1, 2, 3, num_n1, 3])
            e1dUe[:, :, 1] = n1dUe
            e1dUe[:, :, 0, 0, :, 0] = y1dUe[:, None, :]
            e1dUe[:, :, 0, 1, :, 1] = y1dUe[:, None, :]
            e1dUe[:, :, 0, 2, :, 2] = y1dUe[:, None, :]

            e1dUe_2 = torch.zeros([num_g1, num_e1, 2, 3, num_n1, 3, num_n1, 3])
            e1dUe_2[:, :, 1] = n1dUe_2

            # Surface 2 derivatives
            n2dN2 = torch.einsum('ij, ge->geij', torch.eye(3), 1/nnorm2) + \
                torch.einsum('gei, gej, ge->geij', n2, n2, -1/nnorm2)
            n2dN2_2 = torch.einsum('ij, gek, ge->geijk', torch.eye(3), n2, -1/nnorm2**2) + \
                torch.einsum('geik, gej, ge->geijk', n2dN2, n2, -1/nnorm2) + \
                torch.einsum('gei, gejk, ge->geijk', n2, n2dN2, -1/nnorm2) + \
                torch.einsum('gei, gej, gek, ge->geijk', n2, n2, n2, 1/nnorm2**2)
            
            y2dUe = self.surface_element2.shape_function_gaussian[0]

            N2dUe = torch.einsum('ijl, geja->geial', epsilon, 
                                torch.einsum('gei, ga->geia', NR2[:, :, 0], 
                                            self.surface_element2.shape_function_gaussian[1][:, 1]) - 
                                torch.einsum('gei, ga->geia', NR2[:, :, 1], 
                                            self.surface_element2.shape_function_gaussian[1][:, 0]))
            N2dUe_2 = torch.einsum('ipl, gab->gialbp', epsilon, 
                                torch.einsum('gb,ga->gab', self.surface_element2.shape_function_gaussian[1][:, 0], self.surface_element2.shape_function_gaussian[1][:, 1])-
                                torch.einsum('gb,ga->gab', self.surface_element2.shape_function_gaussian[1][:, 1], self.surface_element2.shape_function_gaussian[1][:, 0]))


            n2dUe = torch.einsum('geij, geial->gejal', n2dN2, N2dUe)
            n2dUe_2 = torch.einsum('geijk, geial, gekbp->gejalbp', n2dN2_2, N2dUe, N2dUe) + \
                    torch.einsum('geij, gialbp->gejalbp', n2dN2, N2dUe_2)

            e2dUe = torch.zeros([num_g2, num_e2, 2, 3, num_n2, 3])
            e2dUe[:, :, 1] = n2dUe
            e2dUe[:, :, 0, 0, :, 0] = y2dUe[:, None, :]
            e2dUe[:, :, 0, 1, :, 1] = y2dUe[:, None, :]
            e2dUe[:, :, 0, 2, :, 2] = y2dUe[:, None, :]

            e2dUe_2 = torch.zeros([num_g2, num_e2, 2, 3, num_n2, 3, num_n2, 3])
            e2dUe_2[:, :, 1] = n2dUe_2

            # Calculate penalty derivatives (similar to self-contact but for two surfaces)
            # g = torch.exp(D * self._penalty_factor_g) * self._penalty_distance_g
            gdD = (self._penalty_factor_g) * g
            gdD_2 = (self._penalty_factor_g**2) * g

            gdE = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3])
            gdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dn)
            gdE[:, :, :, 1, 0, :] = -gdE[:, :, :, 0, 0, :]
            gdE[:, :, :, 0, 1, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dy)
            gdE[:, :, :, 1, 1, :] = -gdE[:, :, :, 0, 1, :]

            gdE_2 = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3, 2, 2, 3])
            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dn)
            gdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dy) + \
                    torch.einsum('gGp, ij->gGpij', gdD / 2, torch.eye(3))
            gdE_2[:, :, :, 0, 0, :, 0, 1, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 1, :] = tmp

            tmp = tmp.permute([0, 1, 2, 4, 3])
            gdE_2[:, :, :, 0, 1, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 1, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 1, 0, :] = tmp

            temp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dy, dy)
            gdE_2[:, :, :, 0, 1, :, 0, 1, :] = temp
            gdE_2[:, :, :, 1, 1, :, 0, 1, :] = -temp
            gdE_2[:, :, :, 0, 1, :, 1, 1, :] = -temp
            gdE_2[:, :, :, 1, 1, :, 1, 1, :] = temp

            fdM = -30*MM**2*(MM-1)**2 / (self._penalty_start_f-self._penalty_end_f)
            fdM_2 = 60*MM*(MM-1)*(2*MM-1) / (self._penalty_start_f-self._penalty_end_f)**2
            fdM[MM>=1] = 0 
            fdM[MM<=0] = 0
            fdM_2[MM>=1] = 0 
            fdM_2[MM<=0] = 0
            # M = (E[:, None, :, 0, 1, :] * E[None, :, :, 1, 1, :]).sum(dim=-1)

            fdE = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3])
            fdE[:, :, :, 0, 1, :] = torch.einsum('gGp, Gpi->gGpi', fdM, E2[:, :, 1, :])
            fdE[:, :, :, 1, 1, :] = torch.einsum('gGp, gpi->gGpi', fdM, E1[:, :, 1, :])

            fdE_2 = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3, 2, 2, 3])
            fdE_2[:, :, :, 0, 1, :, 0, 1, :] = torch.einsum('gGp, Gpi, Gpj->gGpij', fdM_2, E2[:, :, 1, :], E2[:, :, 1, :])
            fdE_2[:, :, :, 0, 1, :, 1, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
                                                torch.einsum('gGp, Gpi, gpj->gGpij', fdM_2, E2[:, :, 1, :], E1[:, :, 1, :])
            fdE_2[:, :, :, 1, 1, :, 1, 1, :] = torch.einsum('gGp, gpi, gpj->gGpij', fdM_2, E1[:, :, 1, :], E1[:, :, 1, :])
            fdE_2[:, :, :, 1, 1, :, 0, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
            torch.einsum('gGp, gpi, Gpj->gGpij', fdM_2, E1[:, :, 1, :], E2[:, :, 1, :])

            hdE = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3])

            # L = dy.norm(dim=-1)
            # T = (self._penalty_distance - L) / (0.5 * self._penalty_distance)
            # T = T.clamp(0, 1)
            # h = T**3 * (6*T**2 - 15*T + 10)
            Lddy = torch.einsum('gGpi, gGp->gGpi', dy, 1/L)
            Lddy_2 = torch.einsum('ij, gGp->gGpij', torch.eye(3), 1/L) + torch.einsum('gGpi, gGpj, gGp->gGpij', dy, Lddy, -1/L**2)
            hdL = -30*T**2*(T-1)**2 / (self._penalty_ratio_h * self._penalty_threshold_h)
            hdL_2 = 60*T*(T-1)*(2*T-1) / (self._penalty_ratio_h * self._penalty_threshold_h)**2
            hdL[T>=1] = 0
            hdL[T<=0] = 0
            hdL_2[T>=1] = 0
            hdL_2[T<=0] = 0
            hdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', hdL, Lddy)
            hdE[:, :, :, 1, 0, :] = -hdE[:, :, :, 0, 0, :]

            hdE_2 = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3, 2, 2, 3])
            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', hdL_2, Lddy, Lddy) + \
                    torch.einsum('gGp, gGpij->gGpij', hdL, Lddy_2)
            hdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
            hdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
            hdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
            hdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

            pdE = torch.einsum('gGpmxi, gGp, gGp->gGpmxi', fdE, g, h) + \
                torch.einsum('gGp, gGpmxi, gGp->gGpmxi', f, gdE, h) + \
                torch.einsum('gGp, gGp, gGpmxi->gGpmxi', f, g, hdE)
            
            pdE = pdE * weight[:, :, :, None, None, None]

            pdE_2 = torch.einsum('gGpmxinyj, gGp, gGp->gGpmxinyj', fdE_2, g, h) + \
                    torch.einsum('gGpmxi, gGpnyj, gGp->gGpmxinyj', fdE, gdE, h) + \
                    torch.einsum('gGpmxi, gGp, gGpnyj->gGpmxinyj', fdE, g, hdE) + \
                    \
                    torch.einsum('gGpnyj, gGpmxi, gGp->gGpmxinyj', fdE, gdE, h) +\
                    torch.einsum('gGp, gGpmxinyj, gGp->gGpmxinyj', f, gdE_2, h) +\
                    torch.einsum('gGp, gGpmxi, gGpnyj->gGpmxinyj', f, gdE, hdE) +\
                    \
                    torch.einsum('gGpnyj, gGp, gGpmxi->gGpmxinyj', fdE, g, hdE)+\
                    torch.einsum('gGp, gGpnyj, gGpmxi->gGpmxinyj', f, gdE, hdE)+\
                    torch.einsum('gGp, gGp, gGpmxinyj->gGpmxinyj', f, g, hdE_2)
            
            pdE_2 = pdE_2 * weight[:, :, :, None, None, None, None, None, None]

            # Calculate force contributions
            pdEsum0 = pdE.sum(0)
            pdEsum1 = pdE.sum(1)

            pdUe_values1 = torch.einsum('gpxi, gpxial->pal', pdEsum1[:, :, 0], e1dUe[:, point_pairs[0]])
            pdUe_values2 = torch.einsum('Gpxi, Gpxial->pal', pdEsum0[:, :, 1], e2dUe[:, point_pairs[1]])

            pdU_values = torch.cat([pdUe_values1.flatten(), pdUe_values2.flatten()], dim=0)

            tri_ind = point_pairs.cpu()
            index_start1 = self._assembly.RGC_list_indexStart[instance1._RGC_index]
            index_start2 = self._assembly.RGC_list_indexStart[instance2._RGC_index]

            pdU_indices1 = self.surface_element1._elems[tri_ind[0]].to(torch.int64)
            pdU_indices1 = torch.stack([pdU_indices1*3, pdU_indices1*3+1, pdU_indices1*3+2], dim=-1) + index_start1
            
            pdU_indices2 = self.surface_element2._elems[tri_ind[1]].to(torch.int64)
            pdU_indices2 = torch.stack([pdU_indices2*3, pdU_indices2*3+1, pdU_indices2*3+2], dim=-1) + index_start2
            
            pdU_indices = torch.cat([pdU_indices1.flatten(), pdU_indices2.flatten()], dim=0).to(self._assembly.device)

            if if_onlyforce:
                
                return pdU_values, pdU_indices, torch.tensor([[], []], dtype=torch.int64), torch.tensor([])


            # For stiffness matrix, return simplified version (full implementation would be very complex)
            # Return empty stiffness for now
            pdUe_2_values00 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(1)[:, :, 0, :, :, 0], e1dUe[:, point_pairs[0]], e1dUe[:, point_pairs[0]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum1[:, :, 0], e1dUe_2[:, point_pairs[0]])
            
            pdUe_2_values01 = torch.einsum('gGpxiyj, gpxial, GpyjbL->palbL', pdE_2[:, :, :, 0, :, :, 1], e1dUe[:, point_pairs[0]], e2dUe[:, point_pairs[1]])

            pdUe_2_values10 = torch.einsum('gGpxiyj, Gpxial, gpyjbL->palbL', pdE_2[:, :, :, 1, :, :, 0], e2dUe[:, point_pairs[1]], e1dUe[:, point_pairs[0]])
            
            pdUe_2_values11 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(0)[:, :, 1, :, :, 1], e2dUe[:, point_pairs[1]], e2dUe[:, point_pairs[1]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum0[:, :, 1], e2dUe_2[:, point_pairs[1]])

            pdU_2_values = torch.cat([pdUe_2_values00.flatten(), pdUe_2_values01.flatten(), pdUe_2_values10.flatten().flatten(), pdUe_2_values11.flatten()], dim=0)

            # Build indices


            

            pdU_2_indices00 = torch.stack([
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, num_n1, 1, 1, 1]).repeat([1, 1, 3, num_n1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n1, 1, num_n1, 3]),
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n1, 1]).repeat([1, num_n1, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n1, 3, num_n1, 1]),
            ]).reshape([4, -1])
            pdU_2_indices00 = torch.stack([pdU_2_indices00[0]*3+pdU_2_indices00[1], pdU_2_indices00[2]*3+pdU_2_indices00[3]], dim=0)
            pdU_2_indices00[0] += index_start1
            pdU_2_indices00[1] += index_start1

            pdU_2_indices01 = torch.stack([
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, num_n1, 1, 1, 1]).repeat([1, 1, 3, num_n2, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n1, 1, num_n2, 3]),
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n2, 1]).repeat([1, num_n1, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n1, 3, num_n2, 1]),
            ]).reshape([4, -1])
            pdU_2_indices01 = torch.stack([pdU_2_indices01[0]*3+pdU_2_indices01[1], pdU_2_indices01[2]*3+pdU_2_indices01[3]], dim=0)
            pdU_2_indices01[0] += index_start1
            pdU_2_indices01[1] += index_start2

            pdU_2_indices10 = torch.stack([
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, num_n2, 1, 1, 1]).repeat([1, 1, 3, num_n1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n2, 1, num_n1, 3]),
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n1, 1]).repeat([1, num_n2, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n2, 3, num_n1, 1]),
            ]).reshape([4, -1])
            pdU_2_indices10 = torch.stack([pdU_2_indices10[0]*3+pdU_2_indices10[1], pdU_2_indices10[2]*3+pdU_2_indices10[3]], dim=0)
            pdU_2_indices10[0] += index_start2
            pdU_2_indices10[1] += index_start1

            pdU_2_indices11 = torch.stack([
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, num_n2, 1, 1, 1]).repeat([1, 1, 3, num_n2, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n2, 1, num_n2, 3]),
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n2, 1]).repeat([1, num_n2, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n2, 3, num_n2, 1]),
            ]).reshape([4, -1])
            pdU_2_indices11 = torch.stack([pdU_2_indices11[0]*3+pdU_2_indices11[1], pdU_2_indices11[2]*3+pdU_2_indices11[3]], dim=0)
            pdU_2_indices11[0] += index_start2
            pdU_2_indices11[1] += index_start2
            

            pdU_2_indices = torch.cat([pdU_2_indices00, pdU_2_indices01, pdU_2_indices10, pdU_2_indices11], dim=1).to(self._assembly.device)

            index_now += batch_size

            pdU_indices_total.append(pdU_indices)
            pdU_values_total.append(pdU_values)
            pdU_2_indices_total.append(pdU_2_indices)
            pdU_2_values_total.append(pdU_2_values)

        pdU_indices = torch.cat(pdU_indices_total, dim=0)
        pdU_values = torch.cat(pdU_values_total, dim=0)
        pdU_2_indices = torch.cat(pdU_2_indices_total, dim=1)
        pdU_2_values = torch.cat(pdU_2_values_total, dim=0)

        return pdU_indices.flatten(), -pdU_values.flatten(), pdU_2_indices, -pdU_2_values

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        instance1 = self._assembly.get_instance(self.instance_name1)
        instance2 = self._assembly.get_instance(self.instance_name2)
        RGC_remain_index[instance1._RGC_index][self.surface_element1._elems.flatten().unique().cpu()] = True
        RGC_remain_index[instance2._RGC_index][self.surface_element2._elems.flatten().unique().cpu()] = True
        return RGC_remain_index

