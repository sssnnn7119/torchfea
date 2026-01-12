from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pypardiso

if TYPE_CHECKING:
    from ... import Assembly
import time
import torch
from .. import _linear_solver
from ..basesolver import BaseSolver
from .result import StaticResult

class StaticImplicitSolver(BaseSolver):

    def __init__(self, maximum_iteration: int = 10000, tol_error: float = 1e-5) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """

        self.maximum_iteration: int = maximum_iteration
        """
        the allowed maximum number of iterations for the solver.
        """

        self._iter_now: int = 0
        """
        The iteration of the FEA step
        """

        self._maximum_step_length = 1e10
        """
        The allowable maximum step length for each step.
        """

        self.tol_error: float = tol_error
        """
        The tolerance error for the solver.
        """

        self.__low_alpha_count = 0

    def solve(self, GC0: torch.Tensor = None, *args, **kwargs) -> bool:
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            bool: True if the solution converged, False otherwise.
        """
        # initialize the RGC
        t0 = time.time()
        # start the iteration
        if GC0 is None:
            GC0 = self.assembly.GC
        result = self._solve_iteration(GC=GC0, tol_error=self.tol_error)

        if type(result) == bool:
            return result
        
        self.GC = result
        self.assembly.GC = self.GC
        self.assembly.RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(self.GC))
        t2 = time.time()

        # print the information
        print('total_iter:%d, total_time:%.2f' % (self._iter_now, t2 - t0))
        R = self.get_stiffness_matrix(GC_now=self.GC)[0]
        print('max_error:%.4e' % (R.abs().max()))
        print('---' * 8, 'FEA Finished', '---' * 8, '\n')

        # build the result object
        return StaticResult(GC=self.GC, load_params=self.assembly.get_load_parameters())
   
    def get_jacobian(self, result: StaticResult) -> torch.Tensor:
        """
        Calculate the Jacobian matrix for the current configuration.

        Args:
            result (StaticResult): The result object containing the current state.
        Returns:
            torch.Tensor: The Jacobian matrix.
        """
        # set the load parameters to the assembly
        self.assembly.set_load_parameters(result.load_params)
        
        # get the current load parameters as a single tensor
        total_params_list = []
        for load in self.assembly._loads.values():
            total_params_list.append(load._parameters.flatten().detach().clone())
        total_params = torch.cat(total_params_list, dim=0)

        # define the closure function to compute R
        def closure_R(total_params: torch.Tensor):

            index_now = 0
            for load in self.assembly._loads.values():
                param_len = load._parameters.numel()
                load._parameters = total_params[index_now:index_now+param_len].reshape(load._parameters.shape)
                index_now += param_len
            R = self.assembly.assemble_force(GC=result.GC.to(self.assembly.device))

            # remove the leaf parameters
            for load in self.assembly._loads.values():
                load._parameters = load._parameters.detach()

            return R
        

        from torch.autograd.functional import jvp
        
        num_params = total_params.numel()
        if num_params > 0:
            Rdp_cols = []

            for i in range(num_params):
                # Compute Jacobian-Vector Product for the i-th parameter
                # This computes the i-th column of the Jacobian
                basis_vector_now = torch.zeros_like(total_params)
                basis_vector_now[i] = 1.0
                _, col = jvp(closure_R, total_params, v=basis_vector_now, create_graph=False)
                Rdp_cols.append(col)
            
            Rdp = torch.stack(Rdp_cols, dim=1)
        else:
            # Handle case with no parameters
            R_dummy = closure_R(total_params)
            Rdp = torch.zeros((R_dummy.shape[0], 0), device=total_params.device, dtype=total_params.dtype)


        K_indices, K_values = self.assembly.assemble_Stiffness_Matrix(GC=result.GC)[1:]
        import scipy.sparse as sp
        K_sp = sp.coo_matrix(
            (K_values.detach().cpu().numpy(), (K_indices[0].cpu().numpy(),
                                        K_indices[1].cpu().numpy()))).tocsr()
        jacobian = -pypardiso.spsolve(K_sp, Rdp.detach().cpu().numpy())
        jacobian = jacobian.reshape(-1, num_params) # Shape: (params, dofs)

        jacobian_output = {}
        index_now = 0
        for load_name, load in self.assembly._loads.items():
            param_len = load._parameters.numel()
            jacobian_output[load_name] = torch.from_numpy(jacobian[:, index_now:index_now+param_len]).to(result.GC.dtype).to(result.GC.device)
            index_now += param_len

        return jacobian_output
    
    def get_total_energy(self, GC_now: torch.Tensor) -> float:
        
        potential_energy = self.assembly._total_Potential_Energy(GC=GC_now)
        return potential_energy
    
    def get_stiffness_matrix(self, GC_now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the stiffness matrix for the current configuration.

        Args:
            GC_now (torch.Tensor): Current generalized coordinates.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Indices and values of the stiffness matrix.
        """
        R, K_indices, K_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)
        return R,K_indices, K_values

    # region solve iteration

    def _line_search(self,
                     GC0: torch.Tensor,
                     dGC: torch.Tensor,
                     R: torch.Tensor,
                     energy0: float, *args, **kwargs):
        # line search
        alpha = 1.0
        beta = float('inf')
        c1 = 0.3
        c2 = 0.4
        dGC0 = dGC.clone()
        deltaE = (dGC * R).sum()

        if deltaE > 0:
            dGC = -dGC
            deltaE = -deltaE
            print('the newton dirction is not the decrease direction')

        if torch.isnan(dGC).sum() > 0 or torch.isinf(dGC).sum() > 0:
            raise ValueError('dGC has nan or inf')
            dGC = -R
            deltaE = (dGC * R).sum()

        # if abs(deltaE / energy0) < tol_error:
        #     return 1, GC0

        loopc2 = 0
        while True:
            GCnew = GC0 + alpha * dGC
            # GCnew.requires_grad_()
            energy_new = self.get_total_energy(
                GC_now=GCnew)

            if torch.isnan(energy_new) or torch.isinf(
                    energy_new) or \
                energy_new > energy0 + c1 * deltaE * alpha or \
                (alpha * dGC).abs().max() > self._maximum_step_length:
                alpha = 0.5 * alpha
                if alpha < 1e-12:
                    alpha = 0.0
                    GCnew = GC0.clone()
                    energy_new = energy0
                    break
            else:
                # Rnew = -torch.autograd.grad(energy_new, GCnew)[0]
                # if torch.dot(Rnew, dGC) > c2 * deltaE:
                #     beta = alpha
                #     alpha = 0.6 * (alpha + beta)
                # elif torch.dot(Rnew, dGC) < -c2 * deltaE:
                #     beta = alpha
                #     alpha = 0.4 * (alpha + beta)
                # else:
                break
            loopc2 += 1
            if loopc2 > 20:
                c2 = 1000000000000000

        # if abs(alpha) < 1e-6:
        #     # gradient direction line search
        #     alpha = 1
        #     dGC = R
        #     while True:
        #         GCnew = GC0 + alpha * dGC
        #         energy_new = self.get_total_energy(
        #             RGC=self.assembly._GC2RGC(GCnew))
        #         if energy_new < energy0:
        #             # pressure *= 1.2
        #             # pressure = min(pressure0, pressure)
        #             break
        #         alpha *= 0.8
        #         if abs(alpha) < 1e-10:
        #             alpha = 0.0
        #             GCnew = GC0.clone()
        #             energy_new = energy0
        #             break

        # if abs(alpha) < 1e-3:
        #     alpha = 1
        #     GCnew = GC0 + alpha * dGC0
        return alpha, GCnew.detach(), energy_new.detach()

    def _solve_iteration(self,
                         GC: torch.Tensor,
                         tol_error: float):

        # iteration now
        self._iter_now = 0

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self.get_total_energy(GC_now=GC)
        ]

        dGC = torch.zeros_like(GC)

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:

            if self._iter_now > self.maximum_iteration:
                print('maximum iteration reached')
                return False

            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R, K_indices, K_values = self.get_stiffness_matrix(GC_now=GC)

            self._iter_now += 1

            # evaluate the newton direction
            t2 = time.time()
            dGC = self._solve_linear_equation(K_indices=K_indices,
                                              K_values=K_values,
                                              R=-R,
                                              iter_now=self._iter_now,
                                              alpha0=alpha,
                                              tol_error=tol_error,
                                              dGC0=dGC).flatten()




            # line search
            t3 = time.time()
            alpha, GCnew, energynew = self._line_search(
                    GC, dGC, R, energy[-1])

            if alpha==0 and R.abs().max() > tol_error:
                return False
            if alpha==0:
                break

            # if convergence has difficulty, reduce the load percentage
            if alpha < 0.01:
                low_alpha += 1
            else:
                low_alpha -= 5
                if low_alpha < 0:
                    low_alpha = 0

            if low_alpha > 10:
                return False

            # update the GC
            GC = GCnew

            # update the RGC
            RGC = self.assembly._GC2RGC(GC)

            # self.show_surface(nodes=self.nodes+RGC[0])

            # update the energy
            energynew = self.get_total_energy(
                GC_now=GC)
            energy.append(energynew)

            t4 = time.time()

            # return the index to the first line
            if self._iter_now > 1:
                print('\033[1A', end='')
                print('\033[1A', end='')
                print('\033[K', end='')

            print(  "{:^8}".format("iter") + \
                    "{:^8}".format("alpha") + \
                    "{:^8}".format("total") + \
                    "{:^15}".format("energy") + \
                    "{:^15}".format("delta_energy") + \
                    "{:^15}".format("error") + \
                    "{:^10}".format("Ktime") + \
                    "{:^10}".format("linear") + \
                    "{:^10}".format("search") + \
                    "{:^10}".format("step"))

            print(  "{:^8}".format(self._iter_now) + \
                    "{:^8.2f}".format(alpha) + \
                    "{:^8.2f}".format(t4 - t00) + \
                    "{:^15.4e}".format(energy[-1]) + \
                    "{:^15.4e}".format(energy[-1] - energy[-2]) + \
                    "{:^15.4e}".format(R.abs().max()) + \
                    "{:^10.2f}".format(t2 - t1) + \
                    "{:^10.2f}".format(t3 - t2) + \
                    "{:^10.2f}".format(t4 - t3) + \
                    "{:^10.2f}".format(t4 - t1))
            
            if (dGC.abs().max() < tol_error and R.abs().max() < tol_error) or R.abs().max() < 1e-6:
                break

            # if len(energy)>2 and abs((energy[-1]-energy[-2])/energy[-1])<1e-7:
            #     break
        return GC

    def _solve_linear_equation(self,
                               K_indices: torch.Tensor,
                               K_values: torch.Tensor,
                               R: torch.Tensor,
                               iter_now: int = 0,
                               alpha0: float = None,
                               dGC0: torch.Tensor = None,
                               tol_error=1e-8):
        if dGC0 is None:
            dGC0 = torch.zeros_like(R)

        if alpha0 is None:
            alpha0 = 1e-10

        # result = torch.sparse.spsolve(torch.sparse_coo_tensor(K_indices, K_values, [R.shape[0], R.shape[0]]).to_sparse_csr(), R)

        # precondition for the linear equation
        index = torch.where(K_indices[0] == K_indices[1])[0]
        diag = torch.zeros_like(R).scatter_add(0, K_indices[0, index],
                                               K_values[index]).abs().sqrt()
        diag[diag==0] = 1.0  # Avoid division by zero
        K_values_preconditioned = K_values / diag[K_indices[0]]
        K_values_preconditioned = K_values_preconditioned / diag[K_indices[1]]
        R_preconditioned = R / diag
        x0 = dGC0 * diag

        # record the number of low alpha
        if alpha0 < 1e-1:
            self.__low_alpha_count += 1
        else:
            self.__low_alpha_count = 0

        if self.__low_alpha_count > 5 or R_preconditioned.abs().max() < 1e-3 or K_values_preconditioned.device.type == 'cpu':
            dx = _linear_solver.pypardiso_solver(A_indices=K_indices,
                                                 A_values=K_values_preconditioned,
                                                 b=R_preconditioned)
            self.__low_alpha_count = 0
        else:
            if self.__low_alpha_count > 0:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=3000)
            else:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=1200)
        result = dx.to(R.dtype) / diag
        return result

    # endregion

