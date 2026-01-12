from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Assembly
import time
import torch
from .. import _linear_solver
from ..basesolver import BaseSolver

class DynamicImplicitSolver(BaseSolver):

    def __init__(self, maximum_iteration: int = 10000, deltaT: float = 1e-2, time_end: float = 1.0, tol_error: float = 1e-5) -> None:
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
        
        self._GC_list: list[torch.Tensor] = None
        """ The generalized coordinates of the nodes. """

        self._GV_list: list[torch.Tensor] = None
        """ The velocity of the nodes. """

        self._time_list: list[float] = []
        """ The time of each step. """

        self._deltaT : float = deltaT
        """ The time increment of each step. """

        self._time_end : float = time_end
        """ The end time of the simulation. """

        self._gamma : float = 0.5
        """ The Newmark gamma parameter. """

        self._beta : float = 0.25
        """ The Newmark beta parameter. """

    def initialize(self, assembly: Assembly):
        """
        Initialize the solver with the assembly and initial conditions.
        """
        super().initialize(assembly=assembly)
        self.assembly.initialize_dynamic()

        self._GC_list = []
        self._GV_list = []
        self._GA_list = []
        self._time_list = []
        self._deltaT_list = []

    def set_deltaT(self, deltaT: float):
        """
        Update the time increment for the next step.

        Args:
            deltaT (float): The time increment for the next step.
        """
        self._deltaT = deltaT

    def get_total_energy(self, GC_now: torch.Tensor, GC0: torch.Tensor, GV0: torch.Tensor, GA0: torch.Tensor, deltaT: float = None) -> float:

        if deltaT is None:
            deltaT = self._deltaT

        potential_energy = self.assembly._total_Potential_Energy(GC=GC_now)

        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC_now)
        GV_now = self.get_next_velocity(GC_now=GC_now, GC0=GC0, GV0=GV0, GA0=GA0, deltaT=deltaT)[0]
        kinetic_energy_all = mass_values * GV_now[mass_indices[0]] * GV_now[mass_indices[1]] / 2
        kinetic_energy = kinetic_energy_all.sum()

        return potential_energy + kinetic_energy
    
    def get_incremental_energy(self, GC_now: torch.Tensor, GC_pre: torch.Tensor, deltaT: float = None) -> float:

        if deltaT is None:
            deltaT = self._deltaT

        potential_energy = self.assembly._total_Potential_Energy(GC=GC_now)

        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC_now)
        GC_diff = GC_now - GC_pre
        kinetic_energy_all = mass_values * GC_diff[mass_indices[0]] * GC_diff[mass_indices[1]] / (2 * self._beta * deltaT ** 2)
        kinetic_energy = kinetic_energy_all.sum()

        return potential_energy + kinetic_energy

    def get_incremental_stiffness_matrix(self, GC_now: torch.Tensor, GC_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the stiffness matrix for the current configuration.

        Args:
            GC_now (torch.Tensor): Current generalized coordinates.
            GC0 (torch.Tensor): Previous generalized coordinates.
            GV0 (torch.Tensor): Previous velocities.
            GA0 (torch.Tensor): Previous accelerations.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Residual force, Indices and values of the stiffness matrix.
        """
        # assemble the internal force and stiffness matrix
        Rv, Kv_indices, Kv_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)
        
        # assemble the mass matrix
        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC_now)
        GC_diff = GC_now - GC_pre
        Ri_values = mass_values * GC_diff[mass_indices[0]] / (self._beta * self._deltaT ** 2)
        Ri_indices = mass_indices[1]
        Ri = torch.zeros_like(GC_now).scatter_add_(0, Ri_indices, Ri_values)
        Ki_values = mass_values / (self._beta * self._deltaT ** 2)
        Ki_indices = mass_indices

        K_indices = torch.cat([Kv_indices, Ki_indices], dim=1)
        K_values = torch.cat([Kv_values, Ki_values], dim=0)
        R = Rv + Ri

        return R, K_indices, K_values

    def solve(self, GC0: torch.Tensor = None, GV0: torch.Tensor = None, *args, **kwargs) -> bool:
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

        if GC0 is None:
            GC0 = self.assembly.GC.clone()

        if GV0 is None:
            GV0 = torch.zeros_like(self.assembly.GC)

        self._GC_list = [GC0]
        self._GV_list = [GV0]
        self._GA_list = [self.get_current_acceleration(GC0=GC0, GV0=GV0)]
        self._time_list = [0.0]
        E_history = [self.get_total_energy(GC_now=GC0, GC0=GC0, GV0=GV0, GA0=self._GA_list[-1])]
        
        # start the iteration
        iteration = 0
        while True:
            print('---' * 8, 'FEA Step %d' % (iteration + 1), '---' * 8)
            print('time:%.8f, deltaT:%.8f' % (self._time_list[-1], self._deltaT))
            t0 = time.time()

            GC_pre = self._GC_list[-1] + self._deltaT * self._GV_list[-1] + (self._deltaT ** 2) * self._GA_list[-1] * (1 - 2 * self._beta) / 2

            GC_now = self._solve_iteration(GC0=self._GC_list[-1],
                                            GC_pre=GC_pre,
                                            deltaT=self._deltaT,
                                            tol_error=self.tol_error)
            t2 = time.time()

            # print the information
            print('total_iter:%d, total_time:%.2f' % (iteration, t2 - t0))
            E = self.get_total_energy(GC_now=GC_now, GC0=self._GC_list[-1], GV0=self._GV_list[-1], GA0=self._GA_list[-1])

            # energy_diff = (E - E_history[-1]).abs() / abs(E_history[-1])
            # if energy_diff > 1e-2:
            #     print('energy increase too much, reduce the time step')
            #     self._deltaT = self._deltaT / 2
            #     print('new deltaT:%.8f' % self._deltaT)
            #     print('---' * 8, 'FEA Continued', '---' * 8, '\n')
            #     continue
            # elif energy_diff < 1e-3:
            #     self._deltaT = self._deltaT * 1.2

            print('max_error:%.4e' % (((E - E_history[-1]) / E_history[-1]).abs()))
            print('---' * 8, 'FEA Continued', '---' * 8, '\n')

            # update the results
            GVnew, GAnew = self.get_next_velocity(GC_now=GC_now, GC0=self._GC_list[-1], GV0=self._GV_list[-1], GA0=self._GA_list[-1], deltaT=self._deltaT)
            E_history.append(E)
            self._GC_list.append(GC_now)
            self._GV_list.append(GVnew)
            self._GA_list.append(GAnew)
            self._time_list.append(self._time_list[-1] + self._deltaT)

            iteration += 1
            if self._time_list[-1] >= self._time_end:
                break
        
        self.GC=self._GC_list[-1]
        self.assembly.RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(self.assembly.GC))
        t2 = time.time()

        # print the information
        print('total_time:%.2f' % (t2 - t0))
        R = self.assembly.assemble_Stiffness_Matrix(RGC=self.assembly.RGC)[0]
        print('max_error:%.4e' % (R.abs().max()))
        print('---' * 8, 'FEA Finished', '---' * 8, '\n')

        return True 

    def get_next_velocity(self, GC_now: torch.Tensor, GC0: torch.Tensor, GV0: torch.Tensor, GA0: torch.Tensor, deltaT: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the velocity and acceleration based on the Newmark-beta method.

        Args:
            GC_now (torch.Tensor): Current generalized coordinates.
            GC0 (torch.Tensor): Previous generalized coordinates.
            GV0 (torch.Tensor): Previous velocities.
            GA0 (torch.Tensor): Previous accelerations.
            deltaT (float): Time increment.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Current velocities and accelerations.
        """
        # Newmark-beta method for velocity and acceleration update

        GV_now = self._gamma / (self._beta * deltaT) * (GC_now - GC0) + (1 - self._gamma / self._beta) * GV0 + deltaT * (1 - self._gamma / (2 * self._beta)) * GA0
        GA_now = 1 / (self._beta * deltaT ** 2) * (GC_now - GC0) - 1 / (self._beta * deltaT) * GV0 - (1 / (2 * self._beta) - 1) * GA0

        return GV_now, GA_now

    def get_current_acceleration(self, GC0: torch.Tensor, GV0: torch.Tensor) -> torch.Tensor:
        """
        Calculate the current acceleration based on the newton's law.

        Args:
            GC0 (torch.Tensor): Previous generalized coordinates.
            GV0 (torch.Tensor): Previous velocities.
            GA0 (torch.Tensor): Previous accelerations.

        Returns:
            torch.Tensor: Current accelerations.
        """
        Rv = self.assembly.assemble_Stiffness_Matrix(GC=GC0)[0]

        # Newmark initial acceleration
        # M * GA0 = F_ext(GC0) - F_int(GC0)
        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC0)
        GA0 = _linear_solver.pypardiso_solver(mass_indices, mass_values, Rv)
        return GA0
    
    # region solve iteration

    def _line_search(self,
                     GC_now: torch.Tensor,
                     GC_pre: torch.Tensor,
                     dGC: torch.Tensor,
                     R: torch.Tensor,
                     energy0: float, deltaT: float, *args, **kwargs):
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
            GCnew = GC_now + alpha * dGC
            # GCnew.requires_grad_()
            energy_new = self.get_incremental_energy(GC_now=GCnew, GC_pre=GC_pre, deltaT=deltaT)

            if torch.isnan(energy_new) or torch.isinf(
                    energy_new) or \
                energy_new > energy0 + c1 * deltaE * alpha or \
                (alpha * dGC).abs().max() > self._maximum_step_length:
                alpha = 0.5 * alpha
                if alpha < 1e-12:
                    alpha = 0.0
                    GCnew = GC_now.clone()
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
        #         energy_new = self.assembly._total_Potential_Energy(
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
        return alpha, GCnew.detach(), energy_new

    def _solve_iteration(self,
                         GC0: torch.Tensor,
                         GC_pre: torch.Tensor,
                         deltaT: float,
                         tol_error: float):

        GC_now = GC0.clone()

        # iteration now
        self._iter_now = 0

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self.get_incremental_energy(GC_now=GC_now, GC_pre=GC_pre, deltaT=deltaT)
        ]

        dGC = torch.zeros_like(GC0)

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:

            if self._iter_now > self.maximum_iteration:
                print('maximum iteration reached')
                self.assembly.GC = GC_now
                return False

            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R, K_indices, K_values = self.get_incremental_stiffness_matrix(
                GC_now=GC_now, GC_pre=GC_pre)

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
                    GC_now=GC_now, GC_pre=GC_pre, dGC=dGC, R=R, energy0=energy[-1], deltaT=deltaT)

            if alpha==0 and R.abs().max() > tol_error:
                self.assembly.GC = GC_now
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
                if R.abs().max() < 1e-3:
                    print('low alpha, but convergence achieved')
                    self.assembly.GC = GC_now
                    break
                return False



            # self.show_surface(nodes=self.nodes+RGC[0])

            # update the energy
            energynew = self.get_incremental_energy(GC_now=GCnew, GC_pre=GC_pre, deltaT=deltaT)
            energy.append(energynew)

            # update the GC
            GC_now = GCnew

            t4 = time.time()

            # return the index to the first line
            if self._iter_now > 1:
                print('\033[1A', end='')
                print('\033[1A', end='')
                print('\033[K', end='')

            print(  "{:^8}".format("iter") + \
                    "{:^8}".format("alpha") + \
                    "{:^15}".format("total") + \
                    "{:^15}".format("energy") + \
                    "{:^15}".format("error") + \
                    "{:^15}".format("assemble") + \
                    "{:^15}".format("linearEQ") + \
                    "{:^15}".format("line search") + \
                    "{:^15}".format("step"))

            print(  "{:^8}".format(self._iter_now) + \
                    "{:^8.2f}".format(alpha) + \
                    "{:^15.2f}".format(t4 - t00) + \
                    "{:^15.4e}".format(energy[-1]) + \
                    "{:^15.4e}".format(R.abs().max()) + \
                    "{:^15.2f}".format(t2 - t1) + \
                    "{:^15.2f}".format(t3 - t2) + \
                    "{:^15.2f}".format(t4 - t3) + \
                    "{:^15.2f}".format(t4 - t1))
            
            if dGC.abs().max() < tol_error and R.abs().max() < tol_error:
                break

        return GC_now

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

        if self.__low_alpha_count > 3 or R_preconditioned.abs().max() < 1e-3 or K_values_preconditioned.device.type == 'cpu':
            dx = _linear_solver.pypardiso_solver(K_indices,
                                                 K_values_preconditioned,
                                                 R_preconditioned)
            self.__low_alpha_count = 0
        else:
            if iter_now % 20 == 0 or self.__low_alpha_count > 0:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=6000)
            else:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=1500)
        result = dx.to(R.dtype) / diag
        return result

    # endregion

