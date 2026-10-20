from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ... import Assembly
import time
import torch
from .. import _linear_solver
from ..basesolver import BaseSolver
from .result import StaticResult

import logging
logger = logging.getLogger(__name__)

class StaticImplicitSolver(BaseSolver):

    _serialized_attributes_exclude = ['_StaticImplicitSolver__low_alpha_count']

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


    def solve(self, GC0: torch.Tensor = None, need_jacobian: bool = False, *args, **kwargs) -> StaticResult:
        """
        Solves the finite element analysis problem and returns a StaticResult object.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            StaticResult: The result object containing GC, jacobian (optional), and convergence status.
        """
        # initialize the RGC
        t0 = time.time()
        # start the iteration
        if GC0 is None:
            GC0 = self.assembly._GC


        logger.info('---' * 8 + ' FEA Start ' + '---' * 8)

        with torch.no_grad():
            solve_output = self._solve_iteration(GC=GC0, tol_error=self.tol_error)

        GC_final, total_time_iter, time_items, converged = solve_output
        self.assembly._GC = GC_final
        self.assembly._RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(GC_final))
        t2 = time.time()

        # print the information
        logger.info('total_iter:%d, total_time:%.2f' % (self._iter_now, t2 - t0))
        R = self.get_stiffness_matrix(GC_now=GC_final)[0]
        logger.info('max_error:%.4e' % (R.abs().max()))
        logger.info('---' * 8 + ' FEA Finished ' + '---' * 8)

        # build the result object
        fe_result = StaticResult(
            GC=GC_final,
            work_conditions=self.assembly.get_work_conditions(),
            total_time=total_time_iter,
            time_items=time_items,
            converged=converged,
        )

        if need_jacobian:
            jacobian = self.get_jacobian(result=fe_result)
            fe_result.jacobian = jacobian

        return fe_result
   
    def get_jacobian(self, result: StaticResult, load_names: list[str] = None) -> torch.Tensor:
        """
        Calculate the Jacobian matrix for the current configuration.

        Args:
            result (StaticResult): The result object containing the current state.
            load_names (list[str], optional): The names of the loads for which to compute the Jacobian. If None, all loads are considered.
        Returns:
            torch.Tensor: The Jacobian matrix.
                shape: (num_dofs, num_load_params).
        """
        # set the load parameters to the assembly
        self.assembly.set_work_conditions(result.work_conditions)

        # if load_names is None, compute for all loads
        if load_names is None:
            load_names = list(self.assembly._loads.keys())
        
        # get the current load parameters as a single tensor
        total_params_list = []
        for load_name in load_names:
            load = self.assembly._loads[load_name]
            total_params_list.append(load._parameters.flatten())
        total_params = torch.cat(total_params_list, dim=0)

        # define the closure function to compute R
        def closure_R(total_params: torch.Tensor):

            index_now = 0
            for load_name in load_names:
                load = self.assembly._loads[load_name]
                param_len = load._parameters.numel()
                load._parameters = total_params[index_now:index_now+param_len].reshape(load._parameters.shape)
                index_now += param_len
            R = self.assembly.assemble_force(GC=result.GC.to(self.assembly.device))

            # remove the leaf parameters
            for load_name in load_names:
                load = self.assembly._loads[load_name]
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


        if result.if_factorized is False:
            result.factorize_stiffness_matrix(assembly=self.assembly)
        
        jacobian = -result.K_solver.solve(result.K_sp, Rdp.cpu().numpy())
        jacobian = jacobian.reshape(-1, num_params) # Shape: (num_dofs, num_load_params)

        jacobian_output = {}
        index_now = 0
        for load_name in load_names:
            load = self.assembly._loads[load_name]
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


    # region sensitivity analysis
    def get_sensitivity(
        self,
        fe_result: StaticResult,
        design_vars: torch.Tensor,
        apply_func: Callable[[Assembly, torch.Tensor], None],
        compute_objective_func: Callable[[StaticResult, Assembly], torch.Tensor]
    ) -> torch.Tensor:
        """
        Core functional implementation of the adjoint sensitivity analysis.

        This function calculates the gradient of an objective function with respect to design variables,
        constrained by the finite element equilibrium equations, using the discrete adjoint method.

        Args:
            fe_result (StaticResult): The solution containing the factorized stiffness matrix (K) and
                displacement/generalized coordinates (GC). K should be pre-factorized for efficiency.
            assembly (Assembly): The Finite Element Assembly object containing parts, elements, loads, etc.
            design_vars (torch.Tensor): A tensor representing the design variables.
                It must be the source of gradients for `apply_func`.
            apply_func (Callable[[Assembly, torch.Tensor], None]): 
                A callback to apply design variables to the assembly.
                - Signature: `def apply_func(assembly: Assembly, design_vars: torch.Tensor) -> None`
                - Behavior: Modify `assembly` in-place using `design_vars`. Operations must be traceable
                by Autograd (e.g., `part.nodes = original_nodes + design_vars.reshape_as(part.nodes)`).
            compute_objective_func (Callable[[StaticResult, Assembly], torch.Tensor]): 
                A callback to compute the objective scalar.
                - Signature: `def compute_objective_func(fe_result: StaticResult, assembly: Assembly) -> torch.Tensor`
                - Args: `fe_result` contains the necessary solution data.
                - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).

        Returns:
            torch.Tensor: The gradient of the objective with respect to `design_vars`.
                Shape matches `design_vars`.
        """
        try:
            # 0. Set load parameters
            self.assembly.set_work_conditions(fe_result.work_conditions)

            # 1. Factorize system if needed
            if fe_result.if_factorized is False:
                fe_result.factorize_stiffness_matrix(assembly=self.assembly)

            # 2. Prepare Autograd graph
            design_vars_grad = design_vars.clone().detach().requires_grad_(True)
            GC_grad = fe_result.GC.clone().detach().requires_grad_(True)
            fe_result.GC = GC_grad  # Use GC_grad for the assembly to track gradients through R and K

            # 3. Apply Design Variables
            apply_func(self.assembly, design_vars_grad)
            self.assembly.initialize()

            # 4. Compute Objective
            objective = compute_objective_func(fe_result, self.assembly)

            # 5. Backward Pass 1 (d_Obj / d_Vars and d_Obj / d_U)
            objective.backward(retain_graph=True)

            # 6. Adjoint Equation Solve (K * lambda = - d_Obj/d_U)
            if GC_grad.grad is None:
                raise RuntimeError("Objective function must depend on the displacement GC.")
                
            LdU = GC_grad.grad.clone().detach()
            W = fe_result.K_solver.solve(fe_result.K_sp, -LdU.cpu().numpy())
            W_tensor = torch.tensor(W, dtype=GC_grad.dtype, device=GC_grad.device)

            # 7. Backward Pass 2 (Total Sensitivity, lambda^T * R)
            R = self.assembly.assemble_force(GC=fe_result.GC)
            work = torch.dot(W_tensor, R)
            work.backward()
            
            if design_vars_grad.grad is None:
                return torch.zeros_like(design_vars)
        
        finally:
            # Cleanup: Detach all tensors in assembly to prevent graph explosion in next run
            self._detach_recursive(self.assembly)
            self._detach_recursive(fe_result)
            fe_result.remove_stored_factorization()


        return design_vars_grad.grad.clone().detach()
    
    def get_jacobian_sensitivity(
        self,
        fe_result: StaticResult,
        design_vars: torch.Tensor,
        load_names: list[str],
        apply_func: Callable[[Assembly, torch.Tensor], None] ,
        compute_objective_func: Callable[[StaticResult, Assembly], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Jacobian sensitivity (dR/dVars) for the static problem.

        This function computes the sensitivity of the residual forces with respect to design variables,
        which is essential for gradient-based optimization and design.

        Args:
            fe_result (StaticResult): The result object containing the current state and factorized stiffness matrix.
            design_vars (torch.Tensor): A tensor representing the design variables.
                It must be the source of gradients for `apply_func`.
            load_names (list[str]): The names of the loads for which to compute the Jacobian sensitivity.
            apply_func (Callable[[Assembly, torch.Tensor, None]): 
                A callback to apply design variables to the assembly.
                - Signature: `def apply_func(assembly: Assembly, design_vars: torch.Tensor) -> None`
                - Behavior: Modify `assembly` in-place using `design_vars`. Operations must be traceable
                by Autograd (e.g., `part.nodes = original_nodes + design_vars.reshape_as(part.nodes)`).
            compute_objective_func (Callable[[StaticResult, Assembly], torch.Tensor]): 
                A callback to compute the objective scalar.
                - Signature: `def compute_objective_func(fe_result: StaticResult, assembly: Assembly) -> torch.Tensor`
                - Args: `fe_result` contains the necessary solution data.
                - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).
        Returns:
            torch.Tensor: The sensitivity of the objective with respect to design variables.
                Shape matches `design_vars`.
        """
        objfuncs = lambda fe_results, assembly: compute_objective_func(fe_results[0], assembly)
        return self.get_jacobian_sensitivity_multistep(
            fe_results=[fe_result],
            design_vars=design_vars,
            load_names=load_names,
            apply_func=apply_func,
            compute_objective_funcs=objfuncs,
            )

    def get_jacobian_sensitivity_multistep(        
        self,
        fe_results: list[StaticResult],
        design_vars: torch.Tensor,
        load_names: list[str],
        apply_func: Callable[[Assembly, torch.Tensor], None] ,
        compute_objective_funcs: Callable[[list[StaticResult], Assembly], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Jacobian sensitivity (dR/dVars) for the static problem.

        This function computes the sensitivity of the residual forces with respect to design variables,
        which is essential for gradient-based optimization and design.

        Args:
            fe_results (list[StaticResult]): A list of result objects containing the current state and factorized stiffness matrices.
            design_vars (torch.Tensor): A tensor representing the design variables.
                It must be the source of gradients for `apply_func`.
            load_names (list[str]): The names of the loads for which to compute the Jacobian sensitivity.
            apply_func (Callable[[Assembly, torch.Tensor, None]): 
                A callback to apply design variables to the assembly.
                - Signature: `def apply_func(assembly: Assembly, design_vars: torch.Tensor) -> None`
                - Behavior: Modify `assembly` in-place using `design_vars`. Operations must be traceable
                by Autograd (e.g., `part.nodes = original_nodes + design_vars.reshape_as(part.nodes)`).
            compute_objective_funcs (Callable[[list[StaticResult], Assembly], torch.Tensor]): 
                A callback to compute the objective scalar.
                - Signature: `def compute_objective_func(fe_results: list[StaticResult], assembly: Assembly) -> torch.Tensor`
                - Args: `fe_results` contains the necessary solution data.
                - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).
        Returns:
            torch.Tensor: The sensitivity of the objective with respect to design variables.
                Shape matches `design_vars`.
        """
        try:
            design_vars_grad = design_vars.clone().detach().requires_grad_(True)
            apply_func(self.assembly, design_vars_grad)
            self.assembly.initialize()

            # 0. Set gradient required for each step's GC and Jacobians
            for idx, fe_result in enumerate(fe_results):
                # 0. Set load parameters and prepare jacobian
                self.assembly.set_work_conditions(fe_result.work_conditions)

                # 1. Factorize system if needed
                if fe_result.if_factorized is False:
                    fe_result.factorize_stiffness_matrix(assembly=self.assembly)

                # 2. Prepare Jacobian
                if fe_result.jacobian.keys() >= set(load_names):
                    jacobian_dict = fe_result.jacobian
                else:                
                    jacobian_dict = self.get_jacobian(result=fe_result, load_names=load_names)

                # 3. Prepare Autograd graph
                GC_grad = fe_result.GC.clone().detach().requires_grad_(True)
                jacobian_grad = {k: v.detach().clone().requires_grad_(True) for k, v in jacobian_dict.items()}

                fe_result.GC = GC_grad  # Use GC_grad for the assembly to track gradients through R and K
                fe_result.jacobian = jacobian_grad  # Use jacobian_grad to track gradients through the Jacobian

            # 4. Compute first adjoint vector W0
            objective = compute_objective_funcs(fe_results, self.assembly)

            # 5. Get Ldx and Ldy for the adjoint solve
            objective.backward(retain_graph=True)

            logger.info("Computing Jacobian sensitivity for each step:")
            logger.info(f" -Step 0/{len(fe_results)} sensitivity computed.\r")
            for idx, fe_result in enumerate(fe_results):
                # Each step must use its own load state when assembling R, dR/dp and dK/dp.
                self.assembly.set_work_conditions(fe_result.work_conditions)
                
                # 6. Get R for the current step
                R_grad = self.assembly.assemble_force(GC=fe_result.GC)

                if len(load_names) > 0:
                    jacobian = torch.cat([fe_result.jacobian[load_name] for load_name in load_names], dim=1).detach()
                else:
                    jacobian = torch.zeros((fe_result.GC.numel(), 0), device=fe_result.GC.device, dtype=fe_result.GC.dtype)

                # 7. Compute Ldx and Ldy for the adjoint solve
                if fe_result.GC.grad is None:
                    Ldx = torch.zeros_like(fe_result.GC)
                else:
                    Ldx = fe_result.GC.grad.clone().detach()

                if len(load_names) > 0:
                    Ldy_dict = {k: v.grad.clone().detach() if v.grad is not None else torch.zeros_like(v) for k, v in fe_result.jacobian.items()}
                    Ldy = torch.cat([Ldy_dict[load_name] for load_name in load_names], dim=1).detach() # Shape: (num_dofs, num_load_params)
                else:
                    Ldy = torch.zeros((fe_result.GC.numel(), 0), device=fe_result.GC.device, dtype=fe_result.GC.dtype)

                # 8. sensitivity for GC
                W0 = -fe_result.K_solver.solve(fe_result.K_sp, Ldx.cpu().numpy())
                W0_tensor = torch.tensor(W0, dtype=fe_result.GC.dtype, device=fe_result.GC.device)

                # 9. For each parameter, compute the Jacobian sensitivity using the chain rule:

                ## get the current load parameters as a single tensor
                total_load_params_list = []
                for load_name in load_names:
                    load = self.assembly._loads[load_name]
                    total_load_params_list.append(load._parameters.flatten())

                if len(total_load_params_list) > 0:
                    total_load_params = torch.cat(total_load_params_list, dim=0)
                else:
                    total_load_params = torch.zeros((0,), device=fe_result.GC.device, dtype=fe_result.GC.dtype)
                num_load_params = total_load_params.numel()

                obj_part_y = torch.zeros(1, device=fe_result.GC.device, dtype=fe_result.GC.dtype)
                K_indices = self.assembly.assemble_Stiffness_Matrix(GC=fe_result.GC)[1]
                wKdp = torch.zeros_like(fe_result.GC)
                for para_idx in range(num_load_params):
                    ### compute the Jacobian sensitivity using the chain rule:
                    Ldy_now = Ldy[:, para_idx]
                    if Ldy_now.abs().sum() < 1e-8:
                        continue


                    W1 = fe_result.K_solver.solve(fe_result.K_sp, Ldy_now.cpu().numpy())
                    W1_tensor = torch.tensor(W1, dtype=fe_result.GC.dtype, device=fe_result.GC.device)
                    ### evaluate the stiffness matrix sensitivity dK/dPara using autograd
                    def get_Kdp(load_now: torch.Tensor):
                        GC_now = fe_result.GC + jacobian[:, para_idx].detach() * (load_now - load_now.detach())
                        index_now = 0
                        for load_name in load_names:
                            load = self.assembly._loads[load_name]
                            param_len = load._parameters.numel()
                            idx_now = para_idx - index_now
                            if 0 <= idx_now < param_len:
                                load._parameters[idx_now] = load_now
                                break
                            index_now += param_len
                        K_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)[2]
                        return K_values
                    Kdp_values = torch.autograd.functional.jvp(
                        get_Kdp, total_load_params[para_idx: para_idx+1], torch.ones([1]), create_graph=False)[1]
                    Kdp = torch.sparse_coo_tensor(K_indices, Kdp_values, size=fe_result.K_sp.shape)

                    wKdp += Kdp@ W1_tensor

                    ### evaluate the stiffness matrix sensitivity dR/dPara using autograd
                    def get_Rdp(load_now: torch.Tensor):
                        GC_now = fe_result.GC + jacobian[:, para_idx].detach() * (load_now - load_now.detach())
                        index_now = 0
                        for load_name in load_names:
                            load = self.assembly._loads[load_name]
                            param_len = load._parameters.numel()
                            idx_now = para_idx - index_now
                            if 0 <= idx_now < param_len:
                                load._parameters[idx_now] = load_now
                                break
                            index_now += param_len
                        R = self.assembly.assemble_force(GC=GC_now)
                        return (R * W1_tensor).sum()

                    obj_part_y -= torch.autograd.functional.jacobian(get_Rdp, total_load_params[para_idx: para_idx+1], create_graph=True)

                if wKdp.abs().sum() > 1e-8:
                    wKdpKinv = fe_result.K_solver.solve(fe_result.K_sp, wKdp.cpu().numpy())
                else:
                    wKdpKinv = torch.zeros_like(wKdp)
                wKdpKinv_tensor = torch.tensor(wKdpKinv, dtype=fe_result.GC.dtype, device=fe_result.GC.device)
                
                obj_part_x = ((W0_tensor + wKdpKinv_tensor) * R_grad).sum()

                obj_total = obj_part_x + obj_part_y

                if idx < len(fe_results) - 1:
                    obj_total.backward(retain_graph=True)
                else:
                    obj_total.backward()

                fe_result.remove_stored_factorization()
                self._detach_recursive(fe_result)

                logger.info(f" -Step {idx+1}/{len(fe_results)} sensitivity computed.")

                
            sensitivity = design_vars_grad.grad.clone().detach()

        finally:            
            # Cleanup: Detach all tensors in assembly to prevent graph explosion in next run
            self._detach_recursive(self.assembly)

        logger.info("All steps sensitivity computation completed.")

        return sensitivity

    @classmethod
    def _detach_recursive(cls, obj: object, visited: set=None):
        """
        Recursively detach tensors to clean up the computation graph.
        For mutable containers (list, dict, objects), replaces tensors with detached versions.
        This avoids inplace detach_() errors on views.
        """
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    obj[k] = v.detach()
                else:
                    cls._detach_recursive(v, visited)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, torch.Tensor):
                    obj[i] = v.detach()
                else:
                    cls._detach_recursive(v, visited)
        elif isinstance(obj, tuple):
            for v in obj:
                cls._detach_recursive(v, visited)
        elif hasattr(obj, '__dict__'):
            # Iterate over a copy of items to avoid modification issues
            for k, v in list(obj.__dict__.items()):
                if k.startswith('__'): continue 
                if isinstance(v, torch.Tensor):
                    setattr(obj, k, v.detach())
                else:
                    cls._detach_recursive(v, visited)

    # endregion


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
            logger.warning('the newton dirction is not the decrease direction')

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
                break
            loopc2 += 1

        return alpha, GCnew.detach(), energy_new.detach()

    def _solve_iteration(self,
                         GC: torch.Tensor,
                         tol_error: float):

        # record the information of the solver
        total_time = 0.0
        time_items = {'assemble': [], 'linear': [], 'line_search': [], 'step': []}

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

        self.__low_alpha_count = 0

        # begin the iteration
        while True:

            if self._iter_now > self.maximum_iteration:
                logger.warning('maximum iteration reached')
                return GC, time.time() - t00, time_items, False

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
                return GC, time.time() - t00, time_items, False
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
                return GC, time.time() - t00, time_items, False

            # update the GC
            GC = GCnew

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
            
            time_items['assemble'].append(t2 - t1)
            time_items['linear'].append(t3 - t2)
            time_items['line_search'].append(t4 - t3)
            time_items['step'].append(t4 - t1)

            logger.debug('iter:%d, alpha:%.2f, total_time:%.2f, energy:%.4e, delta_energy:%.4e, error:%.4e, Ktime:%.2f, linear:%.2f, search:%.2f, step:%.2f' % (
                self._iter_now, alpha, t4 - t00, energy[-1], energy[-1] - energy[-2], R.abs().max(), t2 - t1, t3 - t2, t4 - t3, t4 - t1))
            
            if (dGC.abs().max() < tol_error and R.abs().max() < tol_error) or R.abs().max() < 1e-6:
                break

        
        total_time = time.time() - t00

        return GC, total_time, time_items, True
    

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

        matrix_size = int(K_indices.max().item()) + 1 if K_indices.numel() else 0
        if matrix_size != R.numel():
            raise ValueError(
                "Reduced stiffness matrix and residual have different dimensions: "
                f"matrix={matrix_size}, residual={R.numel()}. "
                "Check active constraints and boundary conditions.")

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

        if self.__low_alpha_count > 5 or R_preconditioned.abs().max() < 1e-2 or K_values_preconditioned.device.type == 'cpu':
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
                                                       tol=1e-3,
                                                       max_iter=1200)
        result = dx.to(R.dtype) / diag
        return result

    # endregion
