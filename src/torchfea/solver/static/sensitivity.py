from typing import Optional, Callable
import torch
from .result import StaticResult
from ...model.assembly import Assembly

def _detach_recursive(obj, visited=None):
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
                _detach_recursive(v, visited)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, torch.Tensor):
                obj[i] = v.detach()
            else:
                _detach_recursive(v, visited)
    elif isinstance(obj, tuple):
        for v in obj:
            _detach_recursive(v, visited)
    elif hasattr(obj, '__dict__'):
        # Iterate over a copy of items to avoid modification issues
        for k, v in list(obj.__dict__.items()):
            if k.startswith('__'): continue 
            if isinstance(v, torch.Tensor):
                setattr(obj, k, v.detach())
            else:
                _detach_recursive(v, visited)

def get_sensitivity(
    fe_result: StaticResult,
    assembly: Assembly,
    design_vars: torch.Tensor,
    apply_func: Callable[[Assembly, torch.Tensor], None],
    compute_objective_func: Callable[[torch.Tensor, Assembly], torch.Tensor],
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
              by Autograd (e.g., `part.nodes = original_nodes + design_vars`).
        compute_objective_func (Callable[[torch.Tensor, Assembly], torch.Tensor]): 
            A callback to compute the objective scalar.
            - Signature: `def compute_objective_func(GC: torch.Tensor, assembly: Assembly) -> torch.Tensor`
            - Args: `GC` is the displacement vector (detached from physics but tracking gradient).
            - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).

    Returns:
        torch.Tensor: The gradient of the objective with respect to `design_vars`.
            Shape matches `design_vars`.
    """
    try:
        # 1. Factorize system if needed
        if fe_result.if_factorized is False:
            fe_result.factorize_stiffness_matrix(assembly=assembly)

        assembly.set_load_parameters(fe_result.load_params)

        # 2. Prepare Autograd graph
        design_vars_ = design_vars.clone().detach().requires_grad_(True)
        GC_detached = fe_result.GC.clone().detach().requires_grad_(True)

        # 3. Apply Design Variables
        apply_func(assembly, design_vars_)
        assembly.initialize()

        # 4. Compute Objective
        objective = compute_objective_func(GC_detached, assembly)

        # 5. Backward Pass 1 (d_Obj / d_Vars and d_Obj / d_U)
        objective.backward(retain_graph=True)

        # 6. Adjoint Equation Solve (K * lambda = - d_Obj/d_U)
        if GC_detached.grad is None:
            raise RuntimeError("Objective function must depend on the displacement GC.")
            
        LdU = GC_detached.grad.clone().detach()
        W = fe_result.K_solver.solve(fe_result.K_sp, -LdU.cpu().numpy())
        W_tensor = torch.tensor(W, dtype=GC_detached.dtype, device=GC_detached.device)

        # 7. Backward Pass 2 (Total Sensitivity, lambda^T * R)
        R = assembly.assemble_force(GC=GC_detached)
        work = torch.dot(W_tensor, R)
        work.backward()
        
        if design_vars_.grad is None:
            return torch.zeros_like(design_vars)
            
        return design_vars_.grad.clone().detach()
    
    finally:
        # Cleanup: Detach all tensors in assembly to prevent graph explosion in next run
        _detach_recursive(assembly)
