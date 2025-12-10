import torch


@torch.no_grad()
@torch.jit.script
def conjugate_gradient(A_indices: torch.Tensor,
                        A_values: torch.Tensor,
                        b: torch.Tensor,
                        x0: torch.Tensor = torch.zeros([0]),
                        tol: float = 1e-3,
                        max_iter: int = 1500):
    # A_values = A_values.to(torch.float64)
    # b = b.to(torch.float64)
    # x0 = x0.to(torch.float64)
    if x0.numel() == 0:
        x0 = torch.zeros_like(b)

    A = torch.sparse_coo_tensor(A_indices, A_values,
                                [b.shape[0], b.shape[0]]).to_sparse_csr()

    # reference error for convergence
    r_r0 = torch.dot(b, b)

    # 定义初始解x和残差r
    x = x0.clone()

    r = b - A @ x
    p = r
    rsold = torch.dot(r, r)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if rsnew / r_r0 < tol:
            break
        p = r + rsnew / rsold * p
        rsold = rsnew

    return x

def pypardiso_solver(A_indices: torch.Tensor, A_values: torch.Tensor,
                        b: torch.Tensor, if_return_factorization: bool = False) -> torch.Tensor:
    import pypardiso
    import scipy.sparse as sp

    A_sp = sp.coo_matrix(
        (A_values.detach().cpu().numpy(), (A_indices[0].cpu().numpy(),
                                    A_indices[1].cpu().numpy()))).tocsr()

    b_np = b.detach().cpu().numpy()

    if if_return_factorization:
        k_solver = pypardiso.PyPardisoSolver()
        k_solver.factorize(A_sp)

        x = k_solver.solve(A_sp, b_np)

        return torch.from_numpy(x).to(b.dtype).to(b.device), k_solver
    else:
        x = pypardiso.spsolve(A_sp, b_np)
        return torch.from_numpy(x).to(b.dtype).to(b.device)