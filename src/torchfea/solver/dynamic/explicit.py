from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Assembly
import time
import torch
from ..basesolver import BaseSolver

class DynamicExplicitSolver(BaseSolver):

    def __init__(self, time_end: float = 1.0, time_per_storage: float = 1e-4) -> None:
        """
        Initialize the Explicit Dynamic Solver.

        Args:
            time_end (float): The end time of the simulation.
            dump_factor (float): Damping factor for numerical stability.
        """
        self._time_end: float = time_end
        self._time_per_storage: float = time_per_storage
        self._next_storage_time: float = 0.0  # 下一个需要存储的时间点


        self._GC_list: list[torch.Tensor] = None
        self._GV_list: list[torch.Tensor] = None
        self._GA_list: list[torch.Tensor] = None
        self._time_list: list[float] = []
        self._deltaT: float = 0.0 # 将在初始化时计算

    def initialize(self, assembly: Assembly):
        """
        Initialize the solver with the assembly and compute critical time step.
        """
        super().initialize(assembly=assembly)
        self.assembly.initialize_dynamic()

        # 1. 计算临界时间步长 (CFL Condition)
        # 这是一个简化估算，精确计算需要遍历所有单元
        # Δt_crit = L_min / c,  c = sqrt(E/ρ)
        # 这里我们用一个经验值或一个估算函数
        self._deltaT = self.estimate_critical_timestep()
        print(f"Estimated critical timestep: {self._deltaT:.4e} s")
        if self._deltaT <= 0:
            raise ValueError("Critical timestep must be positive.")

        print(f"Critical timestep estimated: {self._deltaT:.4e} s")

        self._GC_list = []
        self._GV_list = []
        self._GA_list = []
        self._time_list = []

    def estimate_critical_timestep(self, safety_factor=0.8) -> float:
        """
        Estimates the critical timestep for stability (CFL condition).
        This is a placeholder and should be implemented based on element sizes and material properties.
        """
        # 这是一个非常粗略的估计，您需要根据您的模型进行调整
        # 例如，遍历所有单元，找到最小的 L/c
        # L: 单元特征长度, c: 材料波速 sqrt(E/rho)
        # 假设一个经验值
        estimated_crit_dt = 5e-6 
        return safety_factor * estimated_crit_dt

    def solve(self, GC0: torch.Tensor = None, GV0: torch.Tensor = None, *args, **kwargs) -> bool:
        """
        Solves the finite element analysis problem using the explicit central difference method.
        """
        t_start = time.time()

        # 1. 设置初始条件
        if GC0 is None:
            GC0 = self.assembly.GC.clone()
        if GV0 is None:
            GV0 = torch.zeros_like(self.assembly.GC)

        # 2. 计算初始加速度 a_0 = M⁻¹ * (F_ext(0) - F_int(u_0))
        # F_int(u_0) 通常为0，除非有预应力
        R0 = -self.assembly.assemble_force(GC=GC0) # 只取残余力向量
        mass_inv = self.get_lumped_mass_inv(GC0)
        GA0 = mass_inv * R0

        # 3. 初始化列表并存储初始状态
        self._GC_list.append(GC0)
        self._GV_list.append(GV0)
        self._GA_list.append(GA0)
        self._time_list.append(0.0)

        # 4. 计算 "半步" 初始速度 v_{-1/2}
        GV_half_prev = GV0 - 0.5 * self._deltaT * GA0

        # 5. 主时间步循环
        iteration = 0
        current_time = 0.0
        GA_now = GA0
        while current_time < self._time_end:
            
            # a. 获取上一步的状态
            GC_prev = self._GC_list[-1]
            
            # b. 更新 "半步" 速度: v_{n+1/2} = v_{n-1/2} + Δt * a_n
            GV_half_now = GV_half_prev + self._deltaT * GA_now

            # c. 更新位移: u_{n+1} = u_n + Δt * v_{n+1/2}
            GC_now = GC_prev + self._deltaT * GV_half_now

            # d. 计算新的内力和外力: R_{n+1} = F_ext(t_{n+1}) - F_int(u_{n+1})
            # 注意：assemble_Stiffness_Matrix 返回的是 F_ext - F_int
            R_now = -self.assembly.assemble_force(GC=GC_now)

            # e. 计算新的加速度: a_{n+1} = M⁻¹ * R_{n+1}
            
            GA_now = mass_inv * R_now

            # f. 更新节点速度 (用于输出): v_{n+1} = (v_{n+1/2} + v_{n-1/2}) / 2
            GV_now = 0.5 * (GV_half_now + GV_half_prev)

            # g. 更新时间
            current_time += self._deltaT

            # h. 【新增】检查是否需要存储当前步结果
            if current_time >= self._next_storage_time or current_time >= self._time_end:
                self._GC_list.append(GC_now)
                self._GV_list.append(GV_now)
                self._GA_list.append(GA_now)
                self._time_list.append(current_time)
                
                # 更新下一个存储时间点
                self._next_storage_time += self._time_per_storage

            # i. 准备下一次迭代
            GV_half_prev = GV_half_now
            iteration += 1

            if iteration % 100 == 0: # 每100步打印一次信息
                print(f"Step: {iteration}, Time: {current_time:.4e} s, Max Disp: {GC_now.abs().max():.4e}, time_cost: {time.time() - t_start:.2f} s")

        # 6. 结束
        self.GC = self._GC_list[-1]
        self.assembly.RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(self.assembly.GC))
        t_end = time.time()
        print('---' * 8, 'Explicit FEA Finished', '---' * 8)
        print(f'Total steps: {iteration}, Total time: {t_end - t_start:.2f} s')
        return True

    def get_lumped_mass_inv(self, GC: torch.Tensor) -> torch.Tensor:
        """
        Assembles the lumped mass matrix and returns its inverse.
        For a diagonal matrix, the inverse is just the reciprocal of its diagonal elements.
        """
        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC)
        
        # 创建一个完整的质量向量 (对角线)
        mass_vector = torch.zeros(GC.shape[0], device=GC.device, dtype=GC.dtype)
        
        # 使用 scatter_add_ 来集成所有质量项到对角线上
        # 注意：这假设 assemble_mass_matrix 返回的是完整的 M_ij 矩阵项
        # 如果它只返回上三角或下三角，需要相应调整
        # 这里我们假设它返回了所有项，包括对角线 M_ii
        mass_vector.scatter_add_(0, mass_indices[0], mass_values * (mass_indices[0] == mass_indices[1]))
        mass_vector.scatter_add_(0, mass_indices[1], mass_values * (mass_indices[0] != mass_indices[1]))

        # 防止除以零
        mass_vector[mass_vector.abs() < 1e-12] = 1.0
        
        return 1.0 / mass_vector