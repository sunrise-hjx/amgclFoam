# AMGCL Solver for OpenFOAM

基于 AMGCL 库开发的高性能代数多重网格线性求解器，支持串并行自动切换。

## 功能特点

- ✅ **自动串并行切换**：无需修改配置，根据运行模式自动选择合适的求解器
- ✅ **多种求解器**：支持 CG, BiCGStab, BiCGStab(L), GMRES, FGMRES, IDRS 等
- ✅ **多种预条件器**：支持 SPAI0, ILU0, Damped Jacobi, Gauss-Seidel 等
- ✅ **兼容 OpenFOAM**：使用标准的残差计算方法
- ✅ **灵活配置**：通过 fvSolution 文件配置所有参数

## 编译

```bash
cd $HUNTERFOAM_ROOT/src/amgclFoam
source $WM_PROJECT_DIR/etc/openfoam
source $HUNTERFOAM_ROOT/etc/HFbashrc
wclean && wmake libso
```

## 使用方法

### 基本配置

在 `system/fvSolution` 中配置：

```cpp
solvers
{
    p
    {
        solver          amgcl;
        tolerance       1e-06;
        relTol          0.01;
        
        amgcl
        {
            // 求解器类型选择
            options
            {
                solver      bicgstab;    // 求解器类型
                relaxation  spai0;        // 松弛方法/预条件器
                coarsening  smoothed_aggregation;  // 粗化方法
                maxiter     100;          // 最大迭代次数
            }
            
            // 矩阵缓存设置
            caching
            {
                matrix
                {
                    update  periodic;
                    periodicCoeffs
                    {
                        frequency  1;
                    }
                }
            }
        }
    }
}
```

### 可用求解器类型

#### Serial & Parallel 支持

| 求解器 | 说明 | 适用场景 |
|--------|------|----------|
| `cg` | 共轭梯度法 | 对称正定矩阵 |
| `bicgstab` | 双共轭梯度稳定法 | 非对称矩阵（默认） |
| `bicgstabl` | BiCGStab(L) | 非对称矩阵，更稳定 |
| `gmres` | 广义最小残差法 | 非对称矩阵 |
| `fgmres` | 灵活GMRES | 非对称矩阵，变预条件 |
| `idrs` | IDR(s) 方法 | 非对称矩阵 |

### 可用松弛方法/预条件器

#### Serial 支持

| 方法 | 说明 | 特点 |
|------|------|------|
| `spai0` | SPAI(0) 稀疏近似逆（默认） | 并行性好，收敛快 |
| `spai1` | SPAI(1) 更高精度稀疏近似逆 | 更好的收敛性，计算量稍大 |
| `ilu0` | 不完全LU分解 | 收敛快，串行性强 |
| `iluk` | ILU(k) | 更高精度 ILU |
| `ilut` | ILUT | 基于阈值的 ILU |
| `damped_jacobi` | 阻尼Jacobi | 并行性好 |
| `gauss_seidel` | Gauss-Seidel | 经典方法 |

#### Parallel (MPI) 支持

- `spai0` - SPAI(0) 稀疏近似逆（并行友好）
- `spai1` - SPAI(1) 更高精度（并行友好）
- `damped_jacobi` - 阻尼Jacobi（并行友好）

### 高级配置示例

#### 1. 压力场求解（对称矩阵，CG求解器）

```cpp
p
{
    solver          amgcl;
    tolerance       1e-06;
    relTol          0.01;
    
    amgcl
    {
        options
        {
            solver      cg;           // 共轭梯度
            relaxation  spai0;        // SPAI0 预条件
            maxiter     100;
            
            // AMG 参数
            amg
            {
                eps_strong  0.0;      // 强连接阈值
                over_interp 1.5;      // 插值参数
            }
        }
        
        caching
        {
            matrix
            {
                update  periodic;
                periodicCoeffs { frequency  1; }
            }
        }
    }
}
```

#### 2. 速度场求解（GMRES + ILU0）

```cpp
U
{
    solver          amgcl;
    tolerance       1e-06;
    relTol          0.1;
    
    amgcl
    {
        options
        {
            solver      gmres;        // GMRES 求解器
            relaxation  ilu0;         // ILU0 预条件
            M           30;           // GMRES 重启参数
            maxiter     100;
        }
        
        caching
        {
            matrix { update  always; }
        }
    }
}
```

#### 3. BiCGStab(L) 求解器

```cpp
p
{
    solver          amgcl;
    tolerance       1e-06;
    relTol          0.01;
    
    amgcl
    {
        options
        {
            solver      bicgstabl;    // BiCGStab(L)
            L           2;            // BiCGStab(L) 参数
            relaxation  spai0;
            maxiter     100;
        }
        
        caching
        {
            matrix { update  periodic; periodicCoeffs { frequency  1; } }
        }
    }
}
```

## 串并行运行

### 串行运行

```bash
pisoFoam
```

### 并行运行

```bash
decomposePar
mpirun -np 4 pisoFoam -parallel
```

**注意**：代码会自动检测运行模式，无需修改配置！

## 性能调优建议

1. **对称正定矩阵**（如压力泊松方程）：
   - 使用 `cg` + `spai0` 或 `ilu0`

2. **非对称矩阵**（如对流扩散方程）：
   - 使用 `bicgstab` + `spai0`（默认，平衡性能）
   - 或 `gmres` + `ilu0`（更稳定）

3. **难收敛问题**：
   - 使用 `bicgstabl` (L=2或3)
   - 或 `fgmres` / `lgmres`
   - 降低 `relTol`，增加 `maxiter`

4. **并行计算**：
   - 优先使用 `spai0` 或 `damped_jacobi`（并行友好）
   - 避免使用 `ilu0`, `gauss_seidel`（串行性强）

## 调试信息

设置环境变量查看详细信息：

```bash
export FOAM_DEBUG_lduMatrix=2  # 查看求解器详细输出
```

## 问题排查

1. **编译错误**：确保已加载 OpenFOAM 和 HunterFoam 环境
2. **运行时找不到求解器**：检查库是否正确编译和加载
3. **MPI 初始化错误**：确保串行运行时不使用 `mpirun`
4. **收敛慢**：尝试不同的求解器和预条件器组合

## 参考

- AMGCL 库：https://github.com/ddemidov/amgcl
- OpenFOAM 官方文档：https://www.openfoam.com

