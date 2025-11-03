# AMGCL Solver for OpenFOAM

基于 AMGCL 库的高性能代数多重网格 (AMG) 线性求解器，支持串并行自动切换。

## 主要特性

- ✅ 自动串并行切换（无需修改配置）
- ✅ 高性能稀疏矩阵求解（优化的 CSR 格式构建）
- ✅ 支持对称和非对称矩阵
- ✅ 兼容 OpenFOAM 标准残差计算
- ✅ 灵活的矩阵缓存策略

## 编译

```bash
cd $HUNTERFOAM_ROOT/src/amgclFoam
wmake libso
```

## 配置示例

在 `system/fvSolution` 中配置求解器：

### 1. 压力场（对称矩阵）

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
            options
            {
                solver      cg;           // 共轭梯度（对称矩阵专用）
                relaxation  spai0;        // SPAI0 预条件器
                maxiter     100;
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
}
```

### 2. 速度场（非对称矩阵）

```cpp
solvers
{
    U
    {
        solver          amgcl;
        tolerance       1e-06;
        relTol          0.1;
        
        amgcl
        {
            options
            {
                solver      bicgstab;     // BiCGStab（非对称矩阵）
                relaxation  spai0;        // SPAI0 预条件器
                maxiter     100;
            }
            
            caching
            {
                matrix { update  always; }  // 每步更新矩阵
            }
        }
    }
}
```

### 3. 难收敛问题（GMRES）

```cpp
solvers
{
    T
    {
        solver          amgcl;
        tolerance       1e-06;
        relTol          0.01;
        
        amgcl
        {
            options
            {
                solver      gmres;
                relaxation  spai0;
                M           30;           // GMRES 重启参数
                maxiter     200;
            }
            
            caching
            {
                matrix { update  periodic; periodicCoeffs { frequency  5; } }
            }
        }
    }
}
```

## 求解器选项

### 可用求解器 (`solver`)

| 求解器 | 适用矩阵 | 说明 |
|--------|---------|------|
| `cg` | 对称正定 | 共轭梯度法，最适合压力方程 |
| `bicgstab` | 非对称 | 双共轭梯度稳定法（默认，推荐） |
| `gmres` | 非对称 | 广义最小残差法（需设置参数 `M`） |
| `bicgstabl` | 非对称 | BiCGStab(L)（需设置参数 `L`） |
| `fgmres` | 非对称 | 灵活GMRES |
| `idrs` | 非对称 | IDR(s) 方法 |

### 可用预条件器 (`relaxation`)

#### 串行模式
- `spai0` - SPAI(0) 稀疏近似逆（默认，推荐）
- `spai1` - SPAI(1) 更高精度
- `ilu0` - 不完全LU分解（收敛快但并行性差）

#### 并行模式（MPI）
- `spai0` - 并行友好（推荐）
- `spai1` - 并行友好

### 求解器特定参数

```cpp
options
{
    solver      gmres;
    M           30;        // GMRES/FGMRES 重启参数（默认30）
}
```

```cpp
options
{
    solver      bicgstabl;
    L           2;         // BiCGStab(L) 参数（默认2）
}
```

## 矩阵缓存策略

```cpp
caching
{
    matrix
    {
        update  always;    // 每步更新（速度场推荐）
        // 或
        update  periodic;  // 周期更新（压力场推荐）
        periodicCoeffs { frequency  1; }
    }
}
```

## 运行方式

### 串行
```bash
simpleFoam
```

### 并行
```bash
decomposePar
mpirun -np 4 simpleFoam -parallel
```

代码自动检测并行模式，无需修改配置。

## 推荐配置

| 场 | 求解器 | 预条件器 | 矩阵更新 |
|----|--------|---------|---------|
| 压力 `p` | `cg` | `spai0` | `periodic` |
| 速度 `U` | `bicgstab` | `spai0` | `always` |
| 温度 `T` | `bicgstab` | `spai0` | `periodic` |
| 标量 `k,epsilon` | `bicgstab` | `spai0` | `always` |

## 性能优化建议

1. **对称矩阵**：使用 `cg` 求解器
2. **非对称矩阵**：优先使用 `bicgstab`，收敛困难时用 `gmres`
3. **并行计算**：使用 `spai0` 预条件器
4. **串行计算**：可尝试 `ilu0` 预条件器（更快收敛）

## 调试

查看求解器详细输出：
```bash
export FOAM_DEBUG_lduMatrix=2
```

## 参考

- AMGCL: https://github.com/ddemidov/amgcl
- OpenFOAM: https://www.openfoam.com
