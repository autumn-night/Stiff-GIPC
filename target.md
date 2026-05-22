# Stiff-GIPC Headless Benchmark Target 实施清单

## 1. 目标

在当前项目中新增一个**独立的、可无头运行的 benchmark target**，用于稳定评测以下三组 baseline：

- `GIPC`
- `StiffGIPC(SRBK)`
- `StiffGIPC(CEMAS+SRBK)`

统一输出以下指标：

- `Hess`
- `LSolver`
- `LineS`
- `Misc`
- `TimeTot`

目标是让该 target：

- 可以在**远程服务器无可视化环境**运行
- 可以被具身智能方向同学直接复用
- 可以作为后续性能调优的唯一基准入口
- 可以先覆盖仓库现有可复现资产，后续再接外部场景/数据集

注意，本环境中不具备cuda环境，无法编译相关代码。

---

## 2. baseline 定义（必须固定）

### 2.1 三组 baseline 的工程定义

- `GIPC`
  - `Legacy Assembly / Legacy SpMV`
  - `GPU MAS`（Morton/非 CEMAS 路径）

- `StiffGIPC(SRBK)`
  - `SRBK Assembly / SRBK SpMV`
  - `GPU MAS`（Morton/非 CEMAS 路径）

- `StiffGIPC(CEMAS+SRBK)`
  - `SRBK Assembly / SRBK SpMV`
  - `CEMAS`（当前 GROUP 路径）

### 2.2 禁止事项

- 不允许把三组 baseline 混成编译期宏开关后“只保留一个可执行路径”
- 不允许把 `GIPC` 近似成 `Diag + SRBK`
- 不允许让场景、步长、容差、材质参数在 baseline 间不一致

---

## 3. 指标口径（先固定再实现）

### 3.1 对外报告口径

- `Hess`：每帧所有 Newton 迭代中梯度/Hessian 计算总时间
- `LSolver`：每帧所有 Newton 迭代中线性系统求解总时间
- `LineS`：每帧所有 Newton 迭代中 line search 总时间
- `TimeTot`：一帧完整求解总时间
- `Misc`：`TimeTot - Hess - LSolver - LineS`

### 3.2 内部保留原始口径

为了后续排查，原始分段建议继续保留：

- `time0 -> Hess`
- `time1 -> LSolver`
- `time2 -> CCD/feasible-step raw`
- `time3 -> LineS`
- `time4 -> postLineSearch raw`

说明：

- `Misc` 对外按汇总口径输出
- `time2/time4` 不纳入首版对外主表，但必须写入 raw report，方便后续回到论文更细口径

---

## 4. 场景范围（v1）

首版只覆盖仓库已有资产，建立稳定 benchmark 流程。

### 4.1 v1 内置场景

- `cloth_bunny`
  - 对应当前 `set_case2()`

- `wrecking_ball`
  - 对应当前 `set_case3()`

- `mat_twist`
  - 对应当前 `set_case5()`

- `box_pile`
  - 对应当前 `set_case6()`

### 4.2 v2 预留

- `gm-100` 不直接作为仿真输入
- 先只在 benchmark 配置层预留：
  - `task_id`
  - `dataset`
  - `asset_root`
  - `notes`
- 等后续有人把 `gm-100` 任务转成 mesh/material/pose manifest 后再接入

---

## 5. 交付物

必须交付以下内容：

- 一个无头 benchmark 可执行：`gipc_bench`
- 一个保留原可视化行为的 viewer 可执行：`gipc_viewer`
- 一个公共核心库：`stiffgipc_core`
- 一份 benchmark suite 配置样例：`bench_suite.json`
- 一份场景 manifest 样例（可选首版空壳）
- 一份结果输出样例：
  - `raw.json`
  - `summary.csv`
  - `meta.json`

---

## 6. 目录与 target 设计

## 6.1 target 拆分

- `stiffgipc_core`
  - 包含仿真核心、线性系统、场景构建、benchmark 公共逻辑

- `gipc_viewer`
  - 仅负责 GLUT/OpenGL 初始化、渲染、交互

- `gipc_bench`
  - 仅负责无头运行、批量场景执行、输出报告

## 6.2 建议新增目录

- `app/viewer/`
  - viewer 专用入口

- `app/common/`
  - 共享的场景、设置、bootstrap 逻辑

- `app/bench/`
  - benchmark CLI、runner、reporter

建议文件布局如下：

- `app/common/sim_settings.h`
- `app/common/sim_settings.cpp`
- `app/common/scene_registry.h`
- `app/common/scene_registry.cpp`
- `app/common/sim_bootstrap.h`
- `app/common/sim_bootstrap.cpp`
- `app/bench/main.cpp`
- `app/bench/benchmark_config.h`
- `app/bench/benchmark_config.cpp`
- `app/bench/benchmark_runner.h`
- `app/bench/benchmark_runner.cpp`
- `app/bench/benchmark_report.h`
- `app/bench/benchmark_report.cpp`
- `app/viewer/main.cpp`

---

## 7. 文件级改动清单

## 7.1 构建系统

- [ ] 修改 `CMakeLists.txt`
  - [ ] 停止直接用全量 `file(GLOB_RECURSE ...)` 把所有 `main` 和 viewer 代码塞进同一个目标
  - [ ] 建立 `stiffgipc_core`
  - [ ] 建立 `gipc_viewer`
  - [ ] 建立 `gipc_bench`
  - [ ] 让 `gipc_bench` 不依赖 `GLEW/GLUT/OpenGL`
  - [ ] 继续链接 `metis_partition`

- [ ] 检查 `MeshProcess` 的链接方式
  - [ ] 明确 `metis_partition` 作为公共依赖被 core 复用

## 7.2 场景与仿真初始化抽取

- [ ] 从 `StiffGIPC/gl_main.cu` 中抽离以下逻辑到 `app/common/`
  - [ ] `DefaultSettings()`
  - [ ] `LoadSettings()`
  - [ ] `initFEM()`
  - [ ] `set_case1 ~ set_case6()`
  - [ ] `setMAS_partition()`
  - [ ] `initScene()` 中非渲染部分
  - [ ] host -> device 数据上传和 `ipc` bootstrap 逻辑

- [ ] 保留 viewer 专属代码在 `gipc_viewer`
  - [ ] `draw_*`
  - [ ] `display()`
  - [ ] GLUT callback

- [ ] 设计一个共享 bootstrap 接口
  - [ ] 输入：场景名、参数、baseline config、输出目录
  - [ ] 输出：初始化后的 `ipc`、`tetMesh`、`d_tetMesh`

## 7.3 baseline 运行时切换

- [ ] 新增统一 runtime config
  - [ ] `enum class BenchmarkBaseline`
  - [ ] `enum class SpmvBackend`
  - [ ] `enum class MasBackend`

- [ ] 在 `GIPC` 或 `GlobalLinearSystem` 上挂接 backend config
  - [ ] 场景初始化完成后再注入 backend
  - [ ] 不允许靠重新编译切 baseline

## 7.4 MAS 路径拆分

- [ ] 改造 `MASPreconditioner.cu`
  - [ ] 去掉用 `#define GROUP` 锁死 `CEMAS` 的方式
  - [ ] 运行时支持：
    - [ ] `GpuMas`
    - [ ] `CEMAS`

- [ ] 把以下阶段做成双实现选择器
  - [ ] `BuildConnectMaskL0()`
  - [ ] `PreparePrefixSumL0()`
  - [ ] `BuildLevel1()`
  - [ ] `BuildConnectMaskLx()`
  - [ ] `BuildCollisionConnection()`
  - [ ] `PrepareHessian_bcoo()`
  - [ ] `BuildMultiLevelR()`
  - [ ] `CollectFinalZ()`

- [ ] 目标结果
  - [ ] `GROUP` 路径对应 `CEMAS`
  - [ ] 非 `GROUP` 路径对应论文中的 `GPU MAS`

## 7.5 SpMV / assembly backend 拆分

- [ ] 改造 `linear_system/utils/spmv.*`
  - [ ] 暴露 `legacy_gipc_spmv()`
  - [ ] 暴露 `srbk_spmv()`

- [ ] 改造 `linear_system/utils/converter.*`
  - [ ] 暴露 `legacy_gipc_convert()`
  - [ ] 暴露 `srbk_convert()`

- [ ] 改造 `GlobalLinearSystem`
  - [ ] `convert_new()` 改为 backend-aware
  - [ ] `spmv()` 改为 backend-aware

- [ ] 从 git 历史恢复 legacy 路径
  - [ ] 优先参考 `a3479e7`
  - [ ] 恢复旧的 non-SRBK assembly / SpMV 路径
  - [ ] 明确与当前 SRBK 路径的边界

## 7.6 预条件器创建逻辑改造

- [ ] 改造 `gipc/gipc.cu` 中 `create_LinearSystem()`
  - [ ] 依据 runtime baseline 创建正确 backend
  - [ ] baseline = `GIPC` 时：`Legacy + GPU MAS`
  - [ ] baseline = `StiffGIPC(SRBK)` 时：`SRBK + GPU MAS`
  - [ ] baseline = `StiffGIPC(CEMAS+SRBK)` 时：`SRBK + CEMAS`

---

## 8. benchmark CLI 实施清单

## 8.1 `gipc_bench` 参数

- [ ] 支持单场景运行：
  - [ ] `--baseline`
  - [ ] `--scene`
  - [ ] `--frames`
  - [ ] `--warmup`
  - [ ] `--output`
  - [ ] `--settings`

- [ ] 支持批量 suite：
  - [ ] `--suite path/to/bench_suite.json`

- [ ] 支持控制复现稳定性的参数：
  - [ ] `--seed`（如需要）
  - [ ] `--frame-start`
  - [ ] `--frame-cap`
  - [ ] `--no-save-surface`

## 8.2 suite 文件格式

建议 `bench_suite.json` 至少包含：

```json
{
  "output_root": "Output/benchmarks",
  "defaults": {
    "frames": 100,
    "warmup": 10,
    "settings": "Assets/scene/parameterSetting.txt"
  },
  "runs": [
    {
      "scene": "wrecking_ball",
      "baseline": "GIPC"
    },
    {
      "scene": "wrecking_ball",
      "baseline": "StiffGIPC_SRBK"
    },
    {
      "scene": "wrecking_ball",
      "baseline": "StiffGIPC_CEMAS_SRBK"
    }
  ]
}
```

---

## 9. benchmark 统计与输出实施清单

## 9.1 输出文件

- [ ] `raw.json`
  - [ ] 逐 run
  - [ ] 逐 frame
  - [ ] 可选逐 Newton

- [ ] `summary.csv`
  - [ ] 每行一个 `(scene, baseline)`

- [ ] `meta.json`
  - [ ] git hash
  - [ ] build type
  - [ ] GPU 型号
  - [ ] CUDA 版本
  - [ ] 命令行参数

## 9.2 summary.csv 最低列要求

- [ ] `scene`
- [ ] `baseline`
- [ ] `frames`
- [ ] `warmup`
- [ ] `avg_Hess_ms`
- [ ] `avg_LSolver_ms`
- [ ] `avg_LineS_ms`
- [ ] `avg_Misc_ms`
- [ ] `avg_TimeTot_ms`
- [ ] `avg_newton`
- [ ] `avg_cg`
- [ ] `avg_contact_pairs`
- [ ] `std_TimeTot_ms`

## 9.3 原始统计建议字段

- [ ] `frame_id`
- [ ] `time_hess_ms`
- [ ] `time_lsolver_ms`
- [ ] `time_ccd_raw_ms`
- [ ] `time_lines_ms`
- [ ] `time_post_ls_raw_ms`
- [ ] `time_tot_ms`
- [ ] `time_misc_ms`
- [ ] `newton_count`
- [ ] `cg_total`
- [ ] `contact_pairs_avg`
- [ ] `contact_pairs_max`

---

## 10. 代码内计时实施清单

## 10.1 保留现有 `cudaEvent` 主链路

- [ ] 不依赖当前 `GlobalTimer` 作为 benchmark 主结果
- [ ] 继续使用 `solve_subIP()` / `IPC_Solver()` 中的显式 `cudaEvent`

## 10.2 调整汇总位置

- [ ] 在每 frame 结束时直接生成结构化统计对象
- [ ] 不再只写 `timeCost.txt`
- [ ] 将现有 `time0..time4` 明确映射为命名字段

## 10.3 `Misc` 计算

- [ ] `Misc = TimeTot - Hess - LSolver - LineS`
- [ ] 若 `Misc < 0`，记录 warning 并保存 raw timing 供排查

## 10.4 可选二级 breakdown（非首版必做）

- [ ] `LSolver` 内再细分：
  - [ ] `assembly`
  - [ ] `convert`
  - [ ] `preconditioner_assemble`
  - [ ] `pcg_spmv`
  - [ ] `pcg_preconditioner_apply`
  - [ ] `pcg_dot`
  - [ ] `pcg_axpby`

说明：

- 这一层建议使用新的低扰动计时方案
- 不建议直接启用当前 `Timer` 深打到 PCG 热路径

---

## 11. viewer 与 benchmark 解耦清单

- [ ] `gipc_bench` 中不得出现：
  - [ ] `glutInit`
  - [ ] `glutMainLoop`
  - [ ] `GLEW`
  - [ ] `OpenGL`

- [ ] `gipc_viewer` 中仅复用公共 bootstrap 和 solver
- [ ] 任何截图、表面导出、屏幕操作都不能成为 benchmark 默认行为

---

## 12. 验证清单

## 12.1 构建验证

- [ ] 本地能同时编过：
  - [ ] `stiffgipc_core`
  - [ ] `gipc_viewer`
  - [ ] `gipc_bench`

- [ ] 远程服务器无图形环境下能直接运行：
  - [ ] `./gipc_bench --scene wrecking_ball --baseline GIPC ...`

## 12.2 baseline 一致性验证

- [ ] 同一 scene 下三组 baseline 使用相同：
  - [ ] dt
  - [ ] Newton tolerance
  - [ ] PCG threshold
  - [ ] 材质参数
  - [ ] 帧数

- [ ] 只允许 backend 不同，不允许场景参数偷偷变化

## 12.3 指标合理性验证

- [ ] `TimeTot >= Hess + LSolver + LineS`
- [ ] `Misc >= 0`（允许极小浮动）
- [ ] `avg_cg > 0`
- [ ] `avg_newton > 0`

## 12.4 场景 smoke test

- [ ] `cloth_bunny`
- [ ] `wrecking_ball`
- [ ] `mat_twist`
- [ ] `box_pile`

每个场景至少验证：

- [ ] `GIPC`
- [ ] `StiffGIPC(SRBK)`
- [ ] `StiffGIPC(CEMAS+SRBK)`

---

## 13. 风险与回退方案

## 13.1 风险：legacy GIPC 路径恢复不完整

处理：

- [ ] 优先从 `git show a3479e7:...` 恢复旧路径
- [ ] 若 assembly 旧路径恢复困难，先把 `SpMV` 与 `MAS` 独立切换做好
- [ ] 在 benchmark 输出中明确标注“legacy fidelity”状态

## 13.2 风险：`gl_main.cu` 抽离过程中引入功能回归

处理：

- [ ] 先做“搬运不改逻辑”
- [ ] viewer 行为保持不变后，再接 benchmark

## 13.3 风险：计时被同步/打印污染

处理：

- [ ] benchmark 模式关闭冗余 `stdout`
- [ ] 不在热路径里开启高频 `Timer`
- [ ] 避免每步都写磁盘

## 13.4 风险：未来接 `gm-100` 时耦合过深

处理：

- [ ] benchmark 只依赖 manifest，不依赖 spreadsheet 本身
- [ ] 将任务元数据与仿真资产描述分离

---

## 14. v1 完成标准

满足以下条件即可视为 v1 完成：

- [ ] 新增 `gipc_bench` 成功构建
- [ ] 无头服务器可运行
- [ ] 三组 baseline 可运行切换
- [ ] 四个内置场景可跑通
- [ ] 输出 `raw.json + summary.csv + meta.json`
- [ ] `summary.csv` 中包含 `Hess/LSolver/LineS/Misc/TimeTot`
- [ ] viewer 旧功能未被破坏

---

## 15. 推荐实施顺序

建议严格按以下顺序推进：

1. **先拆 CMake target**
2. **再抽公共 bootstrap**
3. **再做 baseline runtime config**
4. **再恢复 legacy GIPC backend**
5. **再接入 headless benchmark main**
6. **最后补 suite/output/report**

不要反过来做。否则很容易出现：

- benchmark main 写好了但 target 冲突
- baseline 名字有了但 backend 仍是编译期锁死
- 输出文件有了但三组 baseline 实际跑的是同一条路径

---

## 16. 备注

- 当前仓库最接近论文口径的内置场景优先级建议：
  1. `wrecking_ball`
  2. `cloth_bunny`
  3. `mat_twist`
  4. `box_pile`

- `gm-100` 当前只适合作为将来任务索引，不适合作为 v1 benchmark 输入源。
