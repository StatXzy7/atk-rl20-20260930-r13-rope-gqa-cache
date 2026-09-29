# RoPE + GQA 微型自回归 Decoder（含 prefill / 逐 token / 分块 KV cache）

一个从零实现的 decoder-only 微型 Transformer，**不调用任何现成模型库的 cache
实现、不下载任何权重**。仅依赖 `torch`（张量运算）和 `pytest`（测试）。

- 多查询变体：MHA（`n_kv = n_q`）、MQA（`n_kv = 1`）、一般 GQA（`1 < n_kv < n_q`）。
- 三条缓存路径共用同一套 RoPE 规则与同一套 attention 内核：
  `prefill`、逐 token `decode`、`chunked_decode`。
- 分页 KV cache + 引用计数 + 尾块 copy-on-write，支持前缀分叉、安全释放、
  容量不足时整批原子失败。
- 每一步缓存 logits 都与独立的“整个有效前缀从零重算”参照一致（误差为 fp32
  舍入量级，约 `1e-7`）。

## 目录

```
mini_gqa/
  rope.py      # RoPE：偶数旋转维度、频率公式、绝对位置
  model.py     # ModelConfig、TinyModel、GQA masked attention、参数指纹
  cache.py     # 分页 K/V、引用计数、COW、容量预留、字节统计
  runtime.py   # prefill / decode / chunked_decode / reference_logits
tests/         # 55 个自动化测试
```

## RoPE 的完整定义（prefill 与 decode 共用同一规则）

记每个 attention head 的特征维度为 `d_head`（**必须为偶数**）。把它切成
`d_head/2` 个旋转对 `(x_0,x_1),(x_2,x_3),…`（第 `2k` 维与第 `2k+1` 维配对）。

- **偶数旋转维度**：每个旋转对恰好作用于 2 个坐标，故参与旋转的子维度数是偶数；
  `d_head` 为奇数会在构造时直接报错。
- **频率公式**：对旋转对编号 `k = 0 … d_head/2-1`，
  `θ_k = base^(-2k / d_head)`，默认 `base = 10000`。
- **绝对位置编号**：一个相对序列起点偏移 `base_pos`、在当前请求内序号为 `t`
  的 token，其角度为 `φ_{t,k} = (base_pos + t)·θ_k`。decode 的位置接着
  prefill 停下的地方继续编号（用的是逻辑序列中的绝对位置，而不是“本请求第几
  个 token”）；`base_pos` 允许序列逻辑上从非 0 位置开始。
- **旋转**：
  ```
  x'_{2k}   = cos(φ)·x_{2k}   - sin(φ)·x_{2k+1}
  x'_{2k+1} = sin(φ)·x_{2k}   + cos(φ)·x_{2k+1}
  ```
- **缓存内容**：写入 cache 的 K 是**已经按其正确绝对位置旋转过的 K**，与它
  对应的 V 原样保存；缓存的 K 不会被二次旋转。每次请求新生成的 Q 和新 K 用
  同一个 `RoPE.rotate(..., 绝对位置)` 旋转，参照重算也调用同一函数。

注意一个标准 RoPE 性质：attention 分数只依赖 `(pos_q - pos_k)·θ` 的相对角度，
因此把整条序列整体平移一个常数 `base_pos` 不会改变最终 logits（平移不变性）。
测试 `test_position_offset_baked_into_cached_k` 不依赖 logits，而是直接核验
缓存中旋转后的 K 等于 `R(base_pos+i)·K_raw`，以证明偏移确实被消费。

## 可见性（causal）规则

布尔可见矩阵 `allow[b, i, j]` 决定查询 `i` 能否看键 `j`，三条路径统一在
`runtime._extend` 中构造：

| 路径 | 历史 cache | 本请求内 |
|------|-----------|----------|
| `prefill`（空 session） | 无 | 严格下三角：`j ≤ i` |
| `decode`（每序列 1 token） | 全部可见 | 自身 |
| `chunked_decode`（宽 `n`） | 全部可见 | 严格下三角：`j ≤ i`，**看不到本块未来 token** |

- 分块 decode 的查询能看到：全部历史 cache **和** 本块中已经发生的 token；
  看不到本块未来 token（`test_*chunked*` 与泄漏反例覆盖）。
- padding 行（`valid=False`）不参与 attention，其输出在最终隐藏态上被置零；
  因此不等长 batch 不会相互串味。
- 每个 session 独立维护：有效长度 `length`、绝对位置 `base_pos`、可见宽度
  `lens`，以及按块组织的 K/V。

## GQA 约束

`n_q_heads` 必须是 `n_kv_heads` 的**正整数倍**，否则 `ModelConfig` 直接报错。
KV head 按连续组映射：`q_head i → kv_head i // (n_q // n_kv)`，通过
`repeat_interleave` 扩展。测试覆盖 MHA / MQA / 一般 GQA 三种配置。

## 参数变化后旧 cache 失效

`TinyModel.fingerprint()` 对结构配置 + 每个参数张量的精确字节求 SHA-256。
- session 创建时记录当时的指纹；每次 `reserve_extend` 都核对当前模型指纹。
- 权重被更新（训练、重新初始化、手动改动）或换了模型版本后，旧 session 的
  K/V 是在旧参数下旋转/投影得到的，继续使用会抛 `StaleCacheError`，绝不跨
  版本复用。重新初始化同种子模型得到相同指纹。

## 分页布局、所有权与 copy-on-write

- K、V 各自是一块连续张量，形状
  `[num_blocks, block_size, n_layers, n_kv_heads, head_dim]`。
- 序列是一串有序 block id。**满块冻结、永不变更**，因此可被所有共享该前缀的
  分支永久共享；只有当前的**部分尾块**可变。
- `fork_session` 只复制元数据并把共享块引用计数 +1，零字节拷贝。
- 任一分支首次向“仍被他人共享的部分尾块”写入时，先把该尾块拷贝到新块
  （copy-on-write），再在私有副本上写，并对旧块减引用；已冻结的满块始终
  共享。因此两条分支的后续写入互不污染，释放任一分支不会破坏另一条。
- 引用计数归零的块回到空闲链表；`release_session` 幂等。

### 容量不足：显式失败、不部分追加

- 容量以整 block 计，`capacity_tokens` 必须能被 `block_size` 整除。
- 每个 batch 在任何写入之前先统一 `reserve_extend`：把“新增长块需求 + 尾块
  COW 需求”加总，若空闲块不足，**在任何分配/拷贝发生之前**抛
  `CacheCapacityError`。因此多序列批量追加要么全部生效，要么完全不动
  （`test_batch_capacity_failure_is_atomic`、
  `test_cow_failure_at_fork_boundary_is_atomic` 覆盖）。

## 缓存张量字节

每 token（跨所有层）保存 K 与 V：
```
bytes_per_slot = n_layers * n_kv_heads * head_dim * elem_bytes * 2   # K 和 V
capacity_bytes = num_blocks * block_size * bytes_per_slot
```
- `capacity_bytes` 与 `k_backend/v_backend.numel()*element_size()` 之和一致
  （测试实际核验张量字节）。
- `logical_used_bytes` 按每个存活 session 的有效 token 数求和（共享前缀按每个
  拥有者各计一次）；`physical_used_bytes` 按当前被引用的物理整 block 计
  （向上取整到 block）。分叉只增加逻辑字节、不增加物理字节，直到发生 COW。

## 复杂度

记历史长度 `L`、本请求宽 `T`、head 数 `h_q/h_kv`、维度 `d`、FFN 宽 `F`、层数
`ℓ`、batch `B`：

- **prefill**：时间 `O(B·T²·d·h_q·ℓ + B·T·d²·ℓ·(FFN))`，即标准
  `O(T²)` attention；当批新增 KV 为 `O(B·T·ℓ·h_kv·d)`。
- **逐 token decode**：每步 attention 对全部历史做内积，时间
  `O(B·(L·d·h_q + d² + F·d)·ℓ)`；新增 KV 为 `O(B·ℓ·h_kv·d)`。
- **分块 decode（块宽 c）**：本块 attention 为 `O(c²)`，历史部分为
  `O(c·L)`，合计 `O(B·c·(L+c)·d·h_q·ℓ)`；`c=1` 即逐 token decode，
  `L=0` 且一次给整序列即 prefill。
- **GQA 存储**：KV 相对 MHA 按 `h_kv/h_q` 比例缩小；MQA 时每 token每层只有
  1 个 KV head。
- **分叉**：`fork` 本身 `O(1)` 元数据操作（零 KV 拷贝）；首次写入共享尾块时
  付出一次 `O(block_size·ℓ·h_kv·d)` 的 COW 拷贝；冻结满块永不复制。
- **参照重算 `reference_logits`**：每次 `O((前缀+续接)²)`，是有意保留的高
  成本独立基准，仅用于测试中的逐点比对。

## 运行测试

```
python3 -m pytest tests/ -q
```

测试矩阵（59 项）覆盖：

- **三种 head 配置** MHA / MQA / 一般 GQA（`test_cache_paths.py` 参数化）。
- **不等长 batch**：不同 prefill 长度、padding 列置零、独立有效长度/位置、
  批量逐 token 与不等宽分块 decode。
- **不同分块**：`(2,2,2)`、`(3,2)`、`(1,4,1)`、`(7,1)`、`(8,)` 与逐 token
  路径及全前缀重算三方一致。
- **前缀分叉**：共享满块、尾块 COW、物理内容隔离、释放一条不影响另一条。
- **容量边界**：恰好装满、差一块失败、批量原子失败、COW 容量失败不回滚半成品。
- **位置偏移**：直接核验缓存 K 的绝对位置角度；decode 位置续接 prefill。
- **字节核验**：fp16/fp32 实际张量字节、逻辑/物理用量、分叉不增物理字节。
- **失效**：权重变化与跨模型版本均抛 `StaleCacheError`。

### 两个以上典型错误导致 logits 漂移的反例

`tests/test_drift_counterexamples.py` 用故障注入开关演示并断言“正确路径匹配
参照、错误路径显著漂移”：

1. **decode 把位置重新从 0 编号**（`_restart_position`）：历史 K 保持正确角度、
   新 Q/K 却从 0 开始，所有 q–k 相对角全错；前缀越长漂移越大。
2. **RoPE off-by-one**（`_position_bug`）：decode 在 `length-1` 而非 `length`
   处旋转新 token。
3. **分块 decode 漏掉块内 causal 掩码**（`_leak_future`）：块内 token 看到自己
   的未来 token；最后一个 token 无未来可看因而不受影响，前面的行显著漂移。
4. **GQA head 分组映射错误**（`_group_mode="rotated"`）：KV head 被循环错位地
   分配给 Q head，每个 Q head 读到错误的 K/V 流。
5. **新 K 未应用 RoPE 就入缓存**（`_rotate_new_k=False`）：旋转过的 Q 与未旋转
   的新 K 做点积，污染 attention 分布。

## 使用示例

```python
from mini_gqa import ModelConfig, TinyModel, KVCachePool, prefill, decode, reference_logits

cfg = ModelConfig(vocab_size=257, d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2)
model = TinyModel(cfg, seed=0).eval()
pool = KVCachePool(model, capacity_tokens=4096, block_size=16)

s = pool.create_session(base_pos=0)
prefill(pool, s, prefix_ids)                 # 因果预填
ref = reference_logits(model, s, next_token) # 在 decode 提交前调用：整前缀独立重算
logits, _, _ = decode(pool, s, next_token)   # 逐 token，与 ref 一致

branch = pool.fork_session(s)                # 零拷贝分叉
decode(pool, branch, alt_token)              # 与 s 的写入互不污染
pool.release_session(branch)                 # 释放不影响 s
```
