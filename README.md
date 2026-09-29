# mini-decoder：RoPE + GQA 的微型自回归 decoder 与分页 KV cache

从零实现（仅依赖 PyTorch 的张量运算，**不使用任何现成模型库的 cache 实现，
不下载权重**），覆盖 prefill、逐 token decode、分块 decode 三条 KV cache 路径，
并保证：任意时刻 cached logits 与“独立重算整个有效前缀”的参照逐位一致
（float64 下误差 < 1e-9）。

## 代码布局

| 文件 | 内容 |
| --- | --- |
| `mini_decoder/rope.py` | RoPE 唯一定义来源（prefill/decode 共用） |
| `mini_decoder/model.py` | GQA 注意力、decoder 层、参数指纹 |
| `mini_decoder/cache.py` | 分页 KV cache：块池、引用计数、copy-on-write、容量检查 |
| `mini_decoder/engine.py` | 批量引擎：prefill / decode / 分块 decode 统一入口 |
| `mini_decoder/reference.py` | 独立参照：整段有效前缀重算（不用 cache） |
| `tests/` | 82 个自动化测试（pytest） |

## RoPE 定义（完整）

- `head_dim = d` 必须为**偶数**，维度两两成对：`(2i, 2i+1)`，`i = 0 .. d/2-1`。
- 频率（逆波长）：`theta_i = rope_base ** (-2i / d)`（默认 `rope_base = 10000`）。
- **绝对位置编号** `m`：token 在其所属序列**有效 token 流**中的 0 基下标，
  再加上可选的每序列 `pos_offset`。padding 槽位不分配位置、不进入 cache。
  prefill、逐 token decode、分块 decode 使用**同一套**编号规则——一个 token
  的旋转只取决于 `m`，与它被切成哪一块无关。
- 旋转角 `phi_i = m * theta_i`，旋转向量 `x`（每个 head 独立）：

  ```
  x'_{2i}   = x_{2i} * cos(phi_i) - x_{2i+1} * sin(phi_i)
  x'_{2i+1} = x_{2i} * sin(phi_i) + x_{2i+1} * cos(phi_i)
  ```

  等价于复数乘法 `(x_{2i} + j·x_{2i+1}) · exp(j·phi_i)`。
- **cache 存的是已按绝对位置旋转好的 K 和对应的 V**；decode 不会重旋转旧 K。
  （`tests/test_engine.py::test_cache_stores_rotated_k` 直接核验这一点。）

## GQA / MHA / MQA

- query head 数必须是 KV head 数的整数倍：`n_heads % n_kv_heads == 0`
  （构造时校验，违反即 `ValueError`）。
- 分组规则：query head `h` 使用 KV head `h // (n_heads // n_kv_heads)`（连续分组）。
- MHA = `n_kv_heads == n_heads`，MQA = `n_kv_heads == 1`，其余为一般 GQA。
  三种配置（含 group size 3）全部在测试中参数化覆盖。

## Batch、padding 与 causal 可见性

- batch 采用右 padding；每条序列独立维护有效长度 `len_b`、已缓存长度
  `past_b`、位置偏移 `off_b`。
- query 位置 `q_pos[b, s] = off_b + past_b + s`；key 位置对历史段为
  `off_b + j`，对当前块为 `off_b + past_b + s'`。
- 可见性：`allowed[b, s, j] = key_valid[b, j] & (key_pos[b, j] <= q_pos[b, s])`，
  即**绝对位置空间里的 causal 掩码**再交集“非 padding”。分块 decode 中，
  块内第 `i` 个 query 只能看到历史 cache 与块内 `0..i`，看不到块内未来 token。
- padding 行（无效 query）的输出被丢弃、不写入 cache；其掩码被强制为只
  attend 到 key 0 以避免产生 NaN，不影响任何有效行。

## KV cache 设计

### 分页存储与字节核算

- 池共 `num_blocks` 个块，每块 `block_size` 个 token 槽，块 id 在所有层共享：
  ```
  block_bytes = n_layers · 2(K,V) · block_size · n_kv_heads · head_dim · itemsize
  pool.nbytes = num_blocks · block_bytes        # 与实际张量存储逐字节相等（有测试核验）
  ```
- 序列 = 块表（有序 block id 列表）+ `length`。token `t` 位于
  `table[t // block_size]` 的 `t % block_size` 槽（每层同址）。
- `sequence_bytes`（逻辑，共享块按持有者各计一次）与 `allocated_bytes`
  （物理，共享块只计一次）分开上报。

### 所有权与 copy-on-write（前缀分叉）

- 每个块有引用计数。`fork` 只复制块表并给所有块 refcount+1——O(#blocks)，
  **零数据拷贝**；两条分支共享整个前缀。
- 序列只能写自己独占的块：向“半满且仍被共享”的尾块追加时，先把该块
  拷贝成私有块（COW）再写入；已满的共享前缀块事实上不可变。
- `free` 只递减 refcount，归零才回收块——**释放一个分支不会破坏另一个**。
- 两条分支的后续写入因此互不污染（有测试逐字节核验共享块内容不变）。

### 容量语义：明确失败、绝不部分追加

- `required_blocks(seq, n)` 精确计算一次追加需要的新块数（含可能的 COW 拷贝）。
- `append` 在任何写入之前检查空闲块，不足即抛 `CacheCapacityError`，
  状态完全不变；引擎在整个 batch 维度先做预检，保证**跨序列也是全有或全无**。

### 参数变更 → 旧 cache 失效

- `MiniDecoder.fingerprint()` 是配置 + 全部参数张量的 SHA-256。
- cache 在创建时记录 `model_version`；引擎在每次 prefill/decode/fork 前重新
  计算指纹并比对，不一致即抛 `CacheInvalidError`——**旧 cache 不能跨模型
  版本复用**。更换指纹后新建 cache 可正常工作。

## 复杂度（每层；`d = d_model`，`L` = 已缓存长度，`S` = 块长，`H`/`Hkv` = q/kv head 数，`D` = head_dim）

| 路径 | 注意力 FLOPs | 投影 FLOPs | cache 读写 |
| --- | --- | --- | --- |
| prefill（L 个 token） | `O(L²·H·D)` | `O(L·d²)` | 写 `O(L·Hkv·D)` |
| 逐 token decode | `O(L·H·D)` / token | `O(d²)` / token | 读 `O(L·Hkv·D)`，写 `O(Hkv·D)` |
| 分块 decode（S 个） | `O(S·(L+S)·H·D)` | `O(S·d²)` | 读 `O(L·Hkv·D)`，写 `O(S·Hkv·D)` |
| 参照重算（每步） | `O((L+t)²·H·D)` | `O((L+t)·d²)` | 不用 cache |

- 逐 token decode n 步：cache 路径 `O(n·L·H·D)`，参照重算 `O(n·L²·H·D)`——
  cache 把前缀的 K/V 投影与注意力从历史重算中消掉。
- 显存：`length · Hkv · D · 2 · n_layers · itemsize`，按块取整；
  GQA 通过 `Hkv < H` 直接按比例缩小 cache。
- `fork`：`O(#blocks)` 元数据，无数据拷贝；COW 每次追加均摊 `O(1)` 个块拷贝；
  容量检查 `O(1)`。

## 反例测试（典型错误 → logits 漂移）

`tests/test_drift.py` 实现了三个经典错误变体，断言它们相对参照**显著漂移**
（实测漂移 1.6e-2 ~ 1.6e-1，阈值 1e-3），同时正确实现与参照一致（< 1e-9）：

1. **位置重置**：decode 时新 token 的 RoPE 位置从 0 重新计，而非从前缀长度继续；
2. **causal off-by-one**：分块 decode 中 query `i` 多看到块内第 `i+1` 个未来 token；
3. **padding 泄漏**：不等长 batch 中，短序列的 softmax 未屏蔽长序列补零的 cache 槽。

## 运行测试

```bash
python3 -m pytest tests/ -q     # 82 passed
```

覆盖：MHA/MQA/GQA（含 group size 3）、不等长 batch、5 种 decode 分块、
ragged chunk、前缀分叉与释放、COW 容量记账、容量边界全有/全无、
位置偏移、参数变更失效、cache 张量字节核验、RoPE 数学性质
（复数定义等价、范数保持、相对位置性质 `<R_m q, R_n k> = <R_{m-n} q, k>`）。

## 范围说明

单进程、单线程，聚焦数值与状态正确性；不实现高并发调度、连续批处理或
CUDA kernel。`Engine` 与 `KVCacheManager` 的接口与这些扩展兼容（块表 +
引用计数即 vLLM 式分页注意力的单机版）。
