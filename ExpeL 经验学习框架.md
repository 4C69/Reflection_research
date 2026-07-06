# 1. 一句话概括 ExpeL

**ExpeL = 经验池 + 反思抽象 + 相似经验检索。**

更具体地说，它把 Agent 的经验分成两类记忆：

1. **具体经验记忆**：过去成功完成任务的完整轨迹，类似 episodic memory。
2. **抽象规则记忆**：从多个成功/失败轨迹中总结出来的高层经验，类似 semantic/procedural memory。

测试时，Agent 不是重新训练模型，而是把这两类记忆加入上下文：

$$
\text{Prompt} =
\text{任务描述}
+
\text{抽象 insights}
+
\text{相似成功轨迹}
+
\text{当前任务轨迹}
$$

所以它本质上是一个 **基于自然语言记忆的非参数学习方法**。

---

# 2. 论文要解决什么问题？

论文认为，当前 LLM Agent 有两类主流增强方式：

第一类是 **微调模型参数**。例如用大量交互数据或人工标注数据训练 Agent，使模型适应特定任务。问题是：

- 成本高；
- 需要访问模型权重；
- GPT-4、Claude 这类强模型通常只能 API 调用；
- 微调可能损害模型原有泛化能力。

第二类是 **prompt-based 方法**，比如 ReAct、Reflexion。它们不改参数，只通过 prompt 提升推理和规划能力。问题是：

- 受限于上下文窗口；
- 通常只能利用少量 few-shot 示例；
- Agent 很难真正“记住”过去做过的大量任务；
- Reflexion 主要是在**同一个任务的多次尝试中改进**，而不是跨任务积累经验。

ExpeL 想解决的是：**如何让 Agent 像人做练习题一样，从一批训练任务中积累经验，然后在新任务中受益？**

论文中用 Tom Mitchell 的机器学习定义作为引子：如果系统在任务 $$T$$ 上的性能 $$P$$ 随经验 $$E$$ 增加而提升，就可以说它在学习。ExpeL 的目标就是让 LLM Agent 满足这个定义，但不通过梯度更新，而是通过外显经验记忆和自然语言规则学习。`ExpeL.pdf`

---

# 3. ExpeL 的整体框架

论文图 1 给出了 ExpeL 的三阶段流程：

## 阶段一：Experience Gathering，经验采集

Agent 在训练任务集合上不断尝试任务。每个任务最多尝试 $$Z$$ 次。

如果失败，就用 Reflexion 生成失败反思，然后下一次尝试时把反思加入上下文。无论成功还是失败，轨迹都会被存进经验池。

## 阶段二：Insight Extraction，经验抽象

从经验池中提取两类信息：

1. 比较同一任务的失败轨迹和成功轨迹，总结失败原因和正确做法。
2. 分析一组成功轨迹，总结共性的好策略。

这些总结出来的自然语言规则就是 insights。

## 阶段三：Task Inference，任务推理

测试时，Agent 只有一次尝试机会。它会：

1. 使用全部或部分 learned insights；
2. 从经验池中检索与当前任务最相似的成功轨迹；
3. 把这些内容作为 prompt 的增强信息；
4. 用 ReAct 形式执行当前任务。

这就是 ExpeL 的主线：**训练阶段积累经验，离线阶段抽象规则，测试阶段调用经验。**

---

# 4. 阶段一：经验采集是怎么做的？

ExpeL 不是直接拿人工数据训练，而是让 Agent 自己在训练任务中试错。

对于第 $$n$$ 个任务 $$t_n$$，第 $$z$$ 次尝试的轨迹记为：

$$
\tau_{n,z}
$$

初始反思内容为空：

$$
\nu_{n,0} = ""
$$

Agent 使用 ReAct 作为基础规划方法。每一步根据当前轨迹、人工 few-shot 示例、已有反思生成动作：

$$
a_i =
\mathrm{LLM}_{\mathrm{ReAct}}
(
a_i \mid \tau_{n,z}, F_{\mathrm{manual}}, \nu_{n,z}
)
$$

如果任务失败，ExpeL 会调用反思模型：

$$
\nu_{n,z+1}
=
\mathrm{concat}
(
\nu_{n,z},
\mathrm{LLM}_{\mathrm{reflect}}(\tau_{n,z})
)
$$

然后下一次尝试时带着这个反思继续做同一个任务。

这里需要注意：**Reflexion 在 ExpeL 中主要是经验采集工具**。它帮助 Agent 多试几次，从而获得更多成功轨迹和成功/失败对比样本。ExpeL 的重点不是部署阶段反复尝试，而是训练阶段收集经验，测试阶段一次完成任务。`ExpeL.pdf`

---

# 5. 阶段二：ExpeL 如何从经验中学习？

这是论文最重要的部分。

ExpeL 的经验学习分成两条线：

---

## 5.1 具体经验召回：把成功轨迹当作 demonstrations

论文认为，人在面对新任务时，会回忆过去做过的相似任务。ExpeL 也这么做。

它把成功轨迹存入经验池，然后用向量检索找出和当前任务最相似的成功经验。

论文实现中使用：

- Faiss 作为向量库；
- all-mpnet-base-v2 作为 embedding 模型；
- kNN 检索；
- 相似度使用最大内积；
- 只检索成功轨迹作为 few-shot examples。

可以抽象成：

$$
F_{\mathrm{similar}}
=
\operatorname{TopK}_{\tau \in B_{\mathrm{success}}}
\langle
E(t_{\mathrm{eval}}),
E(t_{\tau})
\rangle
$$

其中：

- $$B_{\mathrm{success}}$$ 是成功轨迹集合；
- $$E(\cdot)$$ 是 embedding 函数；
- $$t_{\mathrm{eval}}$$ 是当前测试任务；
- $$t_{\tau}$$ 是轨迹 $$\tau$$ 对应的任务描述。

这一步解决的是：**在新任务中提供相似案例，让 Agent 模仿过去成功做法。**

---

## 5.2 抽象经验学习：从轨迹中提取 insights

仅仅存成功轨迹还不够，因为具体轨迹只能帮助相似任务。如果任务变化较大，Agent 还需要更抽象的策略。

所以 ExpeL 设计了一个 insight extraction 过程。

它给 LLM 一个已有规则集合：

$$
\hat{\iota} = \{\iota_1, \iota_2, \dots\}
$$

然后让 LLM 根据新经验对规则集合做操作。可用操作有四种：

| 操作 | 含义 |
|---|---|
| ADD | 添加一条新 insight |
| EDIT | 修改已有 insight |
| UPVOTE | 认为已有 insight 有用，提高重要性 |
| DOWNVOTE | 认为已有 insight 不好或冲突，降低重要性 |

每条新 insight 初始重要性计数为 2。

如果后续被 UPVOTE 或 EDIT，计数增加；如果被 DOWNVOTE，计数减少；如果计数降到 0，就删除。

这相当于一个非常简单的 **自然语言规则进化机制**：

$$
\hat{\iota}_{t+1}
=
\mathrm{LLM}_{\mathrm{insights}}
(
\text{new experiences},
\hat{\iota}_{t}
)
$$

这个设计很有意思，因为它不是让 LLM 一次性总结所有经验，而是让 insights 在多批经验中逐渐被添加、修改、强化或淘汰。

---

# 6. insight extraction 的输入是什么？

论文用了两种输入。

## 第一种：同一任务的失败/成功对比

如果某个任务先失败，后来成功，那么 ExpeL 会把失败轨迹和成功轨迹放在一起，让 LLM 比较：

- 失败轨迹哪里做错了；
- 成功轨迹做对了什么；
- 哪些经验可以迁移到未来任务。

这种方式特别适合总结“避免错误”的规则。

例如：

> 如果搜索不到答案，不要立刻回答 Unknown，而要检查已有 observations 中是否已经包含答案。

这类 insight 本质上是从 failure case 中抽象出的纠错规则。

## 第二种：多个成功轨迹的共性总结

ExpeL 还会把若干条成功轨迹组成一个列表，让 LLM 总结共同的成功模式。

这种方式特别适合总结“最佳实践”。

例如在 WebShop 中，可能总结出：

- 搜索商品时要保留核心属性；
- 比较商品时要同时检查颜色、尺寸、价格、品牌；
- 不要只看标题，还要检查选项页面。

这类 insight 更像是 general policy guidance。

---

# 7. 阶段三：测试时 ExpeL 怎么用这些经验？

测试时，ExpeL 不再多次尝试任务，而是一次完成。

每个测试任务 $$t_m$$ 到来时，ExpeL 做三件事：

1. 把 learned insights 加入任务说明；
2. 从经验池中检索 top-k 相似成功轨迹；
3. 用 ReAct 格式执行当前任务。

也就是：

$$
a_i
=
\mathrm{LLM}_{\mathrm{ExpeL}}
(
a_i
\mid
\tau_m,
F_{\mathrm{similar}},
\hat{\iota}
)
$$

其中：

- $$\tau_m$$ 是当前测试任务已经产生的轨迹；
- $$F_{\mathrm{similar}}$$ 是检索出来的相似成功轨迹；
- $$\hat{\iota}$$ 是抽象 insights。

论文图 3 展示了测试 prompt 的结构。白色区域是原始 ReAct prompt，紫色区域是 ExpeL 新增的部分，包括 extracted insights 和 retrieved in-context examples。`ExpeL.pdf`

---

# 8. ExpeL 和 Reflexion 的区别

这是理解这篇论文的关键。

| 维度 | Reflexion | ExpeL |
|---|---|---|
| 主要学习范围 | 同一个任务内部 | 多个任务之间 |
| 经验来源 | 当前任务失败轨迹 | 训练任务集合中的成功/失败轨迹 |
| 学到的东西 | 针对当前任务的反思 | 跨任务可复用 insights + 成功案例 |
| 测试时是否重试 | 通常需要多轮尝试 | 默认一次尝试 |
| 记忆形式 | 当前任务的 verbal reflection | 经验池 + 抽象规则 |
| 更像什么 | 在线自我修正 | 离线经验蒸馏 / 记忆巩固 |

所以可以这样理解：

**Reflexion 是 intra-task self-improvement，ExpeL 是 inter-task experiential learning。**

Reflexion 更像“这道题我刚才做错了，我再试一次”；ExpeL 更像“我之前做过很多题，总结了经验，现在考试时直接用这些经验”。

---

# 9. 和你研究的“智能体记忆反思”有什么关系？

从记忆系统角度看，ExpeL 非常典型。

它不是单纯的 prompt trick，而是一个 **记忆系统 + 反思机制**：

## 9.1 经验池是 episodic memory

每条轨迹都保留了任务执行过程：

$$
\tau = \{o_0, a_0, o_1, r_1, a_1, \dots\}
$$

它记录了 Agent 在具体环境中的行为、观察和结果。

这类记忆是具体的、情境化的，类似 episodic memory。

## 9.2 insights 是 semantic/procedural memory

从多条轨迹中总结出的规则是抽象的：

- “搜索时应该……”
- “如果某个动作没有进展，应该重新评估……”
- “答案可能已经在已有 observation 中……”

这类记忆不是某个具体任务，而是可迁移的策略。

它更像 semantic memory 或 procedural memory。

## 9.3 insight extraction 是事后反思

ExpeL 的核心反思发生在收集经验之后。它拿已有轨迹做分析、对比、压缩和规则更新。

所以如果按照你之前区分的反思类型，ExpeL 主要属于：

**事后反思 / 离线反思 / 经验巩固型反思。**

不过它在经验采集阶段也用了 Reflexion，因此也包含一定的在线试错反思，但那不是 ExpeL 的主要贡献。

---

# 10. 实验设置

论文在 4 个文本任务环境上做实验：

| 数据集 / 环境 | 任务类型 |
|---|---|
| HotpotQA | 多跳问答，需要使用 Wikipedia Docstore API |
| ALFWorld | 家庭环境交互任务，需要多步决策 |
| WebShop | 在线购物任务，需要搜索、比较、购买商品 |
| FEVER | 事实验证任务，用于 transfer learning |

主要 baseline：

- Act：只行动，不显式推理；
- ReAct：Thought + Action + Observation；
- Reflexion：失败后反思并重试；
- Imitation Learning：来自 ReAct 论文的结果；
- ExpeL insights-only；
- ExpeL retrieve-only；
- ExpeL full。

所有 Agent 在测试执行动作时使用的是 `gpt-3.5-turbo-0613`，insight extraction 默认使用 `gpt-4-0613`。实验使用四折验证，指标主要是成功率。`ExpeL.pdf`

---

# 11. 主实验结果怎么理解？

论文图 5 是核心结果。

## 11.1 HotpotQA

大致结果：

| 方法 | 成功率 |
|---|---:|
| Act | 29% |
| ReAct | 28% |
| ExpeL insights-only | 36% |
| ExpeL retrieve-only | 31% |
| ExpeL full | 39% |

HotpotQA 是知识密集型问答任务。它需要分析搜索结果、判断证据、避免过早回答 Unknown。

所以 insights 的作用更明显。因为它提供的是通用推理策略，例如：

- 拆解复杂问题；
- 检查已有 observation；
- 不要过早结束；
- 搜索结果不足时换查询方式。

这说明：**对于推理型任务，抽象规则比具体轨迹更有帮助。**

---

## 11.2 ALFWorld

大致结果：

| 方法 | 成功率 |
|---|---:|
| Imitation Learning | 37% |
| Act | 28% |
| ReAct | 40% |
| ExpeL insights-only | 50% |
| ExpeL retrieve-only | 55% |
| ExpeL full | 59% |

ALFWorld 是具身交互环境，动作空间和物品位置有强烈的任务结构。比如找锅、拿水果、放到某个地方。

在这种环境中，相似成功轨迹非常有用，因为 Agent 可以模仿过去成功的动作序列。

所以 retrieve-only 比 insights-only 更强。

这说明：**对于交互执行型任务，具体成功经验非常重要。**

---

## 11.3 WebShop

大致结果：

| 方法 | 成功率 |
|---|---:|
| Imitation Learning | 29% |
| Act | 34% |
| ReAct | 35% |
| ExpeL insights-only | 37% |
| ExpeL retrieve-only | 38% |
| ExpeL full | 41% |

WebShop 同时需要：

- 搜索策略；
- 商品属性比较；
- 页面导航；
- 选项点击；
- 价格与属性约束检查。

所以抽象 insights 和具体成功轨迹都重要。full ExpeL 最好，但相比 Reflexion 多次重试还有差距。

论文也承认，WebShop 上 ExpeL 仍有提升空间。`ExpeL.pdf`

---

# 12. ExpeL 和 Reflexion 的实验比较

论文指出：

- 在 HotpotQA 上，ExpeL 单次尝试接近 Reflexion 三轮尝试；
- 在 ALFWorld 上，ExpeL 单次尝试甚至超过 Reflexion 多轮尝试；
- 在 WebShop 上，ExpeL 低于 Reflexion 多轮尝试。

这说明 ExpeL 的优势是：

**测试阶段不需要反复试错，也能利用训练阶段积累的跨任务经验。**

但如果允许测试阶段也重试，那么 ExpeL 和 Reflexion 可以叠加。

论文表 2 显示，在 ALFWorld 中：

| 方法 | R0 | R1 | R2 | R3 |
|---|---:|---:|---:|---:|
| ReAct + Reflexion | 40.3% | 47.8% | 52.2% | 54.4% |
| ExpeL retrieve-only | 54.5% | 57.5% | 59.7% | 60.4% |
| ExpeL + Reflexion | 59.0% | 60.4% | 63.4% | 64.2% |

这说明：

**ExpeL 和 Reflexion 不是替代关系，而是互补关系。**

ExpeL 提供跨任务经验，Reflexion 提供当前任务内的失败修正。

---

# 13. Transfer Learning 实验

论文还做了一个迁移学习实验：从 HotpotQA 迁移到 FEVER。

这两个任务都使用 Wikipedia Docstore API，但任务目标不同：

- HotpotQA 是多跳问答；
- FEVER 是事实验证。

由于源任务和目标任务不同，不能直接检索 HotpotQA 的成功轨迹来解决 FEVER。因此论文只迁移 insights。

方法是：

1. 先从 HotpotQA 中提取 insights；
2. 给 LLM 少量 FEVER few-shot examples；
3. 让 LLM 把 HotpotQA insights 改写成适合 FEVER 的规则；
4. 测试 FEVER。

结果：

| 方法 | FEVER 成功率 |
|---|---:|
| Act | 58 ± 0.0 |
| ReAct | 63 ± 0.4 |
| ExpeL Transfer w/o Task Demos | 65 ± 1.7 |
| ExpeL Transfer | 70 ± 0.7 |

结论是：

**从源任务中学到的抽象经验可以迁移到目标任务，且少量目标任务 demonstrations 可以帮助 grounding，降低 hallucination。**`ExpeL.pdf`

---

# 14. 消融实验说明了什么？

论文的消融实验很关键。

## 14.1 经验是否重要？

论文比较了：

1. 只用人工 few-shot 提取 insights；
2. 用 ReAct 采集经验，但不使用 Reflexion 重试；
3. 用完整 ExpeL 采集成功/失败经验。

结果表明：

- 只从 few-shot 提取 insights 基本没什么优势；
- 额外采集的经验越多，效果越好；
- 使用 Reflexion 采集到的成功/失败对比数据更有价值。

说明 ExpeL 的性能不是来自 prompt 模板本身，而是来自 **真实交互经验**。

---

## 14.2 insights 怎么提取最好？

HotpotQA 上的结果：

| 方法 | 成功率 |
|---|---:|
| ReAct | 28.0 ± 1.4 |
| Hand-crafted insights | 32.0 ± 1.1 |
| Insights with reflections | 29.0 ± 0.4 |
| GPT-3.5 insights | 32.0 ± 0.4 |
| ExpeL full | 39.0 ± 1.7 |

几个结论：

第一，LLM 自动提取的 insights 比人工手写更好。

第二，把 Reflexion 生成的 reflections 也加入 insight extraction 反而变差。原因可能是 reflections 本身会幻觉或过拟合当前失败，误导规则总结。

第三，GPT-4 提取 insights 明显优于 GPT-3.5，说明 insight extraction 对模型能力敏感。

这个结果对记忆系统研究很重要：**不是所有反思文本都适合写入长期记忆，反思也需要筛选和验证。**

---

## 14.3 检索策略是否重要？

ALFWorld 上比较了三种检索方式：

| 方法 | 成功率 |
|---|---:|
| ReAct | 40.0 ± 0.3 |
| Reasoning similarity | 48.5 ± 2.1 |
| Random sampled | 42.5 ± 0.8 |
| Task similarity / ExpeL | 59.0 ± 0.3 |

结论：

**按任务相似度检索成功轨迹，比随机检索或按当前 reasoning step 检索更好。**

原因是任务相似度更稳定；reasoning similarity 会随着当前轨迹动态变化，可能导致 few-shot 示例不一致，从而影响执行稳定性。

---

# 15. 论文观察到的 emergent abilities

论文还人工分析了 ExpeL 的轨迹，发现一些行为变化。

## 15.1 Hypothesis Formulation & Constraints Adaptation

在 HotpotQA 中，ExpeL 学会在最后阶段重新检查已有 observations，而不是直接回答 Unknown。

这可能来自某条 insight：

> 答案可能已经在已有观察中。

这说明 insights 能改变 Agent 的终止策略。

## 15.2 World Model Belief Update

在 ALFWorld 中，ReAct 原本可能认为锅在抽屉、台面、柜子里；ExpeL 通过经验学到锅更可能在炉灶上。

这说明 ExpeL 的经验可以改变 Agent 对环境的先验。

## 15.3 Self-correction

ExpeL 有时拿错物品后，会把它放回去，然后继续寻找正确物品。

这说明 learned insights 能提升 Agent 的错误恢复能力。

注意，这些能力不是模型参数变化导致的，而是 prompt 中的经验规则和案例引导出来的。`ExpeL.pdf`

---

# 16. ExpeL 的本质贡献

我认为这篇论文的贡献可以概括为三点。

## 16.1 把 Agent 学习从“参数学习”转成“记忆学习”

传统机器学习通常是：

$$
\theta_{t+1}
=
\theta_t
-
\eta \nabla_{\theta} L
$$

ExpeL 不更新参数：

$$
\theta_{t+1}
=
\theta_t
$$

它更新的是外部记忆：

$$
B_{t+1}
=
B_t
\cup
\{\tau_t\}
$$

$$
\hat{\iota}_{t+1}
=
\mathrm{Update}
(
\hat{\iota}_t,
\tau_t
)
$$

也就是说，学习发生在 memory 层，而不是 model weights 层。

## 16.2 把经验分成“案例”和“规则”

很多 Agent 记忆系统只做向量检索，存一堆历史对话或轨迹。

ExpeL 更进一步，把经验分成：

- 可检索的成功案例；
- 可泛化的抽象规则。

这比单纯存轨迹更合理，因为具体轨迹适合相似任务，抽象规则适合泛化任务。

## 16.3 用成功/失败对比来生成反思

ExpeL 不是只看失败轨迹让 LLM 自我反思，而是把失败轨迹和成功轨迹放在一起对比。

这比单纯 failure reflection 更可靠，因为 LLM 能看到“同一个任务中什么做法失败、什么做法成功”。

这点对于 memory reflection 很有价值。

---

# 17. ExpeL 的局限性

论文自己提到了一些局限，我补充解释如下。

## 17.1 只处理文本观察

实验环境都是 textual observation，例如 HotpotQA、ALFWorld 文本版、WebShop 文本交互。

现实中的 Agent 往往需要处理图像、网页截图、GUI、多模态状态。

如果扩展到真实 Agent，需要结合 VLM 或 captioning 模型。

## 17.2 insight 数量增长后会遇到上下文限制

论文实验中 insights 数量还不大，可以直接拼进 prompt。

但如果是 lifelong agent，insights 会不断增长。

这时需要对 insights 本身做检索、聚类、压缩、去重和遗忘。

## 17.3 依赖强 LLM 做 insight extraction

消融实验已经显示，GPT-4 提取 insights 明显好于 GPT-3.5。

所以 ExpeL 的效果很依赖反思模型能力。如果 insight extraction 模型较弱，可能会生成错误规则。

## 17.4 insight 仍可能 hallucinate

论文发现，把 Reflexion 生成的 reflections 直接加入 insight extraction 反而降低性能。

这说明自然语言反思不是天然可靠的，必须有筛选机制。

ExpeL 用 UPVOTE / DOWNVOTE / EDIT 做了一定过滤，但这个机制仍然比较粗糙。

## 17.5 检索只按任务描述相似度，较简单

ExpeL 的经验检索主要基于任务描述 embedding 相似度。

这在 ALFWorld 等任务中有效，但对于复杂任务，可能还需要结合：

- 状态相似度；
- 动作空间相似度；
- 失败类型相似度；
- goal decomposition 相似度；
- trajectory-level similarity。

---

# 18. 如果你要复现 ExpeL，核心模块应该怎么写？

可以按下面这个系统设计：

## 18.1 数据结构

经验池中每条 trajectory 至少包含：

```json
{
  "task_id": "...",
  "task_description": "...",
  "trial_id": 0,
  "success": true,
  "trajectory": [
    {
      "thought": "...",
      "action": "...",
      "observation": "...",
      "reward": 0
    }
  ],
  "reflection": "...",
  "embedding": [...]
}
```

insight 表可以包含：

```json
{
  "insight_id": "...",
  "content": "...",
  "importance": 3,
  "source_experience_ids": ["..."],
  "created_at": "...",
  "updated_at": "..."
}
```

## 18.2 训练阶段

流程是：

```text
for task in training_tasks:
    reflection = ""
    for z in range(max_retry):
        trajectory, success = run_react(task, reflection)
        save_trajectory(trajectory)

        if success:
            break
        else:
            reflection += reflect(trajectory)
```

## 18.3 经验抽象阶段

流程是：

```text
for fail_success_pair in experience_pool:
    update_insights(pair, existing_insights)

for success_chunk in success_trajectories:
    update_insights(success_chunk, existing_insights)
```

## 18.4 测试阶段

流程是：

```text
similar_successes = retrieve_success_trajectories(task)
prompt = task + insights + similar_successes
run_react(prompt)
```

这个结构很适合作为智能体记忆系统中的 **offline consolidation module**。

---

# 19. 这篇论文和你前面读过的几篇工作的关系

结合你之前读的 Reflexion、Generative Agents、Mem0、Zep，可以这样定位 ExpeL：

| 工作 | 重点 |
|---|---|
| Reflexion | 同一任务失败后的 verbal self-reflection |
| Generative Agents | 长期记忆检索 + 周期性高层 reflection |
| Mem0 | 对话记忆的抽取、更新、删除 |
| Zep | episode / entity / community 多层知识图谱记忆 |
| ExpeL | 从任务轨迹中提取跨任务经验，并在新任务中使用 |

所以 ExpeL 更偏向：

**Agent 任务执行场景中的经验蒸馏与跨任务反思。**

它不是一般对话记忆系统，而是面向 interactive decision-making 的经验学习框架。

---

# 20. 我的评价

这篇论文的价值在于，它把“LLM Agent 学习”从微调模型参数，转向了更工程可行的外部记忆学习：

$$
\text{Learning}
\neq
\text{Weight Update}
$$

而是：

$$
\text{Learning}
=
\text{Experience Collection}
+
\text{Memory Consolidation}
+
\text{Retrieval at Inference}
$$

对智能体记忆反思研究来说，ExpeL 有三个值得借鉴的点：

1. **成功/失败对比比单纯失败反思更可靠。**
2. **抽象 insights 和具体 trajectories 应该同时保留。**
3. **反思结果需要可编辑、可投票、可删除，而不是直接永久写入。**

但它也还比较早期：insight 质量控制、长期记忆规模管理、多模态环境、动态在线更新都还没有完全解决。

总体来说，ExpeL 可以理解为：**把 Reflexion 从“单任务重试反思”扩展成“跨任务经验学习”的一个框架。**

