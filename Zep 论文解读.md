## 1. 这篇论文到底想解决什么问题

传统 RAG 主要面向 **静态文档检索**。例如把 PDF、网页、知识库文档切块后存入向量数据库，查询时召回相关 chunk。

但 Agent 的记忆场景不同。Agent 面对的是持续变化的数据：

用户不断和 Agent 对话；  
用户偏好会变化；  
业务数据会更新；  
旧事实可能被新事实推翻；  
很多问题需要跨 session、跨时间进行推理。

所以论文认为，普通 RAG 不适合直接作为 Agent 长期记忆系统。因为普通 RAG 通常只保存文本片段，不显式建模实体、关系、时间有效期和历史变化。Zep 的目标就是把 Agent 的长期记忆从“静态文本检索”升级为“动态时间知识图谱检索”。`Zep.pdf`

可以把它的核心问题概括为：

> Agent 记忆不是简单地把历史对话塞进向量库，而是要把对话中不断变化的实体、事实、关系和时间状态组织成可更新、可追溯、可检索的结构化记忆。

---

## 2. Zep 的整体架构：用时间知识图谱做长期记忆

论文中将 Zep 的记忆图定义为：

$$
G=(N,E,\phi)
$$

其中：

$$
N
$$

表示节点集合，

$$
E
$$

表示边集合，

$$
\phi:E\rightarrow N\times N
$$

表示边连接哪些节点。`Zep.pdf`

Zep 的知识图谱不是单层结构，而是分成三层：

| 层级 | 名称 | 作用 |
|---|---|---|
| 第一层 | Episode Subgraph | 保存原始对话、文本、JSON，不丢失原始信息 |
| 第二层 | Semantic Entity Subgraph | 抽取实体和实体之间的事实关系 |
| 第三层 | Community Subgraph | 对实体社区进行聚类和摘要，形成更高层次的概念记忆 |

这三个层次大致对应：

```text
原始经历
↓
实体与事实
↓
高层概念/社区摘要
```

这也是论文类比人类记忆的地方：Episode Subgraph 更像 **情景记忆**，Semantic Entity Subgraph 更像 **语义记忆**，Community Subgraph 更像对知识结构的高层总结。`Zep.pdf`

---

## 3. 第一层：Episode Subgraph，也就是原始记忆层

Episode 是 Zep 图构建的起点。论文中说 Episode 可以是三种类型：

```text
message
text
JSON
```

实验主要关注 message，也就是对话消息。每条消息都包含：

```text
消息内容
说话人
参考时间戳 tref
```

这里最关键的是时间戳。因为对话中经常出现相对时间，例如：

```text
下周四
两周前
去年夏天
刚才
```

如果没有当前消息的参考时间，就无法把这些相对时间转换成具体时间。

所以 Zep 在写入每条消息时，会记录一个：

$$
t_{ref}
$$

然后基于它解析相对时间。`Zep.pdf`

Episode Subgraph 的作用不是直接用于最终回答，而是作为 **非损失存储**。也就是说，Zep 不只保存抽取后的实体和事实，也保留原始对话。这样做有两个好处：

第一，可以从实体或事实反向追溯到原始消息，用于引用、核查、解释来源。

第二，如果后续抽取逻辑升级，还可以基于原始 Episode 重新抽取，而不是只能依赖旧的结构化结果。

---

## 4. 第二层：Semantic Entity Subgraph，也就是实体和事实层

这一层是论文的核心。

Zep 会从 Episode 中抽取：

```text
实体节点
事实边
时间信息
关系有效期
```

### 4.1 实体抽取

对于每条当前消息，系统不仅看当前消息，还会看前面的若干条消息。论文中设置：

$$
n=4
$$

也就是用当前消息加最近 4 条消息作为上下文，帮助进行实体识别。论文说这大约对应两轮完整对话上下文。`Zep.pdf`

例如对话是：

```text
User: 我最近在读 Zep 这篇论文。
Assistant: 你关注的是它的图记忆结构吗？
User: 对，我想比较它和 Mem0。
```

当前消息里的“它”如果只看单句，很难知道指的是谁。但结合前文，就能知道“它”大概率是 Zep。

论文还提到，系统会自动把说话人抽取为实体。例如：

```text
User
Assistant
Preston Rasmussen
某个客户
某家公司
某个产品
```

这点在 Agent 记忆中很重要，因为很多记忆都和用户本人、助手、业务对象有关。

### 4.2 实体反思与实体消歧

实体抽取之后，Zep 会使用一种受 Reflexion 启发的 reflection technique，用来减少幻觉并提高实体抽取覆盖率。`Zep.pdf`

这里的 reflection 不是 Generative Agents 那种周期性生成 insight 的反思，而更像 **写入阶段的检查和修正**。

抽取出实体后，Zep 还要判断它是不是已经存在于图中。比如：

```text
Bob
Robert
Robert Smith
Bob Smith
```

可能其实是同一个人。

Zep 的做法是：

先将实体名称编码成 1024 维向量；  
用余弦相似度从已有实体中找相似节点；  
再用全文搜索匹配实体名称和实体摘要；  
把候选实体和当前上下文交给 LLM 判断是否重复；  
如果重复，就合并并更新名称和摘要。`Zep.pdf`

也就是说，Zep 的实体消歧不是只靠 embedding，而是：

```text
向量检索 + 全文检索 + LLM 判断
```

这是一个典型的 hybrid resolution 流程。

---

## 5. 事实抽取：把关系变成图中的边

实体抽取之后，Zep 会从当前消息中抽取实体之间的事实关系。

例如一句话：

```text
Alice 下个月要加入 OpenAI。
```

可能抽取出：

```text
Alice -- WORKS_FOR / WILL_JOIN --> OpenAI
```

事实会被表示为 semantic edge，也就是图中的边。边上不仅有关系类型，还会有更详细的自然语言事实描述。论文附录中的 fact extraction prompt 要求：

```text
只抽取给定实体之间的事实
事实必须连接两个不同实体
relation_type 使用简洁的大写描述
事实描述要包含相关细节
考虑时间因素
```

这说明 Zep 的事实边不是简单的三元组：

```text
subject - predicate - object
```

而更像是带文本描述和时间属性的结构化事实。`Zep.pdf`

论文还提到，同一个事实可以在多个实体之间被抽取出来，因此 Graphiti 可以用类似 hyper-edge 的方式表达复杂多实体事实。也就是说，它并不完全受限于简单的二元关系。

---

## 6. 时间建模：这篇论文最重要的创新之一

Zep 和普通知识图谱/RAG 最大的区别之一是 **时间感知**。

论文使用 bi-temporal model，也就是双时间线模型：

| 时间线 | 含义 |
|---|---|
| $$T$$ | 现实世界中事实成立的时间 |
| $$T'$$ | 系统中数据被创建、更新、失效的事务时间 |

进一步说，Zep 在边上记录四个时间字段：

$$
t'_{created}
$$

表示这条事实边什么时候被系统创建；

$$
t'_{expired}
$$

表示这条事实边什么时候在系统中被标记失效；

$$
t_{valid}
$$

表示这个事实在现实世界中什么时候开始为真；

$$
t_{invalid}
$$

表示这个事实在现实世界中什么时候不再为真。`Zep.pdf`

这个设计很关键。因为 Agent 记忆中经常会出现“事实更新”。

例如：

```text
3 月：用户说“我现在住在上海”
6 月：用户说“我搬到新加坡了”
```

普通向量库里可能同时有两个 chunk：

```text
用户住在上海
用户住在新加坡
```

检索时很可能两个都召回，模型不知道哪个是当前事实。

Zep 的做法是把旧事实标记为失效：

```text
用户 -- LIVES_IN --> 上海
valid_at: 2026-03
invalid_at: 2026-06
```

新事实变为当前有效：

```text
用户 -- LIVES_IN --> 新加坡
valid_at: 2026-06
invalid_at: null
```

这样 Agent 不仅知道“用户现在住在哪里”，还知道“用户以前住在哪里”。

这比简单的 ADD/UPDATE/DELETE 更细，因为它没有直接删除旧事实，而是保留历史状态。

---

## 7. Edge Invalidation：新事实如何推翻旧事实

论文中专门讲了 Temporal Extraction and Edge Invalidation。核心流程是：

当新事实边进入图中时，系统会找出语义相关的已有边；  
用 LLM 判断新边是否和旧边存在矛盾；  
如果存在时间重叠的矛盾，就将旧边的失效时间设为新边的有效时间。`Zep.pdf`

可以理解为：

```text
新事实写入
↓
检索可能冲突的旧事实
↓
LLM 判断是否矛盾
↓
若矛盾，则让旧事实失效
↓
保留旧事实作为历史记录
```

这其实是 Zep 里非常重要的 **写时反思机制**。

它不是先把所有记忆存进去，之后再统一反思；而是在新记忆写入时，就判断：

```text
这个事实是否和旧事实重复？
是否冲突？
是否应该更新旧事实？
旧事实是否应该失效？
新事实的有效时间是什么？
```

所以从你研究的“智能体记忆反思”角度看，Zep 更偏向 **写时反思 / 在线反思**，而不是典型的离线事后反思。

---

## 8. 第三层：Community Subgraph，高层社区记忆

在实体和事实图建立后，Zep 还会构建 Community Subgraph。

Community 可以理解为图中的“实体簇”。如果一批实体之间连接紧密，就把它们划成一个社区，并生成社区摘要。

这类似 GraphRAG 里的社区摘要，但 Zep 有一个区别：它没有采用 GraphRAG 中常见的 Leiden 算法，而是用了 label propagation，也就是标签传播算法。原因是 label propagation 更容易做动态更新。`Zep.pdf`

它的大致流程是：

```text
新实体加入图
↓
查看邻居实体属于哪些社区
↓
将新实体分配给邻居中占多数的社区
↓
更新社区摘要
```

这样做的优点是速度快、成本低，不需要每次有新数据进来都重新跑完整社区检测。

缺点是，动态增量更新久了以后，社区结构会逐渐偏离全量重新聚类的结果。所以论文也承认，仍然需要周期性 community refresh。`Zep.pdf`

从反思角度看，Community Subgraph 有一点像 **事后反思 / 离线总结**，因为它把底层实体关系聚合成更高层次摘要。但论文中没有把它设计成一个显式的 self-reflection loop，而更像图结构维护和层次化摘要。

---

## 9. Zep 的检索流程

Zep 的检索可以表示为：

$$
f(\alpha)=\chi(\rho(\varphi(\alpha)))=\beta
$$

其中：

$$
\alpha
$$

是用户查询；

$$
\varphi
$$

是搜索函数，用来找候选实体、边和社区；

$$
\rho
$$

是 reranker，用来重排候选结果；

$$
\chi
$$

是 constructor，用来把图中的结果组装成文本上下文；

$$
\beta
$$

是最终提供给 LLM 的上下文。`Zep.pdf`

也就是：

```text
用户问题
↓
图搜索
↓
重排序
↓
构造上下文
↓
交给 LLM 回答
```

Zep 的检索对象主要有三类：

```text
semantic edges，也就是事实边
entity nodes，也就是实体节点
community nodes，也就是社区节点
```

最终返回给 LLM 的上下文包括：

```text
相关 facts
fact 的有效时间范围
相关 entities
entity summary
community summary
```

这比普通 RAG 返回 chunk 更结构化。

---

## 10. 三种搜索方式：向量、BM25、图遍历

Zep 使用三类搜索函数：

| 搜索方式 | 作用 |
|---|---|
| Cosine semantic similarity | 语义相似度搜索 |
| Okapi BM25 full-text search | 关键词/全文搜索 |
| Breadth-first search | 图上的邻近关系搜索 |

论文认为这三种搜索对应不同类型的相似性：

```text
全文搜索：词面相似
向量搜索：语义相似
BFS 图搜索：上下文关系相似
```

这点很重要。普通 RAG 主要靠向量相似或 BM25，但 Zep 多了图上的 BFS。比如一个实体最近经常和另一个实体共同出现，即使文本语义不完全相似，图结构也能帮助召回相关记忆。`Zep.pdf`

例如用户问：

```text
我之前和导师讨论的那个记忆系统方案是什么？
```

如果“导师”“记忆系统方案”“某篇论文”“某次会议”在图上连接很近，BFS 可以帮助把这些相关节点和事实一起召回。

---

## 11. Reranker：从高召回到高精度

初始搜索阶段追求高召回，Reranker 负责提高精度。

Zep 支持多种 reranker：

| Reranker | 作用 |
|---|---|
| RRF | 融合多个检索器的排序 |
| MMR | 保证相关性和多样性 |
| Episode mentions reranker | 优先考虑对话中频繁提到的实体或事实 |
| Node distance reranker | 根据图距离重排 |
| Cross-encoder reranker | 用模型对 query 和候选结果进行深度相关性判断 |

其中最有意思的是 episode-mentions reranker。它会让频繁出现的信息更容易被召回。

这符合人类记忆直觉：经常被提到的事情，应该更容易被想起。

但它也有风险：高频不一定重要，低频也可能是关键事实。所以这个机制适合和其他 reranker 组合使用，而不是单独作为最终依据。

---

## 12. 实验一：Deep Memory Retrieval

第一个实验是 DMR，也就是 MemGPT 使用过的 Deep Memory Retrieval benchmark。

DMR 数据集包含 500 个多 session 对话，每个对话有 5 个 session，每个 session 最多 12 条消息。也就是说每个样本最多约 60 条消息。论文指出，这个数据规模其实比较小，现代 LLM 完全可以把整段对话放进上下文窗口。`Zep.pdf`

实验结果如下：

| 方法 | 模型 | 分数 |
|---|---|---|
| Recursive Summarization | gpt-4-turbo | 35.3% |
| Conversation Summaries | gpt-4-turbo | 78.6% |
| MemGPT | gpt-4-turbo | 93.4% |
| Full-conversation | gpt-4-turbo | 94.4% |
| Zep | gpt-4-turbo | 94.8% |
| Full-conversation | gpt-4o-mini | 98.0% |
| Zep | gpt-4o-mini | 98.2% |

Zep 的确超过了 MemGPT，但提升非常小：

```text
94.8% vs 93.4%
```

而且 full-conversation baseline 已经达到：

```text
94.4%
```

这说明 DMR 不太能区分复杂记忆系统的能力。论文自己也承认，DMR 主要是单轮事实检索问题，很多问题表达还有歧义，不足以代表真实企业场景。`Zep.pdf`

所以 DMR 的结论应该谨慎看待：Zep 在 DMR 上略优，但这个 benchmark 太简单，不能充分证明 Zep 的长期记忆能力。

---

## 13. 实验二：LongMemEval

第二个实验更重要。论文使用 LongMemEval 中的 LongMemEvals 数据集，平均上下文长度约 115,000 tokens。这个规模更接近真实长期对话记忆。`Zep.pdf`

LongMemEval 包含六类问题：

```text
single-session-user
single-session-assistant
single-session-preference
multi-session
knowledge-update
temporal-reasoning
```

这些问题比 DMR 更复杂，尤其是：

```text
跨 session 信息整合
用户偏好推理
知识更新
时间推理
```

这些正是长期记忆系统应该解决的问题。

论文结果如下：

| 方法 | 模型 | 准确率 | 延迟 | 平均上下文 tokens |
|---|---|---:|---:|---:|
| Full-context | gpt-4o-mini | 55.4% | 31.3s | 115k |
| Zep | gpt-4o-mini | 63.8% | 3.20s | 1.6k |
| Full-context | gpt-4o | 60.2% | 28.9s | 115k |
| Zep | gpt-4o | 71.2% | 2.58s | 1.6k |

这里的提升更有意义。

Zep 不仅准确率更高，而且把上下文从平均 115k tokens 降到了 1.6k tokens，同时延迟从约 29-31 秒降到约 2.6-3.2 秒。论文称延迟降低约 90%。`Zep.pdf`

这说明，在长对话场景下，直接把所有历史塞进上下文并不一定更好。原因可能有两个：

第一，长上下文会稀释关键信息，模型不一定能有效利用 100k 级上下文。

第二，Zep 的图检索把相关事实、实体和时间范围压缩成更短、更结构化的上下文，降低了模型理解负担。

---

## 14. 分问题类型看 Zep 的表现

论文进一步按问题类型分析。Zep 在以下几类问题上提升明显：

```text
single-session-preference
multi-session
temporal-reasoning
single-session-user
knowledge-update
```

尤其是 preference、multi-session 和 temporal-reasoning，这些正好对应 Agent 长期记忆的核心场景。`Zep.pdf`

但是 Zep 在 single-session-assistant 类型上反而下降：

```text
gpt-4o-mini: 81.8% → 75.0%
gpt-4o: 94.6% → 80.4%
```

论文也承认这是一个问题，需要进一步研究。`Zep.pdf`

我的理解是，single-session-assistant 问题可能更依赖原始对话细节，而 Zep 的图结构化过程可能更强调实体和事实，导致某些 assistant 说过的具体表达、解释、建议没有被完整保留下来。

也就是说，图记忆对“事实关系、偏好、时间变化”很强，但对“某句话具体怎么说的、某段回答的细节”可能不如原始上下文。

---

## 15. 这篇论文和普通 RAG 的区别

普通 RAG 的基本流程是：

```text
文档切块
↓
embedding
↓
向量检索 / BM25
↓
召回 chunk
↓
LLM 生成
```

Zep 的流程是：

```text
对话/文本/JSON 写入
↓
抽取实体
↓
实体消歧
↓
抽取事实关系
↓
抽取时间有效期
↓
检测新旧事实冲突
↓
更新/失效旧事实
↓
构建实体社区摘要
↓
混合检索：向量 + BM25 + BFS
↓
rerank
↓
构造结构化上下文
↓
LLM 生成
```

所以 Zep 不是简单的 GraphRAG，也不是简单的 memory vector store，而是一个 **动态记忆维护系统**。

---

## 16. 从“记忆反思”角度看 Zep

结合你当前研究的 Agent memory reflection，这篇论文非常值得关注，但要注意它的 reflection 类型。

我会把 Zep 的反思机制分成三类。

### 16.1 写时反思：最核心

Zep 的写时反思主要体现在：

```text
实体抽取后的 reflection 检查
实体消歧
事实去重
时间抽取
新旧事实矛盾检测
edge invalidation
```

这些都发生在新数据写入图谱时。

它的核心问题不是：

```text
我能从过去记忆中总结出什么高层 insight？
```

而是：

```text
这条新记忆应该如何结构化？
它是不是已有实体？
它是不是已有事实？
它是否推翻旧事实？
它的有效时间是什么？
旧事实是否应该失效？
```

所以 Zep 是典型的 **写时反思型记忆系统**。

### 16.2 检索时反思：辅助存在

Zep 的 reranker 也有一定反思性质。例如它会根据图距离、mention frequency、cross-encoder 相关性判断哪些记忆更重要。

这不是写入反思，而是 **检索时上下文选择反思**：

```text
当前问题需要哪些记忆？
哪些事实更相关？
哪些实体更应该进入上下文？
```

### 16.3 事后反思：相对较弱

Zep 的 community summary 和 periodic community refresh 有一点事后总结意味，但它并不像 Generative Agents 那样主动周期性生成“高层反思 insight”。

Generative Agents 的反思更像：

```text
从最近 100 条记忆中生成问题
↓
检索相关记忆
↓
总结 insight
↓
把 insight 写回记忆库
```

Zep 更像：

```text
每次写入时维护实体、事实、时间和冲突关系
```

因此，Zep 的主要贡献不是离线自我修正，而是 **在线结构化写入 + 时间一致性维护**。

---

## 17. 和 Mem0、MemGPT、GraphRAG 的关系

### 17.1 和 MemGPT

MemGPT 更强调把 LLM 类比成操作系统，通过显式 memory management 管理上下文、archival memory 等。Zep 则更强调底层 memory layer 的结构，尤其是 temporal knowledge graph。

简单说：

```text
MemGPT：更像 Agent 的记忆管理策略/OS 抽象
Zep：更像生产级记忆数据库/图谱记忆层
```

### 17.2 和 Mem0

Mem0 更偏向“从对话中抽取可长期保存的用户记忆”，通常强调 ADD、UPDATE、DELETE、NOOP 之类的记忆操作。

Zep 的粒度更细。它不是只保存一句自然语言记忆，而是保存：

```text
实体
实体摘要
事实边
事实时间范围
失效时间
社区摘要
原始 episode
```

所以 Zep 的结构化程度更强，尤其适合需要时间推理和关系推理的场景。

### 17.3 和 GraphRAG

GraphRAG 主要面向文档知识库，通过实体图和社区摘要支持全局问答。

Zep 借鉴了 GraphRAG 的 community idea，但更强调：

```text
动态更新
对话记忆
时间有效期
事实失效
生产系统延迟
```

所以可以理解为：

```text
GraphRAG 偏静态文档图谱
Zep / Graphiti 偏动态 Agent 记忆图谱
```

---

## 18. 这篇论文的优点

第一，问题定义比较准确。它指出 Agent 记忆和传统 RAG 的区别：Agent 记忆是动态、长期、时间相关、会被新事实更新的。

第二，结构设计比较完整。Episode、Entity、Community 三层结构既保留原始信息，又支持结构化推理和高层摘要。

第三，时间建模是亮点。双时间线和边有效期设计，让系统能够处理“过去为真、现在不为真”的事实。

第四，写入阶段的反思机制比较实用。实体消歧、事实去重、冲突检测、edge invalidation 都是实际 Agent 记忆系统中必须解决的问题。

第五，LongMemEval 结果比 DMR 更有说服力。Zep 在长上下文任务上同时提升准确率和降低延迟，这说明结构化记忆比暴力长上下文更高效。`Zep.pdf`

---

## 19. 这篇论文的局限

第一，DMR 实验意义有限。Zep 只比 full-conversation baseline 高 0.4%，而 full-conversation 已经超过 MemGPT。这说明 DMR 太简单，不足以证明复杂记忆架构的优势。论文自己也承认这一点。`Zep.pdf`

第二，和 MemGPT 在 LongMemEval 上没有真正完成对比。论文说他们尝试评估 MemGPT，但由于 MemGPT 当前框架不支持直接 ingestion 历史消息，只能 workaround，最终没有成功得到有效结果。因此 LongMemEval 上缺少与 MemGPT 的直接对照。`Zep.pdf`

第三，图构建成本没有被充分展开。论文强调检索和回答延迟降低，但构建图谱本身需要多次 LLM 调用，包括实体抽取、实体消歧、事实抽取、时间抽取、冲突检测、摘要生成等。这些写入成本可能不低。

第四，结构化抽取会丢失某些表达细节。single-session-assistant 问题表现下降，说明图谱抽取出来的 facts/entities 不一定能覆盖 assistant 回答中的具体措辞或解释细节。`Zep.pdf`

第五，论文由 Zep AI 作者撰写，系统本身也是商业产品，所以实验结论需要等待更多第三方复现。

第六，论文虽然声称可以融合 structured business data，但实验主要还是 conversation memory，没有充分评估结构化业务数据和对话数据融合后的效果。`Zep.pdf`

---

## 20. 如果你要借鉴这篇论文做研究，可以抓这几个点

### 方向一：写时反思机制

Zep 适合作为写时反思的代表系统。你可以把它归入：

```text
online memory reflection
write-time memory consolidation
temporal memory update
contradiction-aware memory writing
```

重点机制包括：

```text
实体抽取反思
实体消歧
事实去重
时间抽取
事实冲突检测
旧事实失效
```

这和 SAGE 那种 ADD/UPDATE/NOOP 判断有相似性，但 Zep 更偏知识图谱和时间一致性。

### 方向二：时间感知记忆

它的 temporal KG 设计很适合你后续调研 Agent memory reflection，因为很多 memory reflection 论文只总结 insight，却没有处理事实有效期。

你可以把 Zep 的时间机制总结成：

```text
不删除旧记忆，而是维护事实的有效时间区间。
```

这比简单覆盖记忆更适合长期 Agent。

### 方向三：图结构记忆 vs 向量记忆

Zep 提供了一个很好的对比对象：

```text
向量记忆：适合语义召回
图谱记忆：适合实体关系、跨 session、时间推理
混合记忆：两者结合
```

Zep 实际采用的就是混合方案：

```text
知识图谱结构 + 向量搜索 + BM25 + BFS + rerank
```

### 方向四：事后反思不足，可以作为改进点

Zep 主要强在写时维护，但缺少强显式的离线反思机制。你可以提出一个改进方向：

```text
在 Zep 的 temporal KG 上增加 periodic reflection module，
周期性地从 episode/entity/fact/community 中生成高层 insight，
并将 insight 作为新的 semantic node 或 community summary 写回图谱。
```

也就是说，可以把 Generative Agents 的 reflection 和 Zep 的 temporal KG 结合起来。

---

## 21. 一句话总结

这篇论文的核心贡献是：**把 Agent 长期记忆从普通 RAG 的“文本片段检索”升级为“时间感知的动态知识图谱维护与检索”**。它最有价值的部分不是 DMR 上略高的分数，而是 Graphiti 的写入流程：实体抽取、实体消歧、事实抽取、时间有效期建模、冲突检测和旧事实失效。对于你的研究方向，它更适合作为 **写时反思 / 在线记忆反思** 的代表，而不是典型的离线事后反思系统。

