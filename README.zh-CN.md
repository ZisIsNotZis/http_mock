<p align="center">
  <img src="assets/logo.svg" alt="http_mock：请求与响应穿过 GRU seq2seq" width="720">
</p>

<p align="center">
  <strong>简体中文</strong> · <a href="README.md">English</a>
</p>

<h1 align="center">http_mock</h1>

<p align="center"><strong>一个 GRU 能学会扮演服务器吗？</strong></p>

<p align="center">
  即时生成 HTTP 与键值协议轨迹，再从零训练一个字符级 seq2seq 去模仿它们。
  不用外部数据集，不启动真实服务器，也不加载预训练权重。
</p>

<p align="center">
  <sub>归档实验 · 没有保存输出或指标 · AGPL-3.0-only</sub>
</p>

## 想法

服务器本质上是一台带语法的状态机。这个仓库只问一个更窄的问题：
如果从一个玩具服务器生成足够多的请求/响应对，一个小型字符级 GRU
编码器–解码器能否学会把请求文本映射为响应文本，并顺带学到背后的状态变化？

需要检查的东西都在 notebook 里：协议生成器、字符词表、模型、损失函数和训练循环。
没有现成的服务框架，也没有外部数据集。代价同样直白：这是一次实验，
不是基准、库或持续维护的服务。

## 现在能做什么

它能生成两类合成协议流量，并训练模型去模仿生成出来的响应：

| Notebook | 协议 | 探索的内容 |
| --- | --- | --- |
| [`http_mock.ipynb`](notebooks/http_mock.ipynb) | 类 REST 的 HTTP | 关系型玩具数据库（`user`、`group`、`item`、`unit`、`craft`、`robot`）藏在 `GET`、`POST`、`PUT`、`DELETE`、`PATCH` 之后；一个基线 GRU seq2seq。 |
| [`http_mock_2.ipynb`](notebooks/http_mock_2.ipynb) | 键值存储 | 面向行的存储，支持 `ls`、`du`、`echo`、`get`、`set`、`del` 和算术命令；多客户端轨迹编码成二维请求/响应网格。 |
| [`http_mock_3.ipynb`](notebooks/http_mock_3.ipynb) | 带引用的键值存储 | 元组引用；三重损失（翻译加两个重建损失）；embedding 去噪；在二维网格上运行 RNN 的 `Seq2Rnn2Seq`。 |
| [`http_mock_4.ipynb`](notebooks/http_mock_4.ipynb) | 键值存储 | notebook 3 的未完成副本。第一个 cell 在编辑中途被截断；保留它是为了留下历史，不要拿它当示例。 |
| [`box.ipynb`](notebooks/box.ipynb) | 无 | 草稿本：一个基于 `tabulate`/HTML 的显示辅助函数，以及纯文本回退。 |

协议本身刻意做得很小。一条请求大致是这样：

```text
HTTP/1.0 GET /user/3
HTTP/1.0 PATCH /item/7

set a 3
mathadd a 4
ls
```

HTTP mock 的读取和写入响应是状态行加 JSON 体；键值 mock 返回简短的 `ok ...` 行。
模型从不接触 socket。

**仓库里没有保存任何输出。** 没有记录的 loss 曲线、准确率、checkpoint
或评估集，因此 notebook 并不声称模型已经收敛或能够泛化。

## 快速开始

notebook 各自独立；按现有代码，需要可用的 CUDA 版 PyTorch。

```bash
git clone https://github.com/ZisIsNotZis/http_mock.git
cd http_mock

# 先按 https://pytorch.org/get-started/locally/ 安装对应平台的 CUDA 版 PyTorch
pip install tabulate jupyterlab
jupyter lab
```

然后打开 `notebooks/http_mock_3.ipynb`，运行全部 cell。它会边生成 batch
边训练 200,000 步，并沿途打印样本。`notebooks/http_mock.ipynb` 更小，
更适合作为起点。

没有 requirements 文件、入口脚本或测试命令。notebook 直接调用 `.cuda()`
并使用 `device='cuda'`；要在 CPU 上运行需要改这些行，且不受支持。

## 架构

```text
协议状态
   │
   ▼
请求文本 ──► 字符 embedding ──► GRU 编码器 ──► 隐状态
                                                 │
响应文本 ◄── 词表 argmax ◄── GRU 解码器 ◄────────┘
```

训练时目标响应会喂给解码器（teacher forcing），并在其 embedding 上做随机
token dropout。`http_mock_3.ipynb` 加了两个辅助重建损失：一个重建请求，一个重建响应，
迫使隐状态同时保留对话的两边。`Seq2Rnn2Seq` 还把多客户端的请求/响应网格
当作二维序列，在上面运行 RNN。

## 如何开发

- **依赖：** Python 3、CUDA 版 PyTorch、JupyterLab 和 `tabulate`。
- **没有构建步骤：** 每个 notebook 自成一体；没有要安装的包，没有共享模块，
  也没有生成代码。
- **没有测试套件：** 检查改动的方式是在 CUDA 机器上运行受影响的 notebook，
  查看打印出的轨迹和预测。
- **不要把输出提交进 git：** 提交前清空 notebook 输出。这样归档更易读，
  也不会把某次运行的结果带进历史。
- **保留原貌：** `http_mock_4.ipynb` 是有意保留在损坏状态的。不要在一股脑的
  清理中顺手“修好”它。

## 如何贡献

这是一个归档仓库，维护范围仅限于修正：坏掉的 notebook、错误的说法、失效的路径
或迁移修复。请到
[github.com/ZisIsNotZis/http_mock/issues](https://github.com/ZisIsNotZis/http_mock/issues)
提 issue，并注明 notebook 和 cell。

想继续这个实验，请 fork 后按自己的方向推进。这里不预期有新功能或新的训练结果。

## 路线图

### 已实现

- [x] 在关系型玩具数据库上生成合成的类 REST HTTP 轨迹。
- [x] 在这些轨迹上训练基线字符级 GRU 编码器–解码器。
- [x] 面向行的键值协议，支持算术命令和共享状态。
- [x] 多客户端轨迹编码为二维网格，并配套网格感知的 seq2seq 模型。
- [x] 元组引用、去噪、三重重建损失和 `Seq2Rnn2Seq`。
- [x] AGPL-3.0-only 许可证，以及如实记录项目状态。

### 计划

- 无。仓库已归档，没有维护或功能路线图。

**未来设想。** 更大的想法是学习一个协议状态机模型：只从流量中模仿一项服务，
然后在测试客户端时让它临时顶替真实服务。这些 notebook 是朝这个想法迈出的一小步，
而且可以完整检查；它不是“这一步已经成功”的证据。没有记录任何指标，
因此这里没有可复现或可超越的结果。

## 项目状态

迁移记录和归档边界见 [`docs/project-status.md`](docs/project-status.md)。
简而言之：代码仅作参考保留，不再维护；本次发布也没有运行它。

## 许可证

[AGPL-3.0-only](LICENSE)。完整文本见 [`LICENSE`](LICENSE)。
