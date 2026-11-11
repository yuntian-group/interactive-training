# Interactive Training（交互式训练）

语言：简体中文（zh-Hans）

[中文网页](https://interactivetraining.ai/zh/) · [English](https://interactivetraining.ai/)

反馈驱动的神经网络优化

## 直接干预正在运行的训练。

Interactive Training（交互式训练）把实时训练指标与人工、AI 智能体的操作连接起来。观察训练，提交调整，再查看训练进程实际执行了什么、效果如何。

[打开在线演示 →](https://interactivetraining.ai/live) · [使用开源框架 ↗](https://github.com/yuntian-group/interactive-training)

## 三种方式了解和体验系统

01 · 观看

### 视频演示

跟随一次智能体指导的优化实验，了解它如何根据观察提出调整，以及后续训练指标如何变化。

02 · 复盘

### 论文实验记录

对照各轮训练的配置、得分、操作，以及带入下一轮的反思。这里展示的是已完成实验的记录。

03 · 操作

### 在线 CPU 实验环境

加入共享 CPU 队列，运行精简的 tiny-BERT 训练任务，尝试开放的控制项并查看执行结果。此模式不调用大语言模型。

[进入演示，选择体验模式 →](https://interactivetraining.ai/live)（演示界面为英文）

## 从观察训练到执行干预

控制器读取训练指标，通过控制接口提交修改请求；训练进程在继续运行的同时执行请求。优化器设置、数据处理和检查点操作由此组成一个可交互的反馈闭环。

Interactive Training 2 将这一接口扩展为通用控制协议。训练程序声明带类型的可调设置和操作，在训练循环的控制点执行请求，并按时间顺序记录请求与结果。人、脚本和语言模型智能体使用同一套接口。

例如，观察到损失突然升高时，控制器可以请求降低学习率，随后请求评估和保存检查点。提交请求与修改生效是两个不同事件：训练循环到达控制点后验证并执行请求，日志再记录执行结果。

## 论文与作者

EMNLP 2025 · System Demonstrations

### [Interactive Training: Feedback-Driven Neural Network Optimization](https://aclanthology.org/2025.emnlp-demos.65/)

Wentao Zhang, Yang Young Lu, and Yuntian Deng

[引用格式](https://yuntiandeng.com/research/papers/interactive-training-2025/#citation) · [BibTeX](https://yuntiandeng.com/research/data/papers/interactive-training-2025.bib)

原始框架将实时监控与神经网络训练中的反馈驱动干预连接起来，并通过案例研究展示训练稳定性、对初始超参数的敏感性，以及对用户需求变化的适应能力。

2026 · arXiv

### [Interactive Training 2: Auditable Control Plane for Live Model Training](https://arxiv.org/abs/2607.18314)

Wentao Zhang, Xuanhe Pan, Han Zhou, Yang Lu, and Yuntian Deng

[引用格式](https://yuntiandeng.com/research/papers/interactive-training-2026/#citation) · [BibTeX](https://yuntiandeng.com/research/data/papers/interactive-training-2026.bib)

可复用的控制协议与 Aim 工作界面，将实时指标、设置、操作及其执行记录连接起来，并在自然语言处理与强化学习的五个训练流程中展示。

方法与相关研究

## 人和 AI 智能体如何参与训练过程

研究指南介绍交互式训练的方法、它与训练中动态调参及训练智能体的关系，并提供原始论文和可直接使用的文献引用。

[阅读中文研究指南 →](https://yuntiandeng.com/research/zh/interactive-training/) · [English guide](https://yuntiandeng.com/research/interactive-training/)

## 文本与结构化信息

[下载中文正文（Markdown）](https://interactivetraining.ai/zh/index.md) · [中文研究信息（JSON-LD）](https://interactivetraining.ai/zh/research.json)
