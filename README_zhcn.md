[ [English](README.md) ] [ [Español](README_es.md) ] [ [繁體中文](README_zhtw.md) ] [ [日本語](README_ja.md) ] [ [한국어](README_ko.md) ] 

# AI 源动力 (AI Power)

AI 技术正以惊人的速度进步，各种新算法和 AI 库不断涌现和演变。为了让更多人能够掌握最新的 AI 创新技术并积极参与开源项目，我创建了 AI 源动力。加入我们，一起探索 AI 技术的前沿，为塑造未来贡献力量！

## 文章分类

### 基于代码的算法学习

| 标题                 | 描述           | 关键词              |
|-----------------------|-----------------------|-----------------------|
| [理解 Transformer 注意力机制](deep_learning/transformer/transformer_attentions.md)                   | 深入解释 Transformer 中的注意力机制，涵盖自注意力、多头注意力及其在现代 NLP 模型中的实现。 | Transformers, Self-Attention, MHA |
| [Vanilla Transformer 详解](deep_learning/transformer/vanilla_transformer_explained_enus.md)       | 全面指南，详细介绍 Vanilla Transformer 模型的架构、组件和序列到序列任务的前向传播过程。 | vanilla Transformer, Architecture, Sequence-to-Sequence |
| [深入理解 CLIP](deep_learning/transformer/model_multimodal/huggingface_clip_explained.md)               | 深入解释 CLIP 模型，涵盖其架构、训练过程以及在图像和文本链接中的应用。 | CLIP, Architecture |
| [深入理解 LLaVA](deep_learning/transformer/model_multimodal/llava_implementation_explained.md)           | 全面指南，探索 LLaVA 模型的实现，解析其架构、组件及其如何增强语言理解任务。 | LLaVA, Architecture, MultiModal |
| [深入理解视觉 Transformer](deep_learning/transformer/model_vision/huggingface_vit_explained.md)                | 深入解释视觉 Transformer (ViT) 模型，详细介绍其架构、关键组件及其在计算机视觉任务中的应用。 | Vision Transformer, ViT, Architecture |
| [自编码器详解](deep_learning/diffusion/autoencoder_explained.md)                    | 深入解释自编码器的架构、类型及其在数据压缩和特征学习中的应用。 | AutoEncoder, VAE, Architecture |

### Huggingface API

| 标题                 | 描述           | 关键词              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface SBERT API - 第1部分](deep_learning/transformer/huggingface_sbert_api_part1.md) | 介绍 Huggingface Sentence-BERT (SBERT) API，解释其用途、应用以及如何使用它来嵌入句子和计算相似度。 | SBERT, Sentence Transformer, Embeddings         |
| [Huggingface SBERT API - 第2部分](deep_learning/transformer/huggingface_sbert_api_part2.md) | SBERT API 指南的延续，涵盖高级用法、模型微调以及 SBERT 与各种应用的集成。 | SBERT, Sentence Transformer, Embeddings |
| [Huggingface Transformer Auto Class API - 第1部分](deep_learning/transformer/huggingface_transformer_auto_class_api_part1.md) | 介绍 Huggingface Transformer Auto Class API 的概况，详细说明其功能、设置及其在不同 NLP 任务中的基本用法。 | Transformer, Auto Class, NLP, API             |
| [Huggingface Transformer Auto Class API - 第2部分](deep_learning/transformer/huggingface_transformer_auto_class_api_part2.md) | 深入探讨 Transformer Auto Class API，探索自定义配置、模型优化和用例。 | Transformer, Auto Class, NLP, API |
| [Huggingface Transformer Pipeline API](deep_learning/transformer/huggingface_transformer_pipeline_api.md) | 解释 Huggingface Transformer Pipeline API，展示其在文本分类、命名实体识别和文本生成等各种 NLP 任务中的简易使用。 | Transformer Pipeline |
| [Huggingface CLIP API](deep_learning/transformer/model_multimodal/huggingface_clip_api.md)                     | 介绍 Huggingface CLIP (对比语言-图像预训练) API，解释其用途、应用以及如何使用它来嵌入图像和文本。 | Huggingface, CLIP, API |
| [Huggingface LLaVA Next API](deep_learning/transformer/model_multimodal/huggingface_llava_next_api.md)               | 详细介绍 Huggingface LLaVA Next API，概述其功能、设置及其在高级语言理解任务中的使用。 | LLaVA, MultiModal, API |
| [Huggingface Vision Transformer (ViT) API](deep_learning/transformer/model_vision/huggingface_vit_api.md)       | 介绍 Huggingface Vision Transformer (ViT) API，解释其在图像分类和其他视觉任务中的使用。 | ViT, Vision Transformer, Image Classification, API |
| [Huggingface Diffusers API](deep_learning/diffusion/huggingface_diffusers_api.md)                | 介绍 Huggingface Diffusers API，解释其功能及其在从文本描述生成图像方面的使用。 | Diffusers API, Image Generation, Text-to-Image, Image-to-Image |
| [Huggingface Diffusers Chained Pipeline](deep_learning/diffusion/huggingface_diffusers_chained_pipeline.md)   | 解释如何使用 Huggingface Diffusers API 创建链式管道，展示如何结合多个模型进行复杂任务。 | Diffusers API, Chained Pipeline |
| [Huggingface Diffusers Pipeline API](deep_learning/diffusion/huggingface_diffusers_pipeline_api.md)       | 深入探讨 Huggingface Diffusers Pipeline API，详细说明其功能、设置及其在图像和文本生成中的应用。 | Diffusers Pipeline API |

### 张量操作

| 标题                 | 描述           | 关键词              |
|-----------------------|-----------------------|-----------------------|
| [Einops Einsum](deep_learning/coding/einops_einsum.md)                            | 介绍 einops 库中的 einsum 函数，解释其语法、用法及其在张量操作中的应用。 | einops, einsum, tensor operations |
| [Einops Rearrange](deep_learning/coding/einops_rearrange.md)                         | 详细介绍 einops 库中的 rearrange 函数，展示如何高效地操作和转换张量形状。 | einops, einsum, tensor operations |

### 数据集

| 标题                 | 描述           | 关键词              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface Datasets 加载](deep_learning/dataset/huggingface_datasets_loading.md)           | 介绍如何使用 Huggingface Datasets 库加载和预处理数据集，包括处理各种数据格式和来源。 | Huggingface, Datasets |
| [Huggingface Datasets 主要类](deep_learning/dataset/huggingface_datasets_main_classes.md) | 全面指南，介绍 Huggingface Datasets 库的主要类，解释它们的功能和用例。 | Huggingface, Datasets, Main Classes |
| [Alpaca自我指导指南](deep_learning/dataset/alpaca_self_instruct_guide.md)           | 本指南全面概述了使用Alpaca的自我指导过程，包括逐步说明和示例。 | Alpaca, Self-Instruct       |

### 模型训练

| 标题                 | 描述           | 关键词              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface Transformer Trainer API](deep_learning/training/huggingface_transformer_trainer_finetune.md) | 探讨如何使用 Huggingface Trainer API 微调 Transformer 模型，涵盖设置、训练和评估过程。 | Huggingface, Transformer, Trainer API, SFT |
| [Huggingface Evaluate API](deep_learning/training/huggingface_evaluate_api.md) | 介绍 Huggingface Evaluate API，详细说明其用途、设置及其在评估机器学习模型中的应用。 | Huggingface, Evaluate API, Metric |


## 我们的目标

AI 源动力的目标如下：

- **有效理解 AI 算法**：通过阅读代码，加深对各种 AI 算法的理解。
- **快速学习 AI 库**：利用大量示例程序，加速掌握各种 AI 库。
- **分析代码**：通过分析各种 AI 框架和应用程序的代码，促进学习。
- **学习模型训练技巧**：通过丰富的示例程序和经验分享，快速掌握 AI 模型训练所需的技能。
- **MLOps 流程设计**：学习并实践 MLOps 流程设计，提高模型部署和管理的效率与可靠性。
- **系统架构设计**：通过案例学习 AI 应用的系统架构设计，包括软件架构和云架构设计。

## 如何贡献

如果您希望协助以下工作，欢迎加入我们：

- **协助翻译现有文章**：帮助翻译和改进现有的文章和教学材料。
- **贡献新文章**：每月至少贡献一篇新文章，分享您的 AI 知识和经验。

## 加入我们

我们欢迎所有对 AI 技术感兴趣的人士加入我们的社区，无论您的经验水平如何。您的每一份贡献都将对推动 AI 技术的普及和发展起到重要作用。

## 联系我们

如果您有任何问题或建议，请通过 [GitHub Issues](https://github.com/ksmooi/ai_power/issues) 与我们联系。

感谢您的参与和支持！