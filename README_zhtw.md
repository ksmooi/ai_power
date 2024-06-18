[ [English](README.md) ] [ [Español](README_es.md) ] [ [简体中文](README_zhcn.md) ] [ [日本語](README_ja.md) ] [ [한국어](README_ko.md) ] 

# AI 源動力 (AI Power)

AI 技術正以驚人的速度進步，各種新演算法和 AI 庫不斷湧現和演變。為了讓更多人能夠掌握最新的 AI 創新技術並積極參與開源專案，我創建了 AI 源動力 (AI Power)。加入我們，一起探索 AI 技術的前沿，為塑造未來貢獻力量！

## 文章分類

### 基於程式碼的演算法學習

| 標題                 | 描述           | Keywords              |
|-----------------------|-----------------------|-----------------------|
| [理解 Transformer 注意力機制](deep_learning/transformer/transformer_attentions.md)                   | 深入解釋 Transformer 中的注意力機制，涵蓋自注意力、多頭注意力及其在現代 NLP 模型中的應用。 | Transformers, Self-Attention, MHA |
| [解釋原始的 Transformer](deep_learning/transformer/vanilla_transformer_explained_enus.md)       | 詳盡介紹原始的 Transformer 模型，詳細描述其架構、組件和序列到序列任務的前向傳播過程。 | vanilla Transformer, Architecture, Sequence-to-Sequence |
| [深入解析 CLIP](deep_learning/transformer/model_multimodal/openai_clip_explained.md)               | 深入解釋 CLIP 模型，涵蓋其架構、訓練過程及其在連接圖像和文本方面的應用。 | CLIP, Architecture |
| [深入解析 LLaVA](deep_learning/transformer/model_multimodal/llava_implementation_explained.md)           | 詳盡介紹 LLaVA 模型的實現，探索其架構、組件以及如何提升語言理解任務。 | LLaVA, Architecture, MultiModal |
| [深入解析視覺 Transformer](deep_learning/transformer/model_vision/huggingface_vit_explained.md)                | 深入解釋視覺 Transformer (ViT) 模型，詳細描述其架構、關鍵組件及其在計算機視覺任務中的應用。 | Vision Transformer, ViT, Architecture |
| [自動編碼器解釋](deep_learning/diffusion/autoencoder_explained.md)                    | 深入解釋自動編碼器，其架構、類型及其在數據壓縮和特徵學習中的應用。 | AutoEncoder, VAE, Architecture |

### Huggingface API

| 標題                 | 描述           | Keywords              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface SBERT API - 第一部分](deep_learning/transformer/huggingface_sbert_api_part1.md) | 本文介紹了 Huggingface Sentence-BERT (SBERT) API，解釋了其用途、應用以及如何使用它來嵌入句子和計算相似性。 | SBERT, Sentence Transformer, Embeddings         |
| [Huggingface SBERT API - 第二部分](deep_learning/transformer/huggingface_sbert_api_part2.md) | SBERT API 指南的延續，涵蓋高級用法、微調模型及將 SBERT 整合到各種應用中。 | SBERT, Sentence Transformer, Embeddings |
| [Huggingface Transformer Auto Class API - 第一部分](deep_learning/transformer/huggingface_transformer_auto_class_api_part1.md) | 本文概述了 Huggingface Transformer Auto Class API，詳細描述了其功能、設置及用於不同 NLP 任務的基本用法。 | Transformer, Auto Class, NLP, API             |
| [Huggingface Transformer Auto Class API - 第二部分](deep_learning/transformer/huggingface_transformer_auto_class_api_part2.md) | 深入探討 Transformer Auto Class API，探索自定義配置、模型優化及應用案例。 | Transformer, Auto Class, NLP, API |
| [Huggingface Transformer Pipeline API](deep_learning/transformer/huggingface_transformer_pipeline_api.md) | 本指南解釋了 Huggingface Transformer Pipeline API，展示了其在文本分類、命名實體識別和文本生成等各種 NLP 任務中的易用性。 | Transformer Pipeline |
| [Huggingface CLIP API](deep_learning/transformer/model_multimodal/huggingface_clip_api.md)                     | 本文介紹了 Huggingface CLIP (對比語言-圖像預訓練) API，解釋了其用途、應用及如何使用它來嵌入圖像和文本。 | Huggingface, CLIP, API |
| [Huggingface LLaVA Next API](deep_learning/transformer/model_multimodal/huggingface_llava_next_api.md)               | 本指南詳細介紹了 Huggingface LLaVA Next API，概述了其功能、設置及用於高級語言理解任務的用法。 | LLaVA, MultiModal, API |
| [Huggingface 視覺 Transformer (ViT) API](deep_learning/transformer/model_vision/huggingface_vit_api.md)       | 本文概述了 Huggingface 視覺 Transformer (ViT) API，解釋了其在圖像分類和其他視覺任務中的用法。 | ViT, Vision Transformer, Image Classification, API |
| [Huggingface Diffusers API](deep_learning/diffusion/huggingface_diffusers_api.md)                | 本文概述了 Huggingface Diffusers API，解釋了其功能及用於從文本描述生成圖像的用法。 | Diffusers API, Image Generation, Text-to-Image, Image-to-Image |
| [Huggingface Diffusers 串聯 Pipeline](deep_learning/diffusion/huggingface_diffusers_chained_pipeline.md)   | 本指南解釋了如何使用 Huggingface Diffusers API 創建串聯 Pipeline，展示了如何將多個模型結合以完成複雜任務。 | Diffusers API, Chained Pipeline |
| [Huggingface Diffusers Pipeline API](deep_learning/diffusion/huggingface_diffusers_pipeline_api.md)       | 深入探討 Huggingface Diffusers Pipeline API，詳細描述其功能、設置及在圖像和文本生成中的應用。 | Diffusers Pipeline API |

### 張量操作

| 標題                 | 描述           | Keywords              |
|-----------------------|-----------------------|-----------------------|
| [Einops Einsum](deep_learning/coding/einops_einsum.md)                            | 介紹 einops 庫中的 einsum 函數，解釋其語法、用法及在張量操作中的應用。 | einops, einsum, tensor operations |
| [Einops Rearrange](deep_learning/coding/einops_rearrange.md)                         | 本指南詳細介紹了 einops 庫中的 rearrange 函數，展示了如何高效地操作和轉換張量形狀。 | einops, einsum, tensor operations |

### 數據集

| 標題                 | 描述           | Keywords              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface Datasets 加載](deep_learning/dataset/huggingface_datasets_loading.md)           | 本文介紹了如何使用 Huggingface Datasets 庫加載和預處理數據集，包括處理各種數據格式和來源。 | Huggingface, Datasets |
| [Huggingface Datasets 主要類別](deep_learning/dataset/huggingface_datasets_main_classes.md) | 詳盡介紹 Huggingface Datasets 庫的主要類別，解釋其功能及應用案例。 | Huggingface, Datasets, Main Classes |
| [Alpaca自我指導指南](deep_learning/dataset/alpaca_self_instruct_guide.md)           | 本指南全面概述了使用Alpaca的自我指導過程，包括逐步說明和示例。 | Alpaca, Self-Instruct       |
| [使用 Unstructured.io 和 GPT4 生成數據集](deep_learning/dataset/generate_dataset_with_unstructureio_gpt4.md) | 本文演示了如何使用 Unstructured.io 和 GPT-4 處理 PDF 文件並通過提取和組織內容生成數據集。 | Unstructured.io, GPT-4 |
| [使用 Table Transformer 和 GPT4 生成數據集](deep_learning/dataset/generate_dataset_with_table_transformer.md) | 本文解釋了如何使用 Table Transformer 和 GPT-4 通過檢測和提取表格結構從 PDF 文件生成數據集。 | Table Transformer, GPT-4 |

### 模型訓練

| 標題                 | 描述           | Keywords              |
|-----------------------|-----------------------|-----------------------|
| [SFT: 模型架構調整](deep_learning/training/sft_model_arch_tweaks.md) | 本文討論了各種模型架構調整和優化，以提高監督微調的性能。           |
| [SFT: 訓練策略](deep_learning/training/sft_train_strategy.md)       | 本文提供了監督微調的有效訓練策略，包括提示和最佳實踐。         |
| [SFT: 數據處理](deep_learning/training/sft_data_handling.md)        | 本文解釋了用於監督微調任務的數據處理和準備技術。          |
| [SFT: 損失函數](deep_learning/training/sft_loss_function.md)        | 本文探討了用於監督微調的不同損失函數及其對模型性能的影響。           |
| [Huggingface Transformer 訓練 API](deep_learning/training/huggingface_transformer_trainer_finetune.md) | 本指南探討了如何使用 Huggingface Trainer API 微調 Transformer 模型，涵蓋設置、訓練及評估過程。 | Huggingface, Transformer, Trainer API, SFT |
| [Huggingface 評估 API](deep_learning/training/huggingface_evaluate_api.md) | 本文介紹了 Huggingface Evaluate API，詳細描述其用途、設置及用於評估機器學習模型的方法。 | Huggingface, Evaluate API, Metric |


## 我們的目標

AI 源動力 (AI Power) 的目標如下：

- **有效理解 AI 演算法**：透過閱讀程式碼，深入理解各種 AI 演算法。
- **快速學習 AI library**：利用大量範例程式，加速掌握各種 AI library。
- **分析程式碼**：解析各種 AI framework 和 application 的程式碼，促進學習。
- **學習模型訓練技巧**：通過豐富的範例程式與經驗分享，快速掌握 AI 模型訓練所需的技巧。
- **MLOps 流程設計**：學習並實踐機器學習運營 (MLOps) 流程設計，提升模型部署和管理的效率與可靠性。
- **系統架構設計**：借助案例學習 AI 應用的系統架構設計，包括軟體和雲端架構設計。

## 如何貢獻

如果您希望協助以下工作，歡迎有志者一同加入：

- **協助翻譯即有文章**: 幫助翻譯和改進現有的文章和教學材料。
- **貢獻新文章**: 每月至少貢獻一篇新文章，分享您的 AI 知識和經驗。

## 加入我們

我們歡迎所有對 AI 技術有興趣的人士加入我們的社群，無論您的經驗水平如何。您的每一份貢獻都將對推動 AI 技術的普及和發展起到重要作用。

## 聯絡我們

如果您有任何問題或建議，請通過 [GitHub Issues](https://github.com/ksmooi/ai_power/issues) 與我們聯繫。

感謝您的參與和支持！