[ [English](README.md) ] [ [Español](README_es.md) ] [ [简体中文](README_zhcn.md) ] [ [繁體中文](README_zhtw.md) ] [ [한국어](README_ko.md) ] 

# AI Power

AI テクノロジーは驚異的な速さで進化しており、新しいアルゴリズムやAIライブラリが次々と登場し、進化しています。より多くの人々が最新のAI技術を習得し、オープンソースプロジェクトに積極的に参加できるようにするために、私はAI パワーソースを作成しました。最先端のAIテクノロジーを一緒に探求し、未来を形作るために貢献しましょう！

## 記事カテゴリ

### コード駆動アルゴリズム学習

| タイトル                 | 説明           | キーワード              |
|-----------------------|-----------------------|-----------------------|
| [Understanding Transformer Attentions](deep_learning/transformer/transformer_attentions.md)                   | トランスフォーマーにおけるアテンションメカニズムの詳細な説明。セルフアテンション、マルチヘッドアテンション、および現代のNLPモデルにおけるそれらの実装について解説。 | Transformers, Self-Attention, MHA |
| [The Vanilla Transformer Explained](deep_learning/transformer/vanilla_transformer_explained_enus.md)       | バニラトランスフォーマーモデルの包括的ガイド。アーキテクチャ、コンポーネント、およびシーケンス間タスクのためのフォワードパスプロセスについて詳細に説明。 | vanilla Transformer, Architecture, Sequence-to-Sequence |
| [Inside CLIP](deep_learning/transformer/model_multimodal/huggingface_clip_explained.md)               | CLIPモデルの詳細な説明。アーキテクチャ、トレーニングプロセス、および画像とテキストをリンクするアプリケーションについて解説。 | CLIP, Architecture |
| [Deep Dive into LLaVA](deep_learning/transformer/model_multimodal/llava_implementation_explained.md)           | LLaVAモデルの実装に関する包括的なガイド。アーキテクチャ、コンポーネント、および言語理解タスクの向上について探求。 | LLaVA, Architecture, MultiModal |
| [Deep Dive into Vision Transformer](deep_learning/transformer/model_vision/huggingface_vit_explained.md)                | Vision Transformer (ViT) モデルの詳細な説明。アーキテクチャ、主要コンポーネント、およびコンピュータビジョンタスクへの応用について解説。 | Vision Transformer, ViT, Architecture |
| [AutoEncoder Explained](deep_learning/diffusion/autoencoder_explained.md)                    | オートエンコーダーの詳細な説明。アーキテクチャ、タイプ、およびデータ圧縮と特徴学習における応用について解説。 | AutoEncoder, VAE, Architecture |

### Huggingface API

| タイトル                 | 説明           | キーワード              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface SBERT API - Part 1](deep_learning/transformer/huggingface_sbert_api_part1.md) | Huggingface Sentence-BERT (SBERT) APIの紹介。目的、アプリケーション、および文の埋め込みと類似性の計算方法について解説。 | SBERT, Sentence Transformer, Embeddings         |
| [Huggingface SBERT API - Part 2](deep_learning/transformer/huggingface_sbert_api_part2.md) | SBERT APIガイドの続編。高度な使用法、モデルの微調整、および様々なアプリケーションとの統合について解説。 | SBERT, Sentence Transformer, Embeddings |
| [Huggingface Transformer Auto Class API - Part 1](deep_learning/transformer/huggingface_transformer_auto_class_api_part1.md) | Huggingface Transformer Auto Class APIの概要。特徴、セットアップ、および様々なNLPタスクにおける基本的な使用法について解説。 | Transformer, Auto Class, NLP, API             |
| [Huggingface Transformer Auto Class API - Part 2](deep_learning/transformer/huggingface_transformer_auto_class_api_part2.md) | Transformer Auto Class APIの詳細。カスタム設定、モデルの最適化、および使用例について探求。 | Transformer, Auto Class, NLP, API |
| [Huggingface Transformer Pipeline API](deep_learning/transformer/huggingface_transformer_pipeline_api.md) | Huggingface Transformer Pipeline APIのガイド。テキスト分類、固有表現認識、テキスト生成などの様々なNLPタスクにおける使いやすさを紹介。 | Transformer Pipeline |
| [Huggingface CLIP API](deep_learning/transformer/model_multimodal/huggingface_clip_api.md)                     | Huggingface CLIP (Contrastive Language-Image Pre-training) APIの紹介。目的、アプリケーション、および画像とテキストの埋め込み方法について解説。 | Huggingface, CLIP, API |
| [Huggingface LLaVA Next API](deep_learning/transformer/model_multimodal/huggingface_llava_next_api.md)               | Huggingface LLaVA Next APIのガイド。特徴、セットアップ、および高度な言語理解タスクのための使用法について解説。 | LLaVA, MultiModal, API |
| [Huggingface Vision Transformer (ViT) API](deep_learning/transformer/model_vision/huggingface_vit_api.md)       | Huggingface Vision Transformer (ViT) APIの概要。画像分類および他のビジョンタスクにおける使用法について解説。 | ViT, Vision Transformer, Image Classification, API |
| [Huggingface Diffusers API](deep_learning/diffusion/huggingface_diffusers_api.md)                | Huggingface Diffusers APIの概要。機能およびテキスト記述から画像を生成するための使用法について解説。 | Diffusers API, Image Generation, Text-to-Image, Image-to-Image |
| [Huggingface Diffusers Chained Pipeline](deep_learning/diffusion/huggingface_diffusers_chained_pipeline.md)   | Huggingface Diffusers APIを使用して連鎖パイプラインを作成する方法についてのガイド。複数のモデルを組み合わせて複雑なタスクを処理する方法を紹介。 | Diffusers API, Chained Pipeline |
| [Huggingface Diffusers Pipeline API](deep_learning/diffusion/huggingface_diffusers_pipeline_api.md)       | Huggingface Diffusers Pipeline APIの詳細。特徴、セットアップ、および画像とテキスト生成のためのアプリケーションについて解説。 | Diffusers Pipeline API |

### テンソル操作

| タイトル                 | 説明           | キーワード              |
|-----------------------|-----------------------|-----------------------|
| [Einops Einsum](deep_learning/coding/einops_einsum.md)                            | einopsライブラリのeinsum関数の紹介。構文、使用法、およびテンソル操作における応用について解説。 | einops, einsum, tensor operations |
| [Einops Rearrange](deep_learning/coding/einops_rearrange.md)                         | einopsライブラリのrearrange関数のガイド。テンソル形状を効率的に操作および変換する方法を紹介。 | einops, einsum, tensor operations |

### データセット

| タイトル                 | 説明           | キーワード              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface Datasets Loading](deep_learning/dataset/huggingface_datasets_loading.md)           | Huggingface Datasetsライブラリを使用してデータセットをロードおよび前処理する方法についての説明。様々なデータ形式とソースの取り扱いを含む。 | Huggingface, Datasets |
| [Huggingface Datasets Main Classes](deep_learning/dataset/huggingface_datasets_main_classes.md) | Huggingface Datasetsライブラリの主要クラスについての包括的なガイド。機能と使用例を解説。 | Huggingface, Datasets, Main Classes |

### モデル訓練

| タイトル                 | 説明           | キーワード              |
|-----------------------|-----------------------|-----------------------|
| [Huggingface Transformer Trainer API](deep_learning/training/huggingface_transformer_trainer_finetune.md) | Huggingface Trainer APIを使用してトランスフォーマーモデルを微調整する方法についてのガイド。セットアップ、トレーニング、および評価プロセスをカバー。 | Huggingface, Transformer, Trainer API, SFT |
| [Huggingface Evaluate API](deep_learning/training/huggingface_evaluate_api.md) | Huggingface Evaluate APIの紹介。目的、セットアップ、および機械学習モデルの評価のための使用法について解説。 | Huggingface, Evaluate API, Metric |


## 私たちの目標

AI パワーソースの目標は以下の通りです：

- **効果的にAIアルゴリズムを理解する**：コードを読むことで、さまざまなAIアルゴリズムを深く理解します。
- **AIライブラリを迅速に学習する**：多数の例題プログラムを使用して、さまざまなAIライブラリの習得を加速します。
- **コードを分析する**：さまざまなAIフレームワークやアプリケーションのコードを分析し、学習を促進します。
- **モデル訓練の技術を学ぶ**：豊富な例題プログラムと経験共有を通じて、AIモデル訓練に必要なスキルを迅速に習得します。
- **MLOps プロセス設計**：モデルの展開と管理の効率と信頼性を向上させるために、MLOps プロセス設計を学び、実践します。
- **システムアーキテクチャ設計**：ケーススタディを通じて、AIアプリケーションのシステムアーキテクチャ設計、特にソフトウェアアーキテクチャとクラウドアーキテクチャ設計を学びます。

## 貢献する方法

以下の作業を支援したい場合は、ぜひご参加ください：

- **既存の記事の翻訳を支援する**：既存の記事や教育資料の翻訳と改善を手伝います。
- **新しい記事を投稿する**：毎月少なくとも1つの新しい記事を投稿し、あなたのAI知識と経験を共有してください。

## 参加する

AI技術に興味があるすべての人々が、経験レベルに関係なく、私たちのコミュニティに参加することを歓迎します。あなたの貢献の一つ一つが、AI技術の普及と発展に重要な役割を果たします。

## 連絡先

質問や提案がある場合は、[GitHub Issues](https://github.com/ksmooi/ai_power/issues) を通じてお問い合わせください。

ご参加とご支援に感謝します！