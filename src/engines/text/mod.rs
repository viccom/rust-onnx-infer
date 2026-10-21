//! 文本 Embedding 引擎（BERT 族 encoder ONNX + WordPiece 分词）。
//!
//! 与视觉引擎不同：模型输入为 int64 三路张量（无法走 `BaseOnnxEngine::run_inference`
//! 的 f32 入口），由 [`runner::TextSessionRunner`] 锁 session 直跑（grounding_dino 同款
//! 出路）；输出侧做池化 + L2 归一化，产出定维向量供语义检索。
//!
//! # 模型准备
//!
//! ```text
//! # 从 Hugging Face 下载 Xenova/bge-small-zh-v1.5（MIT）：
//! #   model.onnx       (~90 MB fp32)
//! #   tokenizer.json   (BERT WordPiece tokenizer)
//! # 落位 models/bge-small-zh-v1.5/（沿用 tests/engines_smoke.rs 的 MODEL_DIR 约定）
//! ```
//!
//! # ONNX I/O（Xenova/bge-small-zh-v1.5 实际导出；2026-10-21 gate-1 实测）
//!
//! Inputs:
//!
//! - `input_ids`:       [B, L] int64 — BERT WordPiece + [CLS]...[SEP]（截断上限 max_length，含特殊 token）
//! - `attention_mask`:  [B, L] int64 — 1=token, 0=pad
//! - `token_type_ids`:  [B, L] int64 — 全 0（单句）
//!
//! Outputs:
//!
//! - `last_hidden_state`: [B, L, 512] float32 — 导出无池化输出分支，池化在引擎侧做
//!
//! # 池化与确定性约定
//!
//! - 默认 **CLS 池化 + L2 归一化**：FlagEmbedding BGEEncoder 的官方路径是
//!   CLSPooler + normalize（m3e 系才是 mean pooling）；2026-10-21 gate-1 实测
//!   CLS 语义对比度高于 MaskedMean（cos 0.4899/0.1927 vs 0.4146/0.1823）；
//! - [`TextEmbeddingEngine::embed`] 单条自然长度，**任意调用上下文逐位确定**；
//!   [`TextEmbeddingEngine::embed_batch`] 批内最长 padding，**同批重放逐位确定**、
//!   跨批组成变化可能 ulp 漂移（详见 embedding.rs 的确定性口径文档）。
//!
//! # 示例
//!
//! ```ignore
//! let engine = TextEmbeddingEngine::new(
//!     "models/bge-small-zh-v1.5/model.onnx",
//!     "models/bge-small-zh-v1.5/tokenizer.json",
//!     DeviceType::Cpu,
//! )?;
//! let v = engine.embed("在海边玩")?; // 512 维 L2 归一化向量
//! ```

pub mod embedding;
pub mod runner;

pub use embedding::{PoolingKind, TextEmbeddingEngine};

// 编译期锁定：下游引擎池以 Arc 共享并发调用，本引擎必须 Send + Sync。
// 引擎未实现任何带 Send+Sync bound 的 trait（设计 §5），失去 grounding_dino
// 那种 trait bound 天然强制，须显式断言——含 TextSessionRunner 字段，一层锁两层。
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TextEmbeddingEngine>();
};
