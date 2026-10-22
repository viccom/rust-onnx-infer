//! TextRerankingEngine：query × docs 交叉编码重排（cross-encoder，seq-cls 头）。
//!
//! 契约：单查询批模式——query 与每条 doc 组 pair（模板序由 tokenizer 后处理
//! 决定，两族 reranker 为 `<s>q</s></s>d</s>`），批内最长 padding，一批一前向
//! 读 `logits [B,1]`，sigmoid → `Vec<f32>`（**输入序**，长度恒等于
//! docs.len()）。分词/直跑机制复用 [`super::runner::TextSessionRunner`]
//! （pair 编码走 [`super::runner::encode_pairs`]，pad id 从词表解析——XLM-R
//! 族 `<pad>`≠0，pad 值错误会造成 int8 logits 抖动）。
//!
//! # 确定性口径（与 embedding 的逐条循环口径不同，设计 §7）
//!
//! - 同一请求（同一 docs 数组、同一顺序）重放**逐位确定**；
//! - 分数依赖整个批的 padding 形状——**docs 数组组成/顺序变化会导致分数
//!   漂移**（int8 实测 logits 变动 0.08-0.33，2026-10-22；fp32 更小）。
//!   需要跨请求可比分数时保持批组成一致。
//!
//! fail loud 双层：构造期校验 `logits` 输出存在且声明末维 == 1（防模型文件
//! 挂错，见 [`TextSessionRunner::new_for_output`]）；运行期校验 shape ==
//! `[B,1]` 且数据长度 == B（[`validate_logits_shape`]）。
//!
//! **不实现 `OnnxInferenceEngine` trait**（同 embedding.rs：图像向契约），
//! 下游引擎池只要求 `Send + Sync + 'static`。

use crate::core::device_type::DeviceType;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::error::{Result, VisionError};

use super::runner::{OutputSnapshot, TextSessionRunner, OUT_LOGITS};

/// sigmoid 激活：1 / (1 + e^-x)（两族 reranker 官方约定一致，输出概率分数；
/// 纯函数可单测：0→0.5、大正→饱和 1、大负→饱和 0、关于 (0,0.5) 中心对称）。
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 文本 Reranking 引擎：query × docs → 每对 relevance 概率分数（输入序）。
///
/// 输出恒经 sigmoid，不提供原始 logits 开关（两族官方约定一致，YAGNI）。
pub struct TextRerankingEngine {
    runner: TextSessionRunner,
}

impl TextRerankingEngine {
    /// 全参创建（模型 / tokenizer / 设备 / 截断上限 / 线程参数）。
    ///
    /// 内部走 [`TextSessionRunner::new_for_output`] 传 `(OUT_LOGITS, Some(1))`：
    /// 模型缺 `logits` 输出或声明末维 ≠ 1 → 构造期报错拒载。
    pub fn with_options(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        max_length: usize,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        let runner = TextSessionRunner::new_for_output(
            model_path,
            tokenizer_path,
            device_type,
            max_length,
            runtime_config,
            OUT_LOGITS,
            Some(1),
        )?;
        Ok(TextRerankingEngine { runner })
    }

    /// 重排：query 与每条 doc pair 编码 → 一批一前向 → sigmoid → 分数
    /// （**输入序**，长度 == docs.len()）。
    ///
    /// 确定性口径见模块注释：同批重放逐位确定；批组成变化分数会漂移。
    pub fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Err(VisionError::invalid_argument("rerank: docs 批次为空"));
        }
        let batch = docs.len();
        let enc = self.runner.encode_pair_batch(query, docs)?;
        let snap = self.runner.run(&enc, OUT_LOGITS)?;
        validate_logits_shape(&snap, batch)?;
        Ok(snap.data.iter().map(|&logit| sigmoid(logit)).collect())
    }

    /// 批内最长 pair 编码长度（含 query 与特殊 token，已应用 maxlen 截断）。
    ///
    /// 供 HTTP 层做 token 预算校验（`B × max_pair_len ≤ 预算`——批内按最长
    /// padding，真实计算量是 B×L 而非长度之和）；tokenization 是 CPU 密集
    /// 操作，调用方须在 `spawn_blocking` 内执行（设计 §5 注意 3）。
    pub fn max_pair_len(&self, query: &str, docs: &[&str]) -> Result<usize> {
        if docs.is_empty() {
            return Err(VisionError::invalid_argument(
                "max_pair_len: docs 批次为空",
            ));
        }
        Ok(self.runner.encode_pair_batch(query, docs)?.seq_len)
    }

    /// 截断上限（含特殊 token 计数）。
    pub fn max_length(&self) -> usize {
        self.runner.max_length()
    }
}

/// logits 快照形状校验：shape == `[B,1]` 且 `data.len() == B`（纯逻辑，
/// 可无模型单测；不符报 inference 错——快照与批大小脱节属推理结果不可信）。
fn validate_logits_shape(snap: &OutputSnapshot, batch: usize) -> Result<()> {
    if snap.shape.len() != 2
        || snap.shape[0] as usize != batch
        || snap.shape[1] != 1
        || snap.data.len() != batch
    {
        return Err(VisionError::inference(format!(
            "logits 形状不符: shape={:?}（{} 个 f32），期望 [batch={batch}, 1]",
            snap.shape,
            snap.data.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== sigmoid 数学（纯函数） ====================

    #[test]
    fn sigmoid_should_map_zero_to_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) 应为 0.5");
    }

    #[test]
    fn sigmoid_should_saturate_at_extremes() {
        assert!((sigmoid(20.0) - 1.0).abs() < 1e-6, "大正输入应饱和到 1");
        assert!(sigmoid(-20.0).abs() < 1e-6, "大负输入应饱和到 0");
    }

    #[test]
    fn sigmoid_should_be_point_symmetric() {
        // sigmoid(-x) = 1 - sigmoid(x)（关于 (0, 0.5) 中心对称）
        for &x in &[0.5f32, 1.0, 2.7, -3.9, 12.0] {
            assert!(
                (sigmoid(x) - (1.0 - sigmoid(-x))).abs() < 1e-6,
                "sigmoid({x}) 应等于 1 - sigmoid(-{x})"
            );
        }
    }

    // ==================== logits shape 校验（纯函数） ====================

    #[test]
    fn logits_shape_should_be_validated() {
        // 合法：shape [B,1] 且 data.len()==B
        let ok = OutputSnapshot {
            data: vec![0.5, 2.0],
            shape: vec![2, 1],
        };
        assert!(validate_logits_shape(&ok, 2).is_ok());
        // 末维 ≠ 1
        let bad_dim = OutputSnapshot {
            data: vec![0.0; 4],
            shape: vec![2, 2],
        };
        assert!(validate_logits_shape(&bad_dim, 2).is_err());
        // 批大小不符
        let bad_batch = OutputSnapshot {
            data: vec![0.0],
            shape: vec![1, 1],
        };
        assert!(validate_logits_shape(&bad_batch, 2).is_err());
        // data.len() != B
        let bad_len = OutputSnapshot {
            data: vec![0.0; 3],
            shape: vec![2, 1],
        };
        assert!(validate_logits_shape(&bad_len, 2).is_err());
        // 维数 ≠ 2
        let bad_rank = OutputSnapshot {
            data: vec![0.0; 2],
            shape: vec![2, 1, 1],
        };
        assert!(validate_logits_shape(&bad_rank, 2).is_err());
    }
}
