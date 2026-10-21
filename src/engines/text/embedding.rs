//! TextEmbeddingEngine：文本 → 定维 L2 归一化向量（任务语义层）。
//!
//! 职责：池化（[`PoolingKind`]）+ 可选 L2 归一化 + `dim`/`max_length` 访问器；
//! 分词与直跑机制复用 [`super::runner::TextSessionRunner`]（未来 reranking 同样复用）。
//! 池化/归一化实现为模块内**自由函数**（纯逻辑，可无模型单测）。
//!
//! # 确定性口径（设计 §4.4，写死为契约）
//!
//! - [`TextEmbeddingEngine::embed`]：单条自然长度，**任意调用上下文下逐位确定**
//!   （homepix 查询路径恒走此 API）；
//! - [`TextEmbeddingEngine::embed_batch`]：padding 长度 = 批内最长，**同批重放
//!   逐位确定**；跨批组成变化可能引入末位 ulp 漂移（GEMM 分块随矩阵尺寸变化）
//!   ——全库重算按 `source_hash` 整体覆写不做向量比对；若需严格跨批逐位稳定，
//!   路由层循环调 `embed()`。
//!
//! **不实现 `OnnxInferenceEngine` trait**：该 trait 是图像向契约（`predict(&self,
//! &Image)`），纯文本引擎不是视觉任务；下游引擎池只要求 `Send + Sync + 'static`。

use crate::core::device_type::DeviceType;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::error::{Result, VisionError};

use super::runner::{
    OutputSnapshot, TextSessionRunner, OUT_LAST_HIDDEN_STATE, OUT_POOLER_OUTPUT,
};

/// bge 约定默认截断上限（BERT `max_position_embeddings` = 512，含特殊 token）。
const DEFAULT_MAX_LENGTH: usize = 512;

/// 池化方式（构造参数而非写死：池化选错是最典型的静默劣化，设计 §3.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingKind {
    /// CLS 池化：取每条序列首 token hidden（[:,0,:]）。
    /// bge 官方路径（FlagEmbedding BGEEncoder = CLSPooler + normalize）；
    /// 2026-10-21 gate-1 实测 CLS 语义对比度高于 MaskedMean（cos 0.4899/0.1927
    /// vs 0.4146/0.1823）。
    Cls,
    /// Masked mean 池化：attention_mask 加权平均，pad 位贡献 0（m3e 系约定）。
    MaskedMean,
    /// 直接读模型导出的池化输出分支（`pooler_output`，[B,H]）。
    /// 本批 bge 导出**无**此分支（仅 `last_hidden_state`）；其他导出含池化分支时
    /// 选此模式。**仍要求模型导出 `last_hidden_state`**——`dim` 在构造期从其
    /// 声明 shape 解析（runner.rs），缺该分支在**构造期**即报错（而非推理期）。
    PooledOutput,
}

/// 文本 Embedding 引擎：文本批/单条 → 定维（可选 L2 归一化）向量。
///
/// 默认 bge 约定：CLS 池化 + 归一化 + max_length 512。
pub struct TextEmbeddingEngine {
    runner: TextSessionRunner,
    pooling: PoolingKind,
    normalize: bool,
}

impl TextEmbeddingEngine {
    /// bge 约定默认创建：CLS 池化 + L2 归一化 + max_length 512。
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::with_options(
            model_path,
            tokenizer_path,
            device_type,
            PoolingKind::Cls,
            true,
            DEFAULT_MAX_LENGTH,
            OnnxRuntimeConfig::defaults(),
        )
    }

    /// 全参创建（池化 / 归一化 / 截断上限 / 线程与设备参数）。
    pub fn with_options(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        pooling: PoolingKind,
        normalize: bool,
        max_length: usize,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        let runner = TextSessionRunner::new(
            model_path,
            tokenizer_path,
            device_type,
            max_length,
            runtime_config,
        )?;
        Ok(TextEmbeddingEngine {
            runner,
            pooling,
            normalize,
        })
    }

    /// 单条文本 → 向量（自然长度：单元素批的"批内最长"即自身编码长度）。
    ///
    /// 确定性口径：**任意调用上下文下逐位确定**（homepix 查询路径恒走此 API）。
    /// 长度为 [`Self::dim`]；归一化时模长为 1。
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut rows = self.embed_batch(&[text])?;
        rows.pop()
            .ok_or_else(|| VisionError::inference("embed_batch 未返回结果"))
    }

    /// 批量文本 → 向量列表（批内最长 padding；masked 池化对 pad 位贡献 0）。
    ///
    /// 确定性口径：**同批重放逐位确定**；跨批组成变化可能引入末位 ulp 漂移
    /// （GEMM 分块随矩阵尺寸变化）。若下游需要严格跨批逐位稳定，循环调
    /// [`Self::embed`] 而非改批组成。
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(VisionError::invalid_argument(
                "embed_batch: texts 批次为空",
            ));
        }
        if texts.iter().any(|t| t.is_empty()) {
            return Err(VisionError::invalid_argument(
                "embed_batch: texts 含空字符串",
            ));
        }
        let batch = texts.len();
        let dim = self.dim();
        let enc = self.runner.encode_batch(texts)?;
        let snap = self.runner.run(&enc, selected_output(self.pooling))?;
        let rows: Vec<Vec<f32>> = match self.pooling {
            PoolingKind::PooledOutput => pooled_rows(&snap, batch, dim)?,
            PoolingKind::Cls => {
                pool_cls(&snap.data, hidden_shape(&snap, batch, dim)?)?
                    .chunks(dim)
                    .map(<[f32]>::to_vec)
                    .collect()
            }
            PoolingKind::MaskedMean => {
                pool_masked_mean(&snap.data, &enc.attention_mask, hidden_shape(&snap, batch, dim)?)?
                    .chunks(dim)
                    .map(<[f32]>::to_vec)
                    .collect()
            }
        };
        if self.normalize {
            Ok(rows.into_iter().map(l2_normalize).collect())
        } else {
            Ok(rows)
        }
    }

    /// 模型输出维度 H（构造期从 `last_hidden_state` 声明 shape 末维解析）。
    pub fn dim(&self) -> usize {
        self.runner.dim()
    }

    /// 截断上限（含特殊 token 计数）。
    pub fn max_length(&self) -> usize {
        self.runner.max_length()
    }
}

// ==================== 池化 / 归一化（纯逻辑自由函数，可无模型单测） ====================

/// 各池化模式读取的模型输出：Cls/MaskedMean 读 `last_hidden_state` 后自行池化；
/// PooledOutput 直接读导出的池化分支。
fn selected_output(pooling: PoolingKind) -> &'static str {
    match pooling {
        PoolingKind::PooledOutput => OUT_POOLER_OUTPUT,
        PoolingKind::Cls | PoolingKind::MaskedMean => OUT_LAST_HIDDEN_STATE,
    }
}

/// hidden 数据长度与 shape 一致性校验（报错带 shape，reid.rs:133 先例）。
fn validate_hidden(hidden: &[f32], (b, l, h): (usize, usize, usize)) -> Result<()> {
    if b * l * h != hidden.len() {
        return Err(VisionError::inference(format!(
            "hidden 数据长度 {} 与 shape ({b}, {l}, {h}) 不符",
            hidden.len()
        )));
    }
    Ok(())
}

/// CLS 池化：取每条序列首 token hidden（[:,0,:]）→ 平铺 [B*H]。
fn pool_cls(hidden: &[f32], shape: (usize, usize, usize)) -> Result<Vec<f32>> {
    let (b, l, h) = shape;
    validate_hidden(hidden, shape)?;
    let mut out = Vec::with_capacity(b * h);
    for i in 0..b {
        let base = i * l * h; // 第 i 条序列的首 token
        out.extend_from_slice(&hidden[base..base + h]);
    }
    Ok(out)
}

/// Masked mean 池化：attention_mask 加权平均，pad 位贡献 0 → 平铺 [B*H]。
///
/// 同内容不同 padding 长度结果**逐位相等**（pad 位 +0 且除数只数有效 token，
/// f32 精确）；全 pad 序列保持零向量（[`l2_normalize`] 零向量安全）。
fn pool_masked_mean(
    hidden: &[f32],
    mask: &[i64],
    shape: (usize, usize, usize),
) -> Result<Vec<f32>> {
    let (b, l, h) = shape;
    validate_hidden(hidden, shape)?;
    if mask.len() != b * l {
        return Err(VisionError::inference(format!(
            "attention_mask 长度 {} 与 shape ({b}, {l}, {h}) 不符",
            mask.len()
        )));
    }
    let mut out = vec![0f32; b * h];
    for i in 0..b {
        let row = i * l;
        let hidden_base = row * h;
        let out_base = i * h;
        let mut count = 0f32;
        for t in 0..l {
            if mask[row + t] != 0 {
                count += 1.0;
                for k in 0..h {
                    out[out_base + k] += hidden[hidden_base + t * h + k];
                }
            }
        }
        if count > 0.0 {
            for v in out[out_base..out_base + h].iter_mut() {
                *v /= count;
            }
        }
    }
    Ok(out)
}

/// L2 归一化（零向量保持为零，幂等；reid.rs:225 同款语义）。
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

// ==================== 输出 shape 解析（纯逻辑） ====================

/// `last_hidden_state` 快照 → (B, L, H)：3 维 + 批大小 + H==dim 校验（报错带 shape）。
fn hidden_shape(
    snap: &OutputSnapshot,
    batch: usize,
    dim: usize,
) -> Result<(usize, usize, usize)> {
    if snap.shape.len() != 3 {
        return Err(VisionError::inference(format!(
            "last_hidden_state 期望 3 维 [B,L,H]，实际 shape={:?}",
            snap.shape
        )));
    }
    let (b, l, h) = (
        snap.shape[0] as usize,
        snap.shape[1] as usize,
        snap.shape[2] as usize,
    );
    if b != batch || h != dim || l == 0 {
        return Err(VisionError::inference(format!(
            "last_hidden_state 形状不符: shape={:?}，期望 [batch={batch}, L, {dim}]",
            snap.shape
        )));
    }
    Ok((b, l, h))
}

/// 池化输出分支快照（[B,H]）→ B 行向量（PooledOutput 路径；形状不符报错带 shape）。
fn pooled_rows(snap: &OutputSnapshot, batch: usize, dim: usize) -> Result<Vec<Vec<f32>>> {
    if snap.shape.len() != 2 || snap.shape[0] as usize != batch || snap.shape[1] as usize != dim
    {
        return Err(VisionError::inference(format!(
            "池化输出分支形状不符: shape={:?}，期望 [batch={batch}, {dim}]",
            snap.shape
        )));
    }
    Ok(snap.data.chunks(dim).map(<[f32]>::to_vec).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== 纯逻辑单测（无需模型） ====================

    #[test]
    fn masked_mean_should_ignore_padding() {
        // 同内容（3 个有效 token）不同 padding 长度 → 逐位相等
        let natural = pool_masked_mean(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 1], (1, 3, 2))
            .unwrap();
        // 手工 pad 到 [1,5,2]：补两个全零 token 位 + mask 尾 0
        let padded_hidden = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];
        let padded =
            pool_masked_mean(&padded_hidden, &[1, 1, 1, 0, 0], (1, 5, 2)).unwrap();
        assert_eq!(natural, padded, "pad 位贡献 0，同内容不同 padding 应逐位相等");
        assert_eq!(natural, vec![3.0, 4.0]); // (1+3+5)/3, (2+4+6)/3
    }

    #[test]
    fn cls_pool_should_take_first_token() {
        // [2,3,2]：每条序列取首 token hidden（[:,0,:]）
        let hidden = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = pool_cls(&hidden, (2, 3, 2)).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 7.0, 8.0]);
    }

    #[test]
    fn l2_normalize_zero_safe_and_idempotent() {
        // 对齐 reid.rs 既有断言语义
        let v = l2_normalize(vec![3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        // 幂等：已归一化向量再归一化不变
        let v2 = l2_normalize(v.clone());
        assert!(v.iter().zip(v2.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
        // 零向量保持为零
        assert_eq!(l2_normalize(vec![0.0; 4]), vec![0.0; 4]);
    }

    #[test]
    fn dim_mismatch_should_error_with_shape() {
        // shape (2,3,2) 需 12 元素，只给 11 → 报错文案带 shape 数字
        let err = pool_cls(&[0.0; 11], (2, 3, 2)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains('2') && msg.contains('3') && msg.contains("11"),
            "报错应含 shape 数字: {msg}"
        );
    }

    #[test]
    fn pooled_output_kind_should_read_pooled_branch() {
        // PooledOutput 分支选池化输出名；Cls/MaskedMean 走 last_hidden_state
        assert_eq!(selected_output(PoolingKind::PooledOutput), "pooler_output");
        assert_eq!(selected_output(PoolingKind::Cls), "last_hidden_state");
        assert_eq!(selected_output(PoolingKind::MaskedMean), "last_hidden_state");
        // [B,H] 分支拆行 + 形状不符报错带 shape
        let snap = OutputSnapshot {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
        };
        let rows = pooled_rows(&snap, 2, 2).unwrap();
        assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let bad = OutputSnapshot {
            data: vec![1.0, 2.0, 3.0],
            shape: vec![3, 1],
        };
        // 批大小不符（shape[0]=3 vs batch=2）→ 报错带 shape 数字
        let err = pooled_rows(&bad, 2, 1).unwrap_err().to_string();
        assert!(err.contains('3') && err.contains('2'), "报错应含 shape 数字: {err}");
    }

    // ==================== 真实模型自验（缺模型自动跳过，reid.rs:317 同款） ====================

    /// 自验（模型就位时真跑；`MODEL_DIR` 约定同 tests/engines_smoke.rs，默认 `models`）：
    /// 1. `dim() == 512`；
    /// 2. `embed` 归一化模长 ≈ 1（±1e-3）；
    /// 3. 同条两次 `embed()` 逐位相等（f32 位级 ==）；
    /// 4. 同批 `embed_batch()` 重放逐位相等；
    /// 5. 语义判别：cos("在海边玩","沙滩上堆城堡") > cos("在海边玩","财务报表表格截图")；
    /// 6. 交叉验证：`embed("在海边玩")` 前 8 分量与 gate-1 python 参照向量差 < 1e-3。
    ///
    /// 运行：`cargo test --lib engines::text -- --nocapture`（模型放
    /// `models/bge-small-zh-v1.5/` 或 `MODEL_DIR` 指向处）。
    #[test]
    fn text_embedding_real_model_self_check() {
        let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models".to_string());
        let model_path =
            std::path::PathBuf::from(&model_dir).join("bge-small-zh-v1.5/model.onnx");
        let tokenizer_path =
            std::path::PathBuf::from(&model_dir).join("bge-small-zh-v1.5/tokenizer.json");
        if !(model_path.exists() && tokenizer_path.exists()) {
            eprintln!("[SKIP] 模型缺失（{model_dir}/bge-small-zh-v1.5/）");
            return;
        }

        let engine =
            TextEmbeddingEngine::new(&model_path, &tokenizer_path, DeviceType::Cpu).unwrap();

        // 1. dim（gate-1 实测 hidden=512）
        assert_eq!(engine.dim(), 512, "bge-small-zh-v1.5 hidden 维度应为 512");
        assert_eq!(engine.max_length(), 512);

        // 2. 归一化模长 ≈ 1
        let a = engine.embed("在海边玩").unwrap();
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "embedding 模长应为 1，实际 {norm}");

        // 3. 单条两次逐位相等（f32 位级 ==，确定性口径 §4.4）
        let a2 = engine.embed("在海边玩").unwrap();
        assert!(
            a.iter().zip(a2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
            "同条两次 embed() 应逐位相等"
        );

        // 4. 同批重放逐位相等
        let texts = ["在海边玩", "沙滩上堆城堡", "财务报表表格截图"];
        let batch1 = engine.embed_batch(&texts).unwrap();
        let batch2 = engine.embed_batch(&texts).unwrap();
        assert!(
            batch1
                .iter()
                .zip(batch2.iter())
                .all(|(r1, r2)| r1.iter().zip(r2.iter()).all(|(x, y)| x.to_bits() == y.to_bits())),
            "同批 embed_batch() 重放应逐位相等"
        );

        // 5. 语义判别（失败 = 池化/预处理有错，回 gate-1 结论复查）
        let cos = |x: &[f32], y: &[f32]| -> f32 {
            let dot: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
            let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
            dot / (nx * ny)
        };
        let sim_beach = cos(&batch1[0], &batch1[1]);
        let sim_finance = cos(&batch1[0], &batch1[2]);
        eprintln!("bge-small-zh-v1.5 自验：dim={} 模长={norm:.4}", engine.dim());
        eprintln!("  cos(在海边玩, 沙滩上堆城堡)    = {sim_beach:.4}");
        eprintln!("  cos(在海边玩, 财务报表表格截图) = {sim_finance:.4}");
        assert!(
            sim_beach > sim_finance,
            "语义判别失败：beach={sim_beach:.4} 应大于 finance={sim_finance:.4}"
        );

        // 6. 交叉验证：前 8 分量与 gate-1 python 参照向量差 < 1e-3。
        // 常数来源：/tmp/bge_ref_vectors.json vectors[0]（"在海边玩"的 CLS 归一化
        // 向量；2026-10-21 gate-1，python onnxruntime + 同 tokenizer.json）
        const REF_PREFIX: [f32; 8] = [
            0.05056403577327728,
            -0.014244092628359795,
            0.03130625560879707,
            0.03745114058256149,
            -0.017695851624011993,
            -0.013662866316735744,
            -0.003380832727998495,
            0.03465394303202629,
        ];
        for (i, (got, want)) in a.iter().zip(REF_PREFIX.iter()).enumerate().take(8) {
            assert!(
                (got - want).abs() < 1e-3,
                "前 8 分量第 {i} 个与 gate-1 参照不符: got={got} want={want}"
            );
        }
    }
}
