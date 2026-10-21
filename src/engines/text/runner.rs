//! TextSessionRunner：BERT 族文本会话直跑（分词 → int64 三输入 → 锁 session 直跑
//! → `last_hidden_state` 快照）。
//!
//! 分词与张量构造拆成**不依赖 session 的自由函数**（[`encode_texts`] /
//! [`to_i64_tensors`]）：`BaseOnnxEngine` 构造必须加载真实 ONNX 文件，分词逻辑
//! 脱离它才能用内存词表无模型单测（设计 §4.2 可测性关键设计）。
//!
//! runner 只负责输入侧机制；池化/归一化等任务语义在 `embedding.rs`（未来
//! reranking 落地时复用本 runner，仅加 `encode_pairs`）。

use ort::value::Tensor;
use tokenizers::{
    Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::error::{Result, VisionError};

/// BERT 族模型 I/O 名称（Xenova / onnx-community 导出约定，2026-10-21 gate-1 实测）。
pub(crate) const IN_INPUT_IDS: &str = "input_ids";
pub(crate) const IN_ATTENTION_MASK: &str = "attention_mask";
pub(crate) const IN_TOKEN_TYPE_IDS: &str = "token_type_ids";
pub(crate) const OUT_LAST_HIDDEN_STATE: &str = "last_hidden_state";
/// PooledOutput 模式读取的池化输出分支名（BERT 导出约定；本批 bge 导出无此分支）。
pub(crate) const OUT_POOLER_OUTPUT: &str = "pooler_output";
/// [PAD] token id（BERT 词表约定 id=0；手工 padding 与 masked 池化共用）。
pub(crate) const PAD_ID: i64 = 0;

/// 一批文本的分词结果：三路 i64 平铺数据 + 统一序列长度。
///
/// 批大小 B = `input_ids.len() / seq_len`（padding 均匀，整除恒成立）。
#[derive(Debug)]
pub(crate) struct EncodedBatch {
    pub(crate) input_ids: Vec<i64>,
    pub(crate) attention_mask: Vec<i64>,
    pub(crate) token_type_ids: Vec<i64>,
    /// 统一序列长度 L（批内最长 / 单条自然长度）。
    pub(crate) seq_len: usize,
}

/// 分词 + 批内最长 padding（自由函数：不依赖 session，可无模型单测）。
///
/// tokenizer 需已配置 truncation（runner 构造时统一配置）；padding 由本函数
/// **显式手工控制**（不依赖 tokenizer 的 padding 状态，避免共享 tokenizer 的
/// 并发突变）：批内最长者保持自然长度，短句尾部补 `ids=[PAD](0) / mask=0 /
/// type=0`。`encode` 自动加特殊 token（真实 tokenizer.json 的模板后处理）。
pub(crate) fn encode_texts(
    tokenizer: &Tokenizer,
    texts: &[&str],
    max_length: usize,
) -> Result<EncodedBatch> {
    if texts.is_empty() {
        return Err(VisionError::invalid_argument(
            "encode_texts: texts 批次为空",
        ));
    }
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| VisionError::Tokenizer(format!("encode 失败: {e}")))?;

    // 每条截到 max_length（tokenizer 已配 truncation 时通常不触发，此处兜底保证）
    let lens: Vec<usize> = encodings
        .iter()
        .map(|e| e.get_ids().len().min(max_length))
        .collect();
    let seq_len = lens.iter().copied().max().unwrap_or(0);
    if seq_len == 0 {
        return Err(VisionError::invalid_argument(
            "encode_texts: 编码结果为空（max_length 过小？）",
        ));
    }

    let cap = texts.len() * seq_len;
    let mut input_ids = Vec::with_capacity(cap);
    let mut attention_mask = Vec::with_capacity(cap);
    let mut token_type_ids = Vec::with_capacity(cap);
    for (enc, &n) in encodings.iter().zip(lens.iter()) {
        let ids = enc.get_ids();
        let mask = enc.get_attention_mask();
        let types = enc.get_type_ids();
        for pos in 0..seq_len {
            if pos < n {
                input_ids.push(ids[pos] as i64);
                attention_mask.push(mask.get(pos).copied().unwrap_or(1) as i64);
                token_type_ids.push(types.get(pos).copied().unwrap_or(0) as i64);
            } else {
                input_ids.push(PAD_ID);
                attention_mask.push(0);
                token_type_ids.push(0);
            }
        }
    }
    Ok(EncodedBatch {
        input_ids,
        attention_mask,
        token_type_ids,
        seq_len,
    })
}

/// 三路 i64 张量 `[B, L]`（`ort::inputs!` 命名直跑用）。
pub(crate) fn to_i64_tensors(
    b: &EncodedBatch,
) -> Result<(Tensor<i64>, Tensor<i64>, Tensor<i64>)> {
    let batch = (b.input_ids.len() / b.seq_len) as i64;
    let shape = vec![batch, b.seq_len as i64];
    Ok((
        Tensor::from_array((shape.clone(), b.input_ids.clone()))?,
        Tensor::from_array((shape.clone(), b.attention_mask.clone()))?,
        Tensor::from_array((shape, b.token_type_ids.clone()))?,
    ))
}

/// 直跑输出快照：f32 平铺数据 + 运行时 shape（`snapshot_f32` 同款语义，附带形状）。
pub(crate) struct OutputSnapshot {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<i64>,
}

/// 文本会话 runner：持 `BaseOnnxEngine`（仅作 session 容器 + 线程/设备配置；
/// `[B,L]` 二维输入不满足 NCHW 解析，`load_input_info` 跳过并保留无害默认值，
/// 引擎不读它）+ Tokenizer + max_length。
///
/// `Tokenizer` 与 `BaseOnnxEngine` 组合的 Send+Sync 已由 `GroundingDinoEngine`
/// 实证（同字段组合实现了 `Send + Sync` 的 trait）。
pub(crate) struct TextSessionRunner {
    pub(crate) base: BaseOnnxEngine,
    tokenizer: Tokenizer,
    max_length: usize,
    dim: usize,
}

impl TextSessionRunner {
    /// 创建 runner：加载模型与 tokenizer，配置 truncation（Right/LongestFirst，
    /// 含特殊 token 计数），**不设 padding**（由 [`encode_texts`] 显式控制）。
    ///
    /// 构造期解析 `dim`：锁 session 读 `last_hidden_state` 声明 shape 的末维
    /// （`[-1,-1,512]` 末维 512）；拿不到静态末维即报错（fail loud，不返回 Option）。
    pub(crate) fn new(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        max_length: usize,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        let base = BaseOnnxEngine::with_config(
            model_path.as_ref(),
            device_type,
            -1,
            -1,
            runtime_config,
        )?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| VisionError::Tokenizer(format!("加载 tokenizer 失败: {e}")))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(|e| VisionError::Tokenizer(format!("配置 truncation 失败: {e}")))?;

        // dim 解析：构造期读输出声明 shape 末维
        let dim = {
            let session = base.session.lock().unwrap();
            let out = session
                .outputs()
                .iter()
                .find(|o| o.name() == OUT_LAST_HIDDEN_STATE)
                .ok_or_else(|| {
                    VisionError::inference(format!(
                        "模型缺少输出 '{OUT_LAST_HIDDEN_STATE}'，实际输出: {:?}",
                        base.output_names()
                    ))
                })?;
            let dims = match out.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => {
                    shape.iter().copied().collect::<Vec<i64>>()
                }
                other => {
                    return Err(VisionError::inference(format!(
                        "输出 '{OUT_LAST_HIDDEN_STATE}' 非张量: {other:?}"
                    )))
                }
            };
            let last = *dims.last().unwrap_or(&-1);
            if last <= 0 {
                return Err(VisionError::inference(format!(
                    "输出 '{OUT_LAST_HIDDEN_STATE}' 末维非静态，无法解析 dim: shape={dims:?}"
                )));
            }
            last as usize
        };

        tracing::info!(
            "Text session runner initialized: model='{}', outputs={:?}, dim={dim}, max_length={max_length}",
            model_path.as_ref().display(),
            base.output_names()
        );
        Ok(TextSessionRunner {
            base,
            tokenizer,
            max_length,
            dim,
        })
    }

    /// 模型输出维度 H（构造期解析）。
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// 截断上限（含特殊 token）。
    pub(crate) fn max_length(&self) -> usize {
        self.max_length
    }

    /// 分词一批文本（批内最长 padding，见 [`encode_texts`]）。
    pub(crate) fn encode_batch(&self, texts: &[&str]) -> Result<EncodedBatch> {
        encode_texts(&self.tokenizer, texts, self.max_length)
    }

    /// 直跑并按名取 f32 输出快照（锁 session，`ort::inputs!` 命名直跑；
    /// 混合 int64 输入无法走 `BaseOnnxEngine::run_named`，grounding_dino 同款出路）。
    pub(crate) fn run(&self, b: &EncodedBatch, output_name: &str) -> Result<OutputSnapshot> {
        let (ids, mask, types) = to_i64_tensors(b)?;
        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            IN_INPUT_IDS      => ids,
            IN_ATTENTION_MASK => mask,
            IN_TOKEN_TYPE_IDS => types,
        ])?;
        let value = outputs
            .get(output_name)
            .ok_or_else(|| {
                VisionError::inference(format!(
                    "模型缺少输出 '{output_name}'，实际输出: {:?}",
                    self.base.output_names()
                ))
            })?;
        let (shape, view) = value.try_extract_tensor::<f32>()?;
        Ok(OutputSnapshot {
            data: view.to_vec(),
            shape: shape.iter().copied().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{TruncationParams, TruncationDirection, TruncationStrategy};

    /// 内存 WordPiece 小词表 fixture（零文件依赖）：
    /// `[PAD]`=0 `[UNK]`=1 `[CLS]`=2 `[SEP]`=3 海=4 边=5 玩=6 沙=7 堆=8 城=9 堡=10。
    /// Whitespace 预分词（测试文本以空格分字）+ `[CLS] $A [SEP]` 模板后处理 +
    /// truncation 配置与 [`TextSessionRunner::new`] 一致 —— 与真实 BERT 管线同构。
    fn fixture_tokenizer(max_length: usize) -> tokenizers::Tokenizer {
        // vocab 直接用数组（Into<AHashMap>，tokenizers 内部哈希类型）
        let vocab = [
            ("[PAD]".to_string(), 0u32),
            ("[UNK]".to_string(), 1),
            ("[CLS]".to_string(), 2),
            ("[SEP]".to_string(), 3),
            ("海".to_string(), 4),
            ("边".to_string(), 5),
            ("玩".to_string(), 6),
            ("沙".to_string(), 7),
            ("堆".to_string(), 8),
            ("城".to_string(), 9),
            ("堡".to_string(), 10),
        ];
        let wordpiece = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(wordpiece);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]").unwrap()
                .special_tokens(vec![("[CLS]", 2u32), ("[SEP]", 3u32)])
                .build()
                .unwrap(),
        ));
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .unwrap();
        tokenizer
    }

    #[test]
    fn encode_should_truncate_to_max_length() {
        // max_length=5 含特殊 token：7 个内容 token 截到 3 个，总长恰为 5
        let tok = fixture_tokenizer(5);
        let b = encode_texts(&tok, &["海 边 玩 沙 堆 城 堡"], 5).unwrap();
        assert_eq!(b.seq_len, 5, "总长（含 [CLS]/[SEP]）应截到 max_length");
        // Right 截断保留开头：[CLS] 海 边 玩 [SEP]
        assert_eq!(b.input_ids, vec![2, 4, 5, 6, 3]);
        assert_eq!(b.attention_mask, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn encode_batch_should_pad_to_batch_longest() {
        let tok = fixture_tokenizer(32);
        let b = encode_texts(&tok, &["海 边 玩", "沙 堆 城 堡"], 32).unwrap();
        // 批内最长 = [CLS]+4 内容+[SEP] = 6；不 pad 到 max_length
        assert_eq!(b.seq_len, 6);
        assert_eq!(b.input_ids.len(), 12);
        assert_eq!(b.attention_mask.len(), 12);
        // 第一行（自然长 5）尾部 pad：ids=[PAD](0)、mask=0、type=0
        assert_eq!(b.input_ids[..5].to_vec(), vec![2, 4, 5, 6, 3]);
        assert_eq!(b.input_ids[5], 0);
        assert_eq!(b.attention_mask[..5].to_vec(), vec![1, 1, 1, 1, 1]);
        assert_eq!(b.attention_mask[5], 0);
        assert_eq!(b.token_type_ids[5], 0);
        // 第二行占满 6 位无 pad，mask 全 1
        assert_eq!(
            b.input_ids[6..].to_vec(),
            vec![2, 7, 8, 9, 10, 3]
        );
        assert_eq!(b.attention_mask[6..].to_vec(), vec![1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn encode_single_should_keep_natural_length() {
        let tok = fixture_tokenizer(32);
        let b = encode_texts(&tok, &["海 边"], 32).unwrap();
        // 单条路径：L == 自身编码长度（[CLS] 海 边 [SEP]），无任何 padding
        //（确定性口径的锚：embed 单条恒自然长度）
        assert_eq!(b.seq_len, 4);
        assert_eq!(b.input_ids, vec![2, 4, 5, 3]);
        assert!(b.attention_mask.iter().all(|&v| v == 1));
    }

    #[test]
    fn tensors_should_have_batch_seq_shape() {
        let tok = fixture_tokenizer(32);
        let b = encode_texts(&tok, &["海 边 玩", "沙 堆 城 堡"], 32).unwrap();
        let (ids, mask, types) = to_i64_tensors(&b).unwrap();
        // 三路张量形状均为 [B=2, L=批内最长=6]
        assert_eq!(**ids.shape(), [2i64, 6]);
        assert_eq!(**mask.shape(), [2i64, 6]);
        assert_eq!(**types.shape(), [2i64, 6]);
    }

    #[test]
    fn encode_empty_batch_should_error() {
        let tok = fixture_tokenizer(8);
        let err = encode_texts(&tok, &[], 8).unwrap_err();
        assert!(
            matches!(err, VisionError::InvalidArgument(_)),
            "空批次应报 InvalidArgument，实际: {err:?}"
        );
    }
}
