//! 文本 Embedding 推理速度基准（CPU，`TextEmbeddingEngine` 默认 Cls + L2 路径）。
//!
//! 与 gate-1 python 探针（2026-10-21，/tmp/argus-bge-probe.py）同口径的三档单条
//! 计时（端到端：分词 + 推理 + 池化 + 归一化），另加批量 `embed_batch` 与逐条
//! `embed()` 循环对照。token 数运行时按实际分词打印（含特殊 token），便于对表。
//!
//! 用法：
//!
//! ```text
//! cargo run --release --example text_embed_bench -- <model.onnx> <tokenizer.json> [rounds=100]
//! # 例如：
//! cargo run --release --example text_embed_bench -- \
//!     models/bge-small-zh-v1.5/model.onnx models/bge-small-zh-v1.5/tokenizer.json 100
//! ```

use std::hint::black_box;
use std::time::Instant;

use rust_onnx_infer::engines::text::TextEmbeddingEngine;
use rust_onnx_infer::DeviceType;

/// 三档测试文本（对齐 gate-1 的 短/中/长 档；实际 token 数运行时打印）
const TEXT_SHORT: &str = "海边 秋千";
const TEXT_MEDIUM: &str = "海浪轻轻拍打着沙滩，孩子们在沙滩上堆起城堡，笑声随海风飘远。";
const TEXT_LONG: &str =
    "夕阳西下，海面上的金光随着波浪起伏，远处的帆船缓缓归航，沙滩上的脚印被潮水轻轻抹去，海风带来咸咸的气息，人们渐渐散去。";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "用法: cargo run --release --example text_embed_bench -- <model.onnx> <tokenizer.json> [rounds=100]"
        );
        std::process::exit(2);
    }
    let model = &args[0];
    let tokenizer_path = &args[1];
    let rounds: usize = args
        .get(2)
        .map(|s| s.parse().expect("rounds 应为正整数"))
        .unwrap_or(100);

    let t = Instant::now();
    let engine = TextEmbeddingEngine::new(model, tokenizer_path, DeviceType::Cpu).unwrap();
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;

    // 分词统计（token 数含特殊 token；与 gate-1 探针口径一致）
    let counter = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();
    let token_len = |s: &str| counter.encode(s, true).unwrap().get_ids().len();

    println!("=== 文本 Embedding 基准（TextEmbeddingEngine 默认 Cls + L2，CPU）===");
    println!(
        "模型: {model}（加载 {load_ms:.0}ms，dim={}，max_length={}）",
        engine.dim(),
        engine.max_length()
    );
    println!("rounds={rounds}，warmup=5（计时为端到端：分词+推理+池化+归一化）");

    // warmup 5 次（循环覆盖三档长度）
    let tiers = [TEXT_SHORT, TEXT_MEDIUM, TEXT_LONG];
    for i in 0..5 {
        let _ = engine.embed(tiers[i % 3]).unwrap();
    }

    // ---------- 三档单条计时 ----------
    for (name, text) in [("短句", TEXT_SHORT), ("中句", TEXT_MEDIUM), ("长句", TEXT_LONG)] {
        let mut samples = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let t = Instant::now();
            let v = engine.embed(text).unwrap();
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            black_box(&v);
        }
        let (p50, p95, max) = stats(&mut samples);
        println!(
            "{name}（{} tok）: p50={p50:.2}ms p95={p95:.2}ms max={max:.2}ms",
            token_len(text)
        );
    }

    // ---------- 批量对照：embed_batch 一次 vs 逐条 embed() 循环 ----------
    let batch_texts: Vec<&str> = (0..64).map(|i| tiers[i % 3]).collect();
    // 先跑一次不计时（预热批形状 [64, L] 的分配），再计时一次
    let _ = black_box(engine.embed_batch(&batch_texts).unwrap());
    let t = Instant::now();
    let vecs = engine.embed_batch(&batch_texts).unwrap();
    let batch_ms = t.elapsed().as_secs_f64() * 1000.0;
    black_box(&vecs);
    println!(
        "批量 embed_batch × 64（混合长度，批内最长 {} tok）: {batch_ms:.2}ms（每条 {:.2}ms）",
        token_len(TEXT_LONG),
        batch_ms / 64.0
    );

    let t = Instant::now();
    for s in &batch_texts {
        black_box(engine.embed(s).unwrap());
    }
    let loop_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("对照 64 × embed() 循环: {loop_ms:.2}ms（每条 {:.2}ms）", loop_ms / 64.0);
}

/// p50 / p95 / max（nearest-rank 百分位）。
fn stats(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let rank = |p: f64| -> f64 {
        let idx = ((p / 100.0) * n as f64).ceil() as usize;
        samples[idx.clamp(1, n) - 1]
    };
    (rank(50.0), rank(95.0), samples[n - 1])
}
