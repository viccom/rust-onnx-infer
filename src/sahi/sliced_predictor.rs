//! SAHI 切片推理内部编排器。供
//! `DetectionEngine` / `SegmentationEngine` 在开启 SAHI 后由 `predict()` 内部调用——
//! **对用户无感知**：返回类型与关闭 SAHI 时完全一致。
//!
//! **流程**（与官方逐行对齐）：
//! 1. **切片**：[`slicer`](super::slicer) 按官方算法生成切片坐标（重叠像素 int 截断、
//!    末尾切片边界回退吸附）
//! 2. **切片推理**：逐片裁剪原图，调用引擎的**原始单图路径**（引擎内部完成
//!    letterbox（114 灰边、INTER_LINEAR）/255 预处理并把坐标还原到切片坐标系，
//!    与 Python SAHI + ultralytics 的行为一致）
//! 3. **坐标映射**：由 [`SahiResultAdapter`](super::adapters::SahiResultAdapter) 执行
//!    官方 clip 语义（`max(0,·)`、min 到全图尺寸、丢弃无效框）并平移到全图坐标；
//!    分割掩码惰性平移到全图画布
//! 4. **整图标准预测**：切片数 > 1 且开启时，对整图再做一次预测并加入候选
//!    （提升大目标召回，官方 perform_standard_pred 默认开启）
//! 5. **合并后处理**：[`SahiPostprocess`](super::postprocess::SahiPostprocess) 复刻官方
//!    numpy 后端 —— float32 度量矩阵、lexsort 平局排序、`>=`/`>` 双层阈值、并集合并
//!    （分割掩码并集）。默认 GREEDYNMM/IOS/0.5

use std::time::Instant;

use crate::core::engine::OnnxInferenceEngine;
use crate::error::Result;
use crate::imaging::Image;
use crate::model::{Detection, Segmentation};

use super::adapters::{DetectionAdapter, SahiResultAdapter, SegmentationAdapter};
use super::config::{MatchMetric, PostprocessType, SahiBox, SahiConfig};
use super::postprocess::SahiPostprocess;
use super::predictor::SahiSinglePredictor;
use super::result::SahiPredictionResult;
use super::slicer;

/// 检测引擎的切片推理完整流程（对应上游
/// `SahiSlicedPredictor.predictSliced(image, config, SahiAdapters.detection(), engine, single)`）。
///
/// - `engine`：仅用于读取置信度阈值以实现官方低置信度自动切换
/// - `single`：引擎的原始单图推理路径（逐切片调用，绕过 SAHI 防递归）
pub fn predict_sliced_detection(
    engine: &dyn OnnxInferenceEngine<Output = Vec<Detection>>,
    single: &dyn SahiSinglePredictor<Detection>,
    image: &Image,
    config: &SahiConfig,
) -> Result<SahiPredictionResult<Vec<Detection>>> {
    predict_sliced(image, config, &DetectionAdapter, engine.confidence_threshold(), single)
}

/// 分割引擎的切片推理完整流程（对应上游
/// `SahiSlicedPredictor.predictSliced(image, config, SahiAdapters.segmentation(), engine, single)`）。
pub fn predict_sliced_segmentation(
    engine: &dyn OnnxInferenceEngine<Output = Vec<Segmentation>>,
    single: &dyn SahiSinglePredictor<Segmentation>,
    image: &Image,
    config: &SahiConfig,
) -> Result<SahiPredictionResult<Vec<Segmentation>>> {
    predict_sliced(image, config, &SegmentationAdapter, engine.confidence_threshold(), single)
}

/// OBB 引擎的切片推理完整流程（旋转框以 aabb 近似合并，见 `ObbAdapter` 文档）。
pub fn predict_sliced_obb(
    engine: &dyn OnnxInferenceEngine<Output = Vec<crate::engines::obb_detection::ObbResult>>,
    single: &dyn crate::sahi::predictor::SahiSinglePredictor<crate::engines::obb_detection::ObbResult>,
    image: &Image,
    config: &SahiConfig,
) -> Result<SahiPredictionResult<Vec<crate::engines::obb_detection::ObbResult>>> {
    predict_sliced(image, config, &crate::sahi::adapters::ObbAdapter, engine.confidence_threshold(), single)
}

/// 切片推理完整流程核心（对应 `predictSliced` 泛型重载）。
fn predict_sliced<T>(
    image: &Image,
    config: &SahiConfig,
    adapter: &dyn SahiResultAdapter<T>,
    confidence_threshold: f32,
    single: &dyn SahiSinglePredictor<T>,
) -> Result<SahiPredictionResult<Vec<T>>> {
    let image_height = image.height() as i32;
    let image_width = image.width() as i32;

    // 1. 切片
    let t_slice = Instant::now();
    let slices = slicer::get_slice_bboxes(
        image_height,
        image_width,
        config.slice_height,
        config.slice_width,
        config.auto_slice_resolution,
        config.overlap_height_ratio,
        config.overlap_width_ratio,
    )?;
    let slice_millis = t_slice.elapsed().as_millis() as u64;

    if tracing::enabled!(tracing::Level::DEBUG) || config.verbose {
        tracing::info!(
            "SAHI 切片：{} 片（slice={:?}x{:?}, overlap={}/{}, image={}x{}）",
            slices.len(),
            config.slice_height,
            config.slice_width,
            config.overlap_height_ratio,
            config.overlap_width_ratio,
            image_width,
            image_height
        );
    }

    // 官方：低置信度阈值时自动切换 NMS/IOU，避免合并操作放大 bbox。
    // 有效阈值 = max(引擎阈值, min_confidence)：合并前已按 min_confidence 过滤时，
    // 低分碎片不会进入合并，无需为引擎的低构造阈值切换策略。
    let effective_threshold = config
        .min_confidence
        .map_or(confidence_threshold, |m| confidence_threshold.max(m));
    let mut postprocess_type = config.postprocess_type;
    let mut match_metric = config.match_metric;
    if !config.force_postprocess_type
        && effective_threshold < SahiConfig::LOW_MODEL_CONFIDENCE
        && postprocess_type != PostprocessType::Nms
    {
        tracing::warn!(
            "引擎置信度阈值较低（{}），SAHI postprocess 自动切换为 NMS/IOU（官方 LOW_MODEL_CONFIDENCE 行为；可用 force_postprocess_type(true) 禁用）",
            effective_threshold
        );
        postprocess_type = PostprocessType::Nms;
        match_metric = MatchMetric::Iou;
    }
    let postprocess = SahiPostprocess::new(
        postprocess_type,
        match_metric,
        config.match_threshold,
        config.class_agnostic,
        adapter.mask_merger(),
    );

    // 2. 切片推理 + 坐标映射
    let t_predict = Instant::now();
    let mut accumulated: Vec<SahiBox> = Vec::new();
    for slice in &slices {
        let [x_min, y_min, x_max, y_max] = *slice;
        let slice_results = predict_slice(single, image, x_min, y_min, x_max, y_max)?;
        collect_predictions(
            &slice_results,
            adapter,
            config,
            &mut accumulated,
            x_min,
            y_min,
            image_width,
            image_height,
        );
        if let Some(buffer_length) = config.merge_buffer_length {
            if accumulated.len() > buffer_length as usize {
                accumulated = postprocess.process(&accumulated);
            }
        }
    }

    // 官方：切片数 > 1 且 perform_standard_pred 时，整图标准预测参与合并（提升大目标召回）
    if slices.len() > 1 && config.perform_standard_prediction {
        let full_results = single.predict_single(image)?;
        collect_predictions(
            &full_results,
            adapter,
            config,
            &mut accumulated,
            0,
            0,
            image_width,
            image_height,
        );
    }
    let predict_millis = t_predict.elapsed().as_millis() as u64;

    // 3. 合并后处理（官方：预测数 > 1 才执行合并）
    let t_post = Instant::now();
    let merged = if accumulated.len() > 1 {
        postprocess.process(&accumulated)
    } else {
        accumulated
    };
    let postprocess_millis = t_post.elapsed().as_millis() as u64;

    // 4. 重建结果对象（分割场景在此物化全图掩码）
    let results: Vec<T> = merged.iter().map(|b| adapter.to_result(b)).collect();
    if config.verbose {
        tracing::info!("SAHI 完成：{} 切片 → {} 最终结果", slices.len(), results.len());
    }
    Ok(SahiPredictionResult::new(
        results,
        slices.len(),
        image_width,
        image_height,
        slice_millis,
        predict_millis,
        postprocess_millis,
    ))
}

/// 裁剪切片并调用引擎原始单图路径（对应 `predictSlice` 的 ROI 视图；
/// Rust `Image::crop` 为拷贝产出）。
fn predict_slice<T>(
    single: &dyn SahiSinglePredictor<T>,
    image: &Image,
    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,
) -> Result<Vec<T>> {
    let slice_image = image.crop(
        x_min.max(0) as usize,
        y_min.max(0) as usize,
        (x_max - x_min).max(0) as usize,
        (y_max - y_min).max(0) as usize,
    )?;
    single.predict_single(&slice_image)
}

/// 官方顺序：先过滤排除类别，再 clip，再平移到全图坐标（对应 `collectPredictions`）。
fn collect_predictions<T>(
    raw: &[T],
    adapter: &dyn SahiResultAdapter<T>,
    config: &SahiConfig,
    out: &mut Vec<SahiBox>,
    shift_x: i32,
    shift_y: i32,
    full_width: i32,
    full_height: i32,
) {
    for item in raw {
        if adapter.is_excluded(item, &config.exclude_class_names, &config.exclude_class_ids) {
            continue;
        }
        if let Some(boxed) = adapter.to_box(item, shift_x, shift_y, full_width, full_height) {
            if config.min_confidence.is_some_and(|m| boxed.score < m) {
                continue;
            }
            out.push(boxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 固定输出:高分小框 + 同位置低分大框(IOS=1,GREEDYNMM 必合并成 union)。
    struct TwoBoxPredictor {
        low_score: f64,
    }

    impl SahiSinglePredictor<Detection> for TwoBoxPredictor {
        fn predict_single(&self, _image: &Image) -> Result<Vec<Detection>> {
            Ok(vec![
                Detection::new("obj", 0, 10.0, 10.0, 50.0, 50.0, 0.8),
                Detection::new("obj", 0, 10.0, 10.0, 90.0, 90.0, self.low_score),
            ])
        }
    }

    /// 200x200 单片配置(切片 = 整图,关闭整图标准预测避免重复)。
    fn single_slice_config() -> SahiConfig {
        let mut cfg = SahiConfig::default();
        cfg.slice_width = Some(200);
        cfg.slice_height = Some(200);
        cfg.auto_slice_resolution = false;
        cfg.perform_standard_prediction = false;
        cfg
    }

    #[test]
    fn min_confidence_should_drop_low_score_boxes_before_merge() {
        let image = Image::new(200, 200, 3);
        let predictor = TwoBoxPredictor { low_score: 0.15 };

        // 无 min_confidence:低分大框参与合并,union 把高分框放大到 90
        let cfg = single_slice_config();
        let r = predict_sliced(&image, &cfg, &DetectionAdapter, 0.1, &predictor).unwrap();
        assert_eq!(r.detections.len(), 1);
        assert!(
            (r.detections[0].x2() - 90.0).abs() < 1e-3,
            "无过滤时 union 应放大到 90,实际 {}",
            r.detections[0].x2()
        );

        // min_confidence=0.25:低分框在合并前丢弃,高分框几何原样保留——
        // 与引擎直接以 0.25 构造的结果严格等价(类内 NMS 保序)
        let mut cfg = single_slice_config();
        cfg.min_confidence = Some(0.25);
        let r = predict_sliced(&image, &cfg, &DetectionAdapter, 0.1, &predictor).unwrap();
        assert_eq!(r.detections.len(), 1);
        assert!(
            (r.detections[0].x2() - 50.0).abs() < 1e-3,
            "过滤后高分框几何应原样保留(50),实际 {}",
            r.detections[0].x2()
        );
        assert!((r.detections[0].confidence() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn low_model_confidence_switch_should_use_effective_threshold() {
        // 两框都 >= 0.3:IoU=0.25(不抑制) vs IOS=1.0(必合并),可区分 NMS/IOU 与 GREEDYNMM/IOS
        let image = Image::new(200, 200, 3);
        let predictor = TwoBoxPredictor { low_score: 0.5 };

        // 引擎阈值 0.05 < LOW_MODEL_CONFIDENCE 且无 min_confidence → 官方行为切 NMS/IOU,两框都保留
        let cfg = single_slice_config();
        let r = predict_sliced(&image, &cfg, &DetectionAdapter, 0.05, &predictor).unwrap();
        assert_eq!(r.detections.len(), 2, "低阈值无过滤应切换 NMS/IOU,IoU 0.25 两框都留");

        // min_confidence=0.3 使有效阈值 0.3 >= LOW_MODEL_CONFIDENCE → 保持 GREEDYNMM/IOS,合并为一
        let mut cfg = single_slice_config();
        cfg.min_confidence = Some(0.3);
        let r = predict_sliced(&image, &cfg, &DetectionAdapter, 0.05, &predictor).unwrap();
        assert_eq!(r.detections.len(), 1, "有效阈值 0.3 应保持 GREEDYNMM,IOS 1.0 合并为一");
    }
}
