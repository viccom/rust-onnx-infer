//! resize（对应 `opencv_imgproc::resize`，插值语义对齐 INTER_LINEAR/NEAREST/AREA/CUBIC）。

use crate::error::{Result, VisionError};
use crate::imaging::image::Image;

/// 插值方式（对应 OpenCV `INTER_*` 常量）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// 最近邻（`INTER_NEAREST`）
    Nearest,
    /// 双线性（`INTER_LINEAR`，默认）
    Linear,
    /// 区域平均（`INTER_AREA`，缩小质量最好）
    Area,
    /// 双三次（`INTER_CUBIC`）
    Cubic,
}

impl Default for Interpolation {
    fn default() -> Self {
        Interpolation::Linear
    }
}

impl Interpolation {
    fn to_fir_alg(self) -> fast_image_resize::ResizeAlg {
        use fast_image_resize::{FilterType, ResizeAlg};
        match self {
            Interpolation::Nearest => ResizeAlg::Nearest,
            // Interpolation 系列是固定核窗口的快速算法，大比率缩小（>2x）时会跳过
            // 源像素产生锯齿（实测 1080→224 时分类置信度从 0.58 崩至 0.38，
            // 等效 nearest）。语义对齐 OpenCV INTER_* 必须用真卷积实现。
            Interpolation::Linear => ResizeAlg::Convolution(FilterType::Bilinear),
            Interpolation::Cubic => ResizeAlg::Convolution(FilterType::CatmullRom),
            Interpolation::Area => ResizeAlg::Convolution(FilterType::Box),
        }
    }

    fn to_fir_pixel_type(channels: usize) -> Option<fast_image_resize::PixelType> {
        use fast_image_resize::PixelType;
        match channels {
            1 => Some(PixelType::U8),
            2 => Some(PixelType::U8x2),
            3 => Some(PixelType::U8x3),
            4 => Some(PixelType::U8x4),
            _ => None,
        }
    }
}

/// 拉伸 resize 到目标尺寸（对应 `resize(src, dst, Size(w, h), INTER_x)`）。
pub fn resize(image: &Image, width: usize, height: usize, interp: Interpolation) -> Result<Image> {
    if width == 0 || height == 0 {
        return Err(VisionError::image("resize target size must be > 0"));
    }
    if image.is_empty() {
        return Err(VisionError::image("cannot resize empty image"));
    }
    let pixel_type = Interpolation::to_fir_pixel_type(image.channels()).ok_or_else(|| {
        VisionError::image(format!("unsupported channel count {}", image.channels()))
    })?;

    let src = fast_image_resize::images::Image::from_vec_u8(
        image.width() as u32,
        image.height() as u32,
        image.data().to_vec(),
        pixel_type,
    )
    .map_err(|e| VisionError::image(format!("resize source: {e}")))?;

    let mut dst = fast_image_resize::images::Image::new(
        width as u32,
        height as u32,
        pixel_type,
    );

    let options = fast_image_resize::ResizeOptions {
        algorithm: interp.to_fir_alg(),
        cropping: fast_image_resize::SrcCropping::None,
        // 无 alpha 通道语义；显式关闭避免对 2/4 通道做预乘处理
        mul_div_alpha: false,
    };

    let mut resizer = fast_image_resize::Resizer::new();
    resizer
        .resize(&src, &mut dst, &options)
        .map_err(|e| VisionError::image(format!("resize failed: {e}")))?;

    Image::from_raw(width, height, image.channels(), dst.into_vec())
}

/// 等比缩放到能放入 `width x height` 的最大尺寸（不放大，仅缩小；用于 SAHI 等）。
pub fn resize_within(image: &Image, width: usize, height: usize, interp: Interpolation) -> Result<Image> {
    let scale = (width as f64 / image.width() as f64)
        .min(height as f64 / image.height() as f64)
        .min(1.0);
    let nw = ((image.width() as f64 * scale).round() as usize).max(1);
    let nh = ((image.height() as f64 * scale).round() as usize).max(1);
    resize(image, nw, nh, interp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_stripes(w: usize, h: usize, vertical: bool, low: u8, high: u8) -> Image {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let striped = if vertical { x % 2 == 0 } else { y % 2 == 0 };
                let v = if striped { high } else { low };
                let i = (y * w + x) * 3;
                data[i] = v;
                data[i + 1] = v;
                data[i + 2] = v;
            }
        }
        Image::from_raw(w, h, 3, data).expect("solid stripes buffer always valid")
    }

    /// 大比率缩小时 Linear 的核必须随缩放比拉伸（能量扩散），不能是固定核。
    ///
    /// 回归防护：`ResizeAlg::Interpolation` 是固定核快速算法，8x 缩小时核外源像素
    /// 被直接跳过（实测 1080→224 分类置信度从 0.58 崩至 0.38，等效 nearest）；
    /// 真卷积（Convolution）核随缩放比拉伸，脉冲能量应扩散到所属输出块。
    #[test]
    fn linear_downscale_should_spread_impulse_energy() {
        // 64x1 图：背景 0，x=33 处 255 脉冲；缩到 8x1 后脉冲属于输出块 4（[32..40)）
        let mut data = vec![0u8; 64 * 3];
        for c in 0..3 {
            data[33 * 3 + c] = 255;
        }
        let src = Image::from_raw(64, 1, 3, data).expect("buffer always valid");
        let dst = resize(&src, 8, 1, Interpolation::Linear).unwrap();
        let v = dst.data()[4 * 3] as i32;
        // Convolution(Bilinear) 8x 拉伸核的理论脉冲响应 ≈23/255；
        // 固定核快速算法跳过核外像素时为 0。阈值取 15 区分两者。
        assert!(v >= 15, "大比率缩小时脉冲能量丢失（输出块4={v}），核未随缩放比拉伸");
    }

    #[test]
    fn resize_should_keep_constant_image_constant() {
        let src = solid_stripes(64, 64, false, 128, 128);
        let dst = resize(&src, 8, 8, Interpolation::Linear).unwrap();
        assert!(dst.data().iter().all(|&v| v == 128));
    }

    #[test]
    fn nearest_should_sample_source_pixels_without_blending() {
        let src = solid_stripes(4, 4, false, 0, 255);
        let dst = resize(&src, 2, 2, Interpolation::Nearest).unwrap();
        // 最近邻不混合：输出只能落在源值上
        assert!(dst.data().iter().all(|&v| v == 0 || v == 255));
    }
}
