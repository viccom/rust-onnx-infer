//! ONNX Runtime 会话运行参数。
//!
//! 默认值与 原版历史行为一致（intra/inter 线程 = 4，CUDA deviceId = 0）。

use std::sync::atomic::{AtomicUsize, Ordering};

/// 进程级 GPU 显存上限（MB），由引入方在创建任何 Session 前调
/// [`set_global_gpu_mem_limit`] 设置。
///
/// **为何进程级而非 per-session**：CUDA arena 只增不减——A100 实测单 yolov8n
/// 每 Session ~2.9GB（pool=4 饱和时 11.4GB）；不设上限时曾与 vLLM 共存膨胀至
/// **81GB 打死服务**（cuBLAS OOM 后永久失败）。一个进程只有一个 CUDA device，
/// 统一上限语义合理；各 Session 的 arena 从同一池分配，进程级限制即可防雪崩。
///
/// **ORT 版本绑定**：固定 1.26.0（1.27+ 用 CUDA 13.0 构建，与 GPU 包 CUDA 12.8
/// 基底不匹配；1.26 vs 1.29 CPU 性能无差异——CNN 负载不吃 1.27+ 的 LLM 优化）。
static GLOBAL_GPU_MEM_LIMIT_MB: AtomicUsize = AtomicUsize::new(0);

/// 设置进程级 GPU 显存上限（MB）。0 = 不设上限（仅 CPU / 向后兼容）。
///
/// 必须在创建任何 ONNX Session 之前调用。典型用法：引入方（如 vision-ext）
/// 启动时从环境变量读取后调用一次。
pub fn set_global_gpu_mem_limit(mb: usize) {
    GLOBAL_GPU_MEM_LIMIT_MB.store(mb, Ordering::Relaxed);
}

/// 读取进程级 GPU 显存上限（MB）。
pub fn global_gpu_mem_limit_mb() -> usize {
    GLOBAL_GPU_MEM_LIMIT_MB.load(Ordering::Relaxed)
}

/// 会话运行参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxRuntimeConfig {
    /// Intra-op 线程数
    pub intra_op_threads: usize,
    /// Inter-op 线程数
    pub inter_op_threads: usize,
    /// GPU 设备 id
    pub gpu_device_id: i32,
    /// GPU 显存上限（MB）。0 = 不设上限。
    /// 默认取进程级 override（[`set_global_gpu_mem_limit`]），未设置时为 0。
    pub gpu_mem_limit_mb: usize,
}

pub const DEFAULT_INTRA_OP_THREADS: usize = 4;
pub const DEFAULT_INTER_OP_THREADS: usize = 4;
pub const DEFAULT_GPU_DEVICE_ID: i32 = 0;

impl Default for OnnxRuntimeConfig {
    fn default() -> Self {
        OnnxRuntimeConfig {
            intra_op_threads: DEFAULT_INTRA_OP_THREADS,
            inter_op_threads: DEFAULT_INTER_OP_THREADS,
            gpu_device_id: DEFAULT_GPU_DEVICE_ID,
            gpu_mem_limit_mb: global_gpu_mem_limit_mb(),
        }
    }
}

impl OnnxRuntimeConfig {
    /// 与库历史默认行为一致的配置。
    pub fn defaults() -> Self {
        OnnxRuntimeConfig::default()
    }

    /// 校验参数合法性（与默认值约定一致）。
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::VisionError;
        if self.intra_op_threads < 1 {
            return Err(VisionError::InvalidArgument(format!(
                "intraOpThreads must be >= 1, got {}",
                self.intra_op_threads
            )));
        }
        if self.inter_op_threads < 1 {
            return Err(VisionError::InvalidArgument(format!(
                "interOpThreads must be >= 1, got {}",
                self.inter_op_threads
            )));
        }
        if self.gpu_device_id < 0 {
            return Err(VisionError::InvalidArgument(format!(
                "gpuDeviceId must be >= 0, got {}",
                self.gpu_device_id
            )));
        }
        Ok(())
    }
}

/// Builder（对应 `OnnxRuntimeConfig.Builder`）。
#[derive(Debug, Clone)]
pub struct OnnxRuntimeConfigBuilder {
    config: OnnxRuntimeConfig,
}

impl Default for OnnxRuntimeConfigBuilder {
    fn default() -> Self {
        OnnxRuntimeConfigBuilder {
            config: OnnxRuntimeConfig::defaults(),
        }
    }
}

impl OnnxRuntimeConfigBuilder {
    pub fn intra_op_threads(mut self, threads: usize) -> Self {
        self.config.intra_op_threads = threads;
        self
    }

    pub fn inter_op_threads(mut self, threads: usize) -> Self {
        self.config.inter_op_threads = threads;
        self
    }

    pub fn gpu_device_id(mut self, id: i32) -> Self {
        self.config.gpu_device_id = id;
        self
    }

    pub fn gpu_mem_limit_mb(mut self, mb: usize) -> Self {
        self.config.gpu_mem_limit_mb = mb;
        self
    }

    pub fn build(self) -> OnnxRuntimeConfig {
        self.config
    }
}

impl OnnxRuntimeConfig {
    pub fn builder() -> OnnxRuntimeConfigBuilder {
        OnnxRuntimeConfigBuilder::default()
    }
}

impl std::fmt::Display for OnnxRuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OnnxRuntimeConfig{{intraOpThreads={}, interOpThreads={}, gpuDeviceId={}, gpuMemLimitMb={}}}",
            self.intra_op_threads, self.inter_op_threads, self.gpu_device_id, self.gpu_mem_limit_mb
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 全局变量是进程级状态，写它的测试必须串行执行以防互相干扰。
    fn global_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn default_config_should_have_zero_gpu_mem_limit() {
        let _guard = global_lock();
        let saved = global_gpu_mem_limit_mb();
        set_global_gpu_mem_limit(0);
        let config = OnnxRuntimeConfig::default();
        assert_eq!(config.gpu_mem_limit_mb, 0);
        set_global_gpu_mem_limit(saved);
    }

    #[test]
    fn global_override_should_propagate_to_defaults() {
        let _guard = global_lock();
        let saved = global_gpu_mem_limit_mb();
        set_global_gpu_mem_limit(4096);
        let config = OnnxRuntimeConfig::default();
        assert_eq!(config.gpu_mem_limit_mb, 4096);
        set_global_gpu_mem_limit(saved);
    }

    #[test]
    fn builder_should_override_gpu_mem_limit() {
        let config = OnnxRuntimeConfig::builder()
            .gpu_mem_limit_mb(8192)
            .build();
        assert_eq!(config.gpu_mem_limit_mb, 8192);
    }

    #[test]
    fn display_should_include_gpu_mem_limit() {
        let config = OnnxRuntimeConfig {
            gpu_mem_limit_mb: 3072,
            ..OnnxRuntimeConfig::default()
        };
        let s = config.to_string();
        assert!(s.contains("gpuMemLimitMb=3072"), "实际: {s}");
    }
}
