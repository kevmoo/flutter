#![forbid(unsafe_code)]

use bytemuck::Pod;
use impeller_geometry::{BlendMode, Color, ISize, Rect, Scalar};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Number of frame slots in the `HostBuffer` ring arena (`kHostBufferArenaSize` in `host_buffer.h`).
pub const K_HOST_BUFFER_ARENA_SIZE: usize = 4;

/// Default block size in bytes for `HostBuffer` arena allocations (`kAllocatorBlockSize` in `host_buffer.cc`).
pub const K_DEFAULT_BLOCK_CAPACITY: usize = 1_024_000;

/// Maximum number of vertex buffer bindings (`kMaxVertexBuffers` in `vertex_buffer.h`).
pub const K_MAX_VERTEX_BUFFERS: usize = 16;

const ONE_OFF_BUFFER_TAG: usize = 1usize << (usize::BITS - 2);

static NEXT_DEVICE_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum WindingOrder {
    #[default]
    Clockwise,
    CounterClockwise,
}

/// Specifies where an allocation resides and how it is used (`StorageMode` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum StorageMode {
    HostVisible,
    DevicePrivate,
    #[default]
    DeviceTransient,
}

impl StorageMode {
    pub const fn as_str(&self) -> &'static str {
        match self {
            StorageMode::HostVisible => "HostVisible",
            StorageMode::DevicePrivate => "DevicePrivate",
            StorageMode::DeviceTransient => "DeviceTransient",
        }
    }
}

/// Family of a block-compressed pixel format (`CompressedTextureFamily` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompressedTextureFamily {
    BC,
    ETC2,
    ASTC,
    ASTCHDR,
}

/// Pixel formats supported by Impeller (`PixelFormat` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelFormat {
    Unknown,
    A8UNormInt,
    R8UNormInt,
    R8G8UNormInt,
    #[default]
    R8G8B8A8UNormInt,
    R8G8B8A8UNormIntSRGB,
    B8G8R8A8UNormInt,
    B8G8R8A8UNormIntSRGB,
    R32G32B32A32Float,
    R16G16B16A16Float,
    B10G10R10XR,
    B10G10R10XRSRGB,
    B10G10R10A10XR,
    R32Float,
    // Depth and stencil formats.
    S8UInt,
    D24UnormS8Uint,
    D32FloatS8UInt,
    // Block-compressed formats.
    BC1RGBAUNormInt,
    BC1RGBAUNormIntSRGB,
    BC3RGBAUNormInt,
    BC3RGBAUNormIntSRGB,
    BC5RGUNormInt,
    BC7RGBAUNormInt,
    BC7RGBAUNormIntSRGB,
    ETC2RGB8UNormInt,
    ETC2RGB8UNormIntSRGB,
    ETC2RGBA8UNormInt,
    ETC2RGBA8UNormIntSRGB,
    ASTC4x4LDR,
    ASTC4x4LDRSRGB,
    ASTC8x8LDR,
    ASTC8x8LDRSRGB,
    ASTC4x4HDR,
    ASTC8x8HDR,
}

impl PixelFormat {
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::Unknown => 0,
            PixelFormat::A8UNormInt | PixelFormat::R8UNormInt | PixelFormat::S8UInt => 1,
            PixelFormat::R8G8UNormInt => 2,
            PixelFormat::R8G8B8A8UNormInt
            | PixelFormat::R8G8B8A8UNormIntSRGB
            | PixelFormat::B8G8R8A8UNormInt
            | PixelFormat::B8G8R8A8UNormIntSRGB
            | PixelFormat::B10G10R10XR
            | PixelFormat::B10G10R10XRSRGB
            | PixelFormat::R32Float
            | PixelFormat::D24UnormS8Uint => 4,
            PixelFormat::D32FloatS8UInt => 5,
            PixelFormat::R16G16B16A16Float | PixelFormat::B10G10R10A10XR => 8,
            PixelFormat::R32G32B32A32Float => 16,
            PixelFormat::BC1RGBAUNormInt
            | PixelFormat::BC1RGBAUNormIntSRGB
            | PixelFormat::BC3RGBAUNormInt
            | PixelFormat::BC3RGBAUNormIntSRGB
            | PixelFormat::BC5RGUNormInt
            | PixelFormat::BC7RGBAUNormInt
            | PixelFormat::BC7RGBAUNormIntSRGB
            | PixelFormat::ETC2RGB8UNormInt
            | PixelFormat::ETC2RGB8UNormIntSRGB
            | PixelFormat::ETC2RGBA8UNormInt
            | PixelFormat::ETC2RGBA8UNormIntSRGB
            | PixelFormat::ASTC4x4LDR
            | PixelFormat::ASTC4x4LDRSRGB
            | PixelFormat::ASTC8x8LDR
            | PixelFormat::ASTC8x8LDRSRGB
            | PixelFormat::ASTC4x4HDR
            | PixelFormat::ASTC8x8HDR => 0,
        }
    }

    pub const fn is_depth_writable(&self) -> bool {
        matches!(
            self,
            PixelFormat::D24UnormS8Uint | PixelFormat::D32FloatS8UInt
        )
    }

    pub const fn is_stencil_writable(&self) -> bool {
        matches!(
            self,
            PixelFormat::S8UInt | PixelFormat::D24UnormS8Uint | PixelFormat::D32FloatS8UInt
        )
    }

    pub const fn is_depth_stencil(&self) -> bool {
        self.is_depth_writable() || self.is_stencil_writable()
    }

    pub const fn is_color(&self) -> bool {
        !matches!(self, PixelFormat::Unknown) && !self.is_depth_stencil()
    }

    pub const fn is_compressed(&self) -> bool {
        self.compressed_texture_family().is_some()
    }

    pub const fn compressed_texture_family(&self) -> Option<CompressedTextureFamily> {
        match self {
            PixelFormat::BC1RGBAUNormInt
            | PixelFormat::BC1RGBAUNormIntSRGB
            | PixelFormat::BC3RGBAUNormInt
            | PixelFormat::BC3RGBAUNormIntSRGB
            | PixelFormat::BC5RGUNormInt
            | PixelFormat::BC7RGBAUNormInt
            | PixelFormat::BC7RGBAUNormIntSRGB => Some(CompressedTextureFamily::BC),
            PixelFormat::ETC2RGB8UNormInt
            | PixelFormat::ETC2RGB8UNormIntSRGB
            | PixelFormat::ETC2RGBA8UNormInt
            | PixelFormat::ETC2RGBA8UNormIntSRGB => Some(CompressedTextureFamily::ETC2),
            PixelFormat::ASTC4x4LDR
            | PixelFormat::ASTC4x4LDRSRGB
            | PixelFormat::ASTC8x8LDR
            | PixelFormat::ASTC8x8LDRSRGB => Some(CompressedTextureFamily::ASTC),
            PixelFormat::ASTC4x4HDR | PixelFormat::ASTC8x8HDR => {
                Some(CompressedTextureFamily::ASTCHDR)
            }
            _ => None,
        }
    }

    pub const fn compressed_block_width(&self) -> usize {
        match self {
            PixelFormat::ASTC8x8LDR | PixelFormat::ASTC8x8LDRSRGB | PixelFormat::ASTC8x8HDR => 8,
            PixelFormat::BC1RGBAUNormInt
            | PixelFormat::BC1RGBAUNormIntSRGB
            | PixelFormat::BC3RGBAUNormInt
            | PixelFormat::BC3RGBAUNormIntSRGB
            | PixelFormat::BC5RGUNormInt
            | PixelFormat::BC7RGBAUNormInt
            | PixelFormat::BC7RGBAUNormIntSRGB
            | PixelFormat::ETC2RGB8UNormInt
            | PixelFormat::ETC2RGB8UNormIntSRGB
            | PixelFormat::ETC2RGBA8UNormInt
            | PixelFormat::ETC2RGBA8UNormIntSRGB
            | PixelFormat::ASTC4x4LDR
            | PixelFormat::ASTC4x4LDRSRGB
            | PixelFormat::ASTC4x4HDR => 4,
            _ => 1,
        }
    }

    pub const fn compressed_block_height(&self) -> usize {
        self.compressed_block_width()
    }

    pub const fn bytes_per_block(&self) -> usize {
        match self {
            PixelFormat::BC1RGBAUNormInt
            | PixelFormat::BC1RGBAUNormIntSRGB
            | PixelFormat::ETC2RGB8UNormInt
            | PixelFormat::ETC2RGB8UNormIntSRGB => 8,
            PixelFormat::BC3RGBAUNormInt
            | PixelFormat::BC3RGBAUNormIntSRGB
            | PixelFormat::BC5RGUNormInt
            | PixelFormat::BC7RGBAUNormInt
            | PixelFormat::BC7RGBAUNormIntSRGB
            | PixelFormat::ETC2RGBA8UNormInt
            | PixelFormat::ETC2RGBA8UNormIntSRGB
            | PixelFormat::ASTC4x4LDR
            | PixelFormat::ASTC4x4LDRSRGB
            | PixelFormat::ASTC8x8LDR
            | PixelFormat::ASTC8x8LDRSRGB
            | PixelFormat::ASTC4x4HDR
            | PixelFormat::ASTC8x8HDR => 16,
            _ => self.bytes_per_pixel(),
        }
    }

    pub const fn bytes_for_texture_region(&self, width: i64, height: i64) -> usize {
        let block_width = self.compressed_block_width();
        let block_height = self.compressed_block_height();
        let w = if width <= 0 { 0 } else { width as usize };
        let h = if height <= 0 { 0 } else { height as usize };
        let blocks_wide = w.div_ceil(block_width);
        let blocks_high = h.div_ceil(block_height);
        blocks_wide * blocks_high * self.bytes_per_block()
    }

    pub const fn bytes_per_row_for_width(&self, width: i64) -> usize {
        let block_width = self.compressed_block_width();
        let w = if width <= 0 { 0 } else { width as usize };
        let blocks_wide = w.div_ceil(block_width);
        blocks_wide * self.bytes_per_block()
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            PixelFormat::Unknown => "Unknown",
            PixelFormat::A8UNormInt => "A8UNormInt",
            PixelFormat::R8UNormInt => "R8UNormInt",
            PixelFormat::R8G8UNormInt => "R8G8UNormInt",
            PixelFormat::R8G8B8A8UNormInt => "R8G8B8A8UNormInt",
            PixelFormat::R8G8B8A8UNormIntSRGB => "R8G8B8A8UNormIntSRGB",
            PixelFormat::B8G8R8A8UNormInt => "B8G8R8A8UNormInt",
            PixelFormat::B8G8R8A8UNormIntSRGB => "B8G8R8A8UNormIntSRGB",
            PixelFormat::R32G32B32A32Float => "R32G32B32A32Float",
            PixelFormat::R16G16B16A16Float => "R16G16B16A16Float",
            PixelFormat::B10G10R10XR => "B10G10R10XR",
            PixelFormat::B10G10R10XRSRGB => "B10G10R10XRSRGB",
            PixelFormat::B10G10R10A10XR => "B10G10R10A10XR",
            PixelFormat::R32Float => "R32Float",
            PixelFormat::S8UInt => "S8UInt",
            PixelFormat::D24UnormS8Uint => "D24UnormS8Uint",
            PixelFormat::D32FloatS8UInt => "D32FloatS8UInt",
            PixelFormat::BC1RGBAUNormInt => "BC1RGBAUNormInt",
            PixelFormat::BC1RGBAUNormIntSRGB => "BC1RGBAUNormIntSRGB",
            PixelFormat::BC3RGBAUNormInt => "BC3RGBAUNormInt",
            PixelFormat::BC3RGBAUNormIntSRGB => "BC3RGBAUNormIntSRGB",
            PixelFormat::BC5RGUNormInt => "BC5RGUNormInt",
            PixelFormat::BC7RGBAUNormInt => "BC7RGBAUNormInt",
            PixelFormat::BC7RGBAUNormIntSRGB => "BC7RGBAUNormIntSRGB",
            PixelFormat::ETC2RGB8UNormInt => "ETC2RGB8UNormInt",
            PixelFormat::ETC2RGB8UNormIntSRGB => "ETC2RGB8UNormIntSRGB",
            PixelFormat::ETC2RGBA8UNormInt => "ETC2RGBA8UNormInt",
            PixelFormat::ETC2RGBA8UNormIntSRGB => "ETC2RGBA8UNormIntSRGB",
            PixelFormat::ASTC4x4LDR => "ASTC4x4LDR",
            PixelFormat::ASTC4x4LDRSRGB => "ASTC4x4LDRSRGB",
            PixelFormat::ASTC8x8LDR => "ASTC8x8LDR",
            PixelFormat::ASTC8x8LDRSRGB => "ASTC8x8LDRSRGB",
            PixelFormat::ASTC4x4HDR => "ASTC4x4HDR",
            PixelFormat::ASTC8x8HDR => "ASTC8x8HDR",
        }
    }
}

pub const fn is_compressed(format: PixelFormat) -> bool {
    format.is_compressed()
}

pub const fn compressed_texture_family_for_format(
    format: PixelFormat,
) -> Option<CompressedTextureFamily> {
    format.compressed_texture_family()
}

pub const fn compressed_block_width_for_pixel_format(format: PixelFormat) -> usize {
    format.compressed_block_width()
}

pub const fn compressed_block_height_for_pixel_format(format: PixelFormat) -> usize {
    format.compressed_block_height()
}

pub const fn bytes_per_pixel_for_pixel_format(format: PixelFormat) -> usize {
    format.bytes_per_pixel()
}

pub const fn bytes_per_block_for_pixel_format(format: PixelFormat) -> usize {
    format.bytes_per_block()
}

pub const fn bytes_for_texture_region(format: PixelFormat, width: i64, height: i64) -> usize {
    format.bytes_for_texture_region(width, height)
}

pub const fn bytes_per_row_for_texture_width(format: PixelFormat, width: i64) -> usize {
    format.bytes_per_row_for_width(width)
}

/// Sample count for multisampling (`SampleCount` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum SampleCount {
    #[default]
    Count1 = 1,
    Count4 = 4,
}

impl SampleCount {
    pub const fn count(&self) -> u32 {
        *self as u32
    }
}

/// Render pass attachment load action (`LoadAction` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum LoadAction {
    #[default]
    DontCare,
    Load,
    Clear,
}

impl LoadAction {
    pub const fn can_clear_attachment(&self) -> bool {
        match self {
            LoadAction::Load => false,
            LoadAction::DontCare | LoadAction::Clear => true,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            LoadAction::DontCare => "DontCare",
            LoadAction::Load => "Load",
            LoadAction::Clear => "Clear",
        }
    }
}

/// Render pass attachment store action (`StoreAction` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum StoreAction {
    DontCare,
    #[default]
    Store,
    MultisampleResolve,
    StoreAndMultisampleResolve,
}

impl StoreAction {
    pub const fn can_discard_attachment_when_done(&self) -> bool {
        match self {
            StoreAction::Store | StoreAction::StoreAndMultisampleResolve => false,
            StoreAction::DontCare | StoreAction::MultisampleResolve => true,
        }
    }

    pub const fn needs_resolve_texture(&self) -> bool {
        match self {
            StoreAction::DontCare | StoreAction::Store => false,
            StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve => true,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            StoreAction::DontCare => "DontCare",
            StoreAction::Store => "Store",
            StoreAction::MultisampleResolve => "MultisampleResolve",
            StoreAction::StoreAndMultisampleResolve => "StoreAndMultisampleResolve",
        }
    }
}

/// Index buffer element type (`IndexType` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum IndexType {
    Unknown,
    Index16,
    Index32,
    #[default]
    None,
}

impl IndexType {
    pub const fn byte_size(&self) -> usize {
        match self {
            IndexType::Index16 => 2,
            IndexType::Index32 => 4,
            IndexType::Unknown | IndexType::None => 0,
        }
    }

    pub const fn bytes_per_index(&self) -> usize {
        self.byte_size()
    }
}

/// Primitive topology (`PrimitiveType` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum PrimitiveType {
    #[default]
    Triangle,
    TriangleStrip,
    Line,
    LineStrip,
    Point,
    TriangleFan,
}

/// Polygon rasterization mode (`PolygonMode` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
}

/// Face culling mode (`CullMode` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum CullMode {
    #[default]
    None,
    FrontFace,
    BackFace,
}

/// Depth/stencil comparison function (`CompareFunction` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum CompareFunction {
    Never,
    #[default]
    Always,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
}

/// Stencil operation (`StencilOperation` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum StencilOperation {
    #[default]
    Keep,
    Zero,
    SetToReferenceValue,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DepthRange {
    pub z_near: Scalar,
    pub z_far: Scalar,
}

impl Default for DepthRange {
    fn default() -> Self {
        Self {
            z_near: 0.0,
            z_far: 1.0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Viewport {
    pub rect: Rect,
    pub depth_range: DepthRange,
}

/// Color write mask (`ColorWriteMaskBits` in `formats.h`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask(pub u64);

impl ColorWriteMask {
    pub const NONE: Self = Self(0);
    pub const RED: Self = Self(1 << 0);
    pub const GREEN: Self = Self(1 << 1);
    pub const BLUE: Self = Self(1 << 2);
    pub const ALPHA: Self = Self(1 << 3);
    pub const ALL: Self = Self(Self::RED.0 | Self::GREEN.0 | Self::BLUE.0 | Self::ALPHA.0);

    pub const fn bits(&self) -> u64 {
        self.0
    }

    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl Default for ColorWriteMask {
    fn default() -> Self {
        Self::ALL
    }
}

impl BitOr for ColorWriteMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for ColorWriteMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for ColorWriteMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for ColorWriteMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl Not for ColorWriteMask {
    type Output = Self;
    fn not(self) -> Self {
        Self(!self.0 & Self::ALL.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Range {
    pub offset: usize,
    pub length: usize,
}

impl Range {
    pub const fn new(offset: usize, length: usize) -> Self {
        Self { offset, length }
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub const fn end(&self) -> usize {
        self.offset + self.length
    }

    /// Creates a new `Range` that is the union of `self` and `other` (`Range::Merge` in `range.h`).
    pub const fn merge(self, other: Self) -> Self {
        if other.length == 0 {
            return self;
        }
        if self.length == 0 {
            return other;
        }
        let start_offset = if self.offset < other.offset {
            self.offset
        } else {
            other.offset
        };
        let self_end = self.offset + self.length;
        let other_end = other.offset + other.length;
        let end_offset = if self_end > other_end {
            self_end
        } else {
            other_end
        };
        Self {
            offset: start_offset,
            length: end_offset - start_offset,
        }
    }
}

/// Descriptor for a `DeviceBuffer` allocation (`DeviceBufferDescriptor` in `device_buffer_descriptor.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct DeviceBufferDescriptor {
    pub storage_mode: StorageMode,
    pub size: usize,
    pub readback: bool,
}

impl DeviceBufferDescriptor {
    pub const fn new(storage_mode: StorageMode, size: usize) -> Self {
        Self {
            storage_mode,
            size,
            readback: false,
        }
    }
}

/// Backing device buffer allocation (`DeviceBuffer` in `device_buffer.h`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceBuffer {
    id: u64,
    descriptor: DeviceBufferDescriptor,
    label: String,
    bytes: Vec<u8>,
}

impl DeviceBuffer {
    pub fn new(descriptor: DeviceBufferDescriptor) -> Self {
        Self {
            id: NEXT_DEVICE_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            descriptor,
            label: String::new(),
            bytes: vec![0u8; descriptor.size],
        }
    }

    pub(crate) fn new_arena_block(capacity: usize) -> Self {
        Self {
            id: NEXT_DEVICE_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            descriptor: DeviceBufferDescriptor {
                storage_mode: StorageMode::HostVisible,
                size: capacity,
                readback: false,
            },
            label: String::new(),
            bytes: Vec::with_capacity(capacity),
        }
    }

    pub const fn id(&self) -> u64 {
        self.id
    }

    pub const fn get_device_buffer_descriptor(&self) -> &DeviceBufferDescriptor {
        &self.descriptor
    }

    pub fn set_label(&mut self, label: impl Into<String>) -> bool {
        self.label = label.into();
        true
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn on_get_contents(&self) -> &[u8] {
        &self.bytes
    }

    pub fn on_get_contents_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    pub fn copy_host_buffer(&mut self, source: &[u8], source_range: Range, offset: usize) -> bool {
        if source_range.length == 0 {
            return true;
        }
        if self.descriptor.storage_mode != StorageMode::HostVisible {
            return false;
        }
        let Some(dst_end) = offset.checked_add(source_range.length) else {
            return false;
        };
        if dst_end > self.descriptor.size {
            return false;
        }
        let Some(src_end) = source_range.offset.checked_add(source_range.length) else {
            return false;
        };
        if src_end > source.len() {
            return false;
        }
        if self.bytes.len() < dst_end {
            self.bytes.resize(dst_end, 0);
        }
        self.bytes[offset..dst_end].copy_from_slice(&source[source_range.offset..src_end]);
        true
    }
}

/// View into a range of a buffer (`BufferView` in `buffer_view.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct BufferView {
    pub buffer_index: usize,
    pub range: Range,
}

impl BufferView {
    pub const fn new(buffer_index: usize, range: Range) -> Self {
        Self {
            buffer_index,
            range,
        }
    }

    pub const fn empty() -> Self {
        Self {
            buffer_index: 0,
            range: Range::new(0, 0),
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.range.length == 0
    }

    pub const fn is_valid(&self) -> bool {
        self.range.length > 0
    }

    pub const fn get_range(&self) -> Range {
        self.range
    }

    /// Creates a sub-view within `self` if `offset + length <= self.range.length`.
    pub fn take_sub_view(&self, offset: usize, length: usize) -> Option<BufferView> {
        let end = offset.checked_add(length)?;
        if end > self.range.length {
            return None;
        }
        Some(BufferView {
            buffer_index: self.buffer_index,
            range: Range::new(self.range.offset + offset, length),
        })
    }

    /// Accesses the underlying bytes in `host_buffer`.
    pub fn slice<'a>(&self, host_buffer: &'a HostBuffer) -> &'a [u8] {
        host_buffer.get_slice(*self)
    }
}

/// Vertex and index buffer binding (`VertexBuffer` in `vertex_buffer.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct VertexBuffer {
    pub vertex_buffer: BufferView,
    pub index_buffer: BufferView,
    pub vertex_count: usize,
    pub index_type: IndexType,
}

impl VertexBuffer {
    pub const fn new(
        vertex_buffer: BufferView,
        index_buffer: BufferView,
        vertex_count: usize,
        index_type: IndexType,
    ) -> Self {
        Self {
            vertex_buffer,
            index_buffer,
            vertex_count,
            index_type,
        }
    }

    pub const fn is_valid(&self) -> bool {
        self.vertex_buffer.is_valid()
            && !matches!(self.index_type, IndexType::Unknown)
            && (matches!(self.index_type, IndexType::None) || self.index_buffer.is_valid())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum BlendFactor {
    #[default]
    Zero,
    One,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    SourceColor,
    OneMinusSourceColor,
    DestinationColor,
    OneMinusDestinationColor,
    SourceAlphaSaturated,
    BlendColor,
    OneMinusBlendColor,
    BlendAlpha,
    OneMinusBlendAlpha,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum BlendOperation {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorAttachmentDescriptor {
    pub format: PixelFormat,
    pub blending_enabled: bool,
    pub src_color_blend_factor: BlendFactor,
    pub dst_color_blend_factor: BlendFactor,
    pub color_blend_op: BlendOperation,
    pub src_alpha_blend_factor: BlendFactor,
    pub dst_alpha_blend_factor: BlendFactor,
    pub alpha_blend_op: BlendOperation,
    pub write_mask: ColorWriteMask,
}

impl Default for ColorAttachmentDescriptor {
    fn default() -> Self {
        Self {
            format: PixelFormat::default(),
            blending_enabled: false,
            src_color_blend_factor: BlendFactor::SourceAlpha,
            dst_color_blend_factor: BlendFactor::OneMinusSourceAlpha,
            color_blend_op: BlendOperation::Add,
            src_alpha_blend_factor: BlendFactor::SourceAlpha,
            dst_alpha_blend_factor: BlendFactor::OneMinusSourceAlpha,
            alpha_blend_op: BlendOperation::Add,
            write_mask: ColorWriteMask::ALL,
        }
    }
}

impl ColorAttachmentDescriptor {
    pub fn for_blend_mode(format: PixelFormat, mode: BlendMode) -> Self {
        let mut desc = Self {
            format,
            blending_enabled: true,
            color_blend_op: BlendOperation::Add,
            alpha_blend_op: BlendOperation::Add,
            write_mask: ColorWriteMask::ALL,
            ..Default::default()
        };
        let (sc, dc, sa, da) = match mode {
            BlendMode::Clear => (
                BlendFactor::Zero,
                BlendFactor::Zero,
                BlendFactor::Zero,
                BlendFactor::Zero,
            ),
            BlendMode::Src => (
                BlendFactor::One,
                BlendFactor::Zero,
                BlendFactor::One,
                BlendFactor::Zero,
            ),
            BlendMode::Dst => (
                BlendFactor::Zero,
                BlendFactor::One,
                BlendFactor::Zero,
                BlendFactor::One,
            ),
            BlendMode::SrcOver => (
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
            ),
            BlendMode::DstOver => (
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::One,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::One,
            ),
            BlendMode::SrcIn => (
                BlendFactor::DestinationAlpha,
                BlendFactor::Zero,
                BlendFactor::DestinationAlpha,
                BlendFactor::Zero,
            ),
            BlendMode::DstIn => (
                BlendFactor::Zero,
                BlendFactor::SourceAlpha,
                BlendFactor::Zero,
                BlendFactor::SourceAlpha,
            ),
            BlendMode::SrcOut => (
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::Zero,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::Zero,
            ),
            BlendMode::DstOut => (
                BlendFactor::Zero,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::Zero,
                BlendFactor::OneMinusSourceAlpha,
            ),
            BlendMode::SrcATop => (
                BlendFactor::DestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::DestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
            ),
            BlendMode::DstATop => (
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::SourceAlpha,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::SourceAlpha,
            ),
            BlendMode::Xor => (
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
            ),
            BlendMode::Plus => (
                BlendFactor::One,
                BlendFactor::One,
                BlendFactor::One,
                BlendFactor::One,
            ),
            BlendMode::Modulate => (
                BlendFactor::DestinationColor,
                BlendFactor::Zero,
                BlendFactor::DestinationAlpha,
                BlendFactor::Zero,
            ),
            BlendMode::Screen => (
                BlendFactor::One,
                BlendFactor::OneMinusSourceColor,
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
            ),
            _ => (
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
            ),
        };
        desc.src_color_blend_factor = sc;
        desc.dst_color_blend_factor = dc;
        desc.src_alpha_blend_factor = sa;
        desc.dst_alpha_blend_factor = da;
        desc
    }
}

/// Depth attachment pipeline state (`DepthAttachmentDescriptor` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DepthAttachmentDescriptor {
    pub depth_compare: CompareFunction,
    pub depth_write_enabled: bool,
}

impl Default for DepthAttachmentDescriptor {
    fn default() -> Self {
        Self {
            depth_compare: CompareFunction::Always,
            depth_write_enabled: false,
        }
    }
}

/// Stencil attachment pipeline state (`StencilAttachmentDescriptor` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StencilAttachmentDescriptor {
    pub stencil_compare: CompareFunction,
    pub stencil_failure: StencilOperation,
    pub depth_failure: StencilOperation,
    pub depth_stencil_pass: StencilOperation,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl Default for StencilAttachmentDescriptor {
    fn default() -> Self {
        Self {
            stencil_compare: CompareFunction::Always,
            stencil_failure: StencilOperation::Keep,
            depth_failure: StencilOperation::Keep,
            depth_stencil_pass: StencilOperation::Keep,
            read_mask: !0,
            write_mask: !0,
        }
    }
}

/// Minification/magnification filter (`MinMagFilter` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum MinMagFilter {
    #[default]
    Nearest = 0,
    Linear = 1,
}

/// Mipmap filter (`MipFilter` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum MipFilter {
    Base = 0,
    #[default]
    Nearest = 1,
    Linear = 2,
}

/// Sampler wrap/address mode (`SamplerAddressMode` in `formats.h`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum SamplerAddressMode {
    #[default]
    ClampToEdge = 0,
    Repeat = 1,
    Mirror = 2,
    Decal = 3,
}

/// Texture sampler descriptor (`SamplerDescriptor` in `sampler_descriptor.h`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SamplerDescriptor {
    pub min_filter: MinMagFilter,
    pub mag_filter: MinMagFilter,
    pub mip_filter: MipFilter,
    pub width_address_mode: SamplerAddressMode,
    pub height_address_mode: SamplerAddressMode,
    pub depth_address_mode: SamplerAddressMode,
    pub max_anisotropy: u8,
    pub label: String,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            min_filter: MinMagFilter::Nearest,
            mag_filter: MinMagFilter::Nearest,
            mip_filter: MipFilter::Nearest,
            width_address_mode: SamplerAddressMode::ClampToEdge,
            height_address_mode: SamplerAddressMode::ClampToEdge,
            depth_address_mode: SamplerAddressMode::ClampToEdge,
            max_anisotropy: 1,
            label: "NN Clamp Sampler".to_string(),
        }
    }
}

impl SamplerDescriptor {
    pub fn new(
        label: impl Into<String>,
        min_filter: MinMagFilter,
        mag_filter: MinMagFilter,
        mip_filter: MipFilter,
    ) -> Self {
        Self {
            min_filter,
            mag_filter,
            mip_filter,
            label: label.into(),
            ..Default::default()
        }
    }

    /// Packs filter, address, and anisotropy settings into a 64-bit cache key (`SamplerDescriptor::ToKey`).
    pub const fn to_key(&self) -> u64 {
        (self.min_filter as u64)
            | ((self.mag_filter as u64) << 8)
            | ((self.mip_filter as u64) << 16)
            | ((self.width_address_mode as u64) << 24)
            | ((self.height_address_mode as u64) << 32)
            | ((self.depth_address_mode as u64) << 40)
            | ((self.max_anisotropy as u64) << 48)
    }
}

/// Texture dimensionality and type (`TextureType` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum TextureType {
    #[default]
    Texture2D,
    Texture2DMultisample,
    TextureCube,
    TextureExternalOES,
    Texture2DArray,
}

impl TextureType {
    pub const fn is_multisample_capable(&self) -> bool {
        matches!(self, TextureType::Texture2DMultisample)
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            TextureType::Texture2D => "Texture2D",
            TextureType::Texture2DMultisample => "Texture2DMultisample",
            TextureType::TextureCube => "TextureCube",
            TextureType::TextureExternalOES => "TextureExternalOES",
            TextureType::Texture2DArray => "Texture2DArray",
        }
    }
}

/// Single texture usage flag (`TextureUsage` in `formats.h`).
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum TextureUsage {
    #[default]
    Unknown = 0,
    ShaderRead = 1 << 0,
    ShaderWrite = 1 << 1,
    RenderTarget = 1 << 2,
}

impl TextureUsage {
    pub const fn as_str(&self) -> &'static str {
        match self {
            TextureUsage::Unknown => "Unknown",
            TextureUsage::ShaderRead => "ShaderRead",
            TextureUsage::ShaderWrite => "ShaderWrite",
            TextureUsage::RenderTarget => "RenderTarget",
        }
    }
}

/// Bitmask of `TextureUsage` flags (`TextureUsageMask` in `formats.h`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureUsageMask(pub u32);

impl TextureUsageMask {
    pub const UNKNOWN: Self = Self(0);
    pub const SHADER_READ: Self = Self(TextureUsage::ShaderRead as u32);
    pub const SHADER_WRITE: Self = Self(TextureUsage::ShaderWrite as u32);
    pub const RENDER_TARGET: Self = Self(TextureUsage::RenderTarget as u32);

    pub const fn bits(&self) -> u32 {
        self.0
    }

    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl Default for TextureUsageMask {
    fn default() -> Self {
        Self::SHADER_READ
    }
}

impl From<TextureUsage> for TextureUsageMask {
    fn from(usage: TextureUsage) -> Self {
        Self(usage as u32)
    }
}

impl BitOr for TextureUsageMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for TextureUsageMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for TextureUsageMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for TextureUsageMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Additional compression mode for textures (`CompressionType` in `texture_descriptor.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum CompressionType {
    #[default]
    Lossless,
    Lossy,
}

impl CompressionType {
    pub const fn as_str(&self) -> &'static str {
        match self {
            CompressionType::Lossless => "Lossless",
            CompressionType::Lossy => "Lossy",
        }
    }
}

/// Texture allocation descriptor (`TextureDescriptor` in `texture_descriptor.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub storage_mode: StorageMode,
    pub texture_type: TextureType,
    pub format: PixelFormat,
    pub size: ISize,
    pub mip_count: u32,
    pub usage: TextureUsageMask,
    pub sample_count: SampleCount,
    pub compression_type: CompressionType,
    pub array_layer_count: u16,
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self {
            storage_mode: StorageMode::DeviceTransient,
            texture_type: TextureType::Texture2D,
            format: PixelFormat::Unknown,
            size: ISize::new(0, 0),
            mip_count: 1,
            usage: TextureUsageMask::SHADER_READ,
            sample_count: SampleCount::Count1,
            compression_type: CompressionType::Lossless,
            array_layer_count: 1,
        }
    }
}

impl TextureDescriptor {
    pub const fn get_byte_size_for_dimensions(&self, width: i64, height: i64) -> usize {
        self.format.bytes_for_texture_region(width, height)
    }

    pub fn get_byte_size_of_base_mip_level(&self) -> usize {
        if !self.is_valid() {
            return 0;
        }
        self.get_byte_size_for_dimensions(self.size.width, self.size.height)
    }

    pub fn get_byte_size_of_all_mip_levels(&self) -> usize {
        if !self.is_valid() {
            return 0;
        }
        let mut result = 0usize;
        let mut width = self.size.width;
        let mut height = self.size.height;
        for _ in 0..self.mip_count {
            result += self.get_byte_size_for_dimensions(width, height);
            width /= 2;
            height /= 2;
        }
        result
    }

    pub fn get_bytes_per_row(&self) -> usize {
        if !self.is_valid() {
            return 0;
        }
        self.format.bytes_per_row_for_width(self.size.width)
    }

    pub const fn sampling_options_are_valid(&self) -> bool {
        let count = self.sample_count.count();
        if self.texture_type.is_multisample_capable() {
            count > 1
        } else {
            count == 1
        }
    }

    pub fn max_mip_count(&self) -> u32 {
        if self.size.width <= 0 || self.size.height <= 0 {
            return 1;
        }
        let min_dim = self.size.width.min(self.size.height) as u64;
        min_dim.ilog2().max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.format == PixelFormat::Unknown {
            return Err("A pixel format must be specified.".to_string());
        }
        if self.size.width <= 0 || self.size.height <= 0 {
            return Err(format!(
                "The size {}x{} must be nonempty.",
                self.size.width, self.size.height
            ));
        }
        if self.mip_count < 1 {
            return Err("The mip count must be at least one.".to_string());
        }
        if self.texture_type == TextureType::Texture2DArray && self.array_layer_count < 1 {
            return Err("A 2D array texture must have at least one layer.".to_string());
        }
        if !self.sampling_options_are_valid() {
            return Err(format!(
                "The sample count {} is not valid for the texture type {}.",
                self.sample_count.count(),
                self.texture_type.as_str()
            ));
        }
        Ok(())
    }

    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

/// Render target attachment (`Attachment`, `ColorAttachment`, `DepthAttachment`, `StencilAttachment` in `formats.h`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Attachment {
    pub texture_desc: Option<TextureDescriptor>,
    pub resolve_texture_desc: Option<TextureDescriptor>,
    pub load_action: LoadAction,
    pub store_action: StoreAction,
    pub mip_level: u32,
    pub slice: u32,
    pub clear_color: Color,
    pub clear_depth: f64,
    pub clear_stencil: u32,
}

impl Default for Attachment {
    fn default() -> Self {
        Self {
            texture_desc: None,
            resolve_texture_desc: None,
            load_action: LoadAction::DontCare,
            store_action: StoreAction::Store,
            mip_level: 0,
            slice: 0,
            clear_color: Color::TRANSPARENT,
            clear_depth: 0.0,
            clear_stencil: 0,
        }
    }
}

impl Attachment {
    pub fn is_valid(&self) -> bool {
        let Some(desc) = &self.texture_desc else {
            return false;
        };
        if !desc.is_valid() {
            return false;
        }
        if self.mip_level >= desc.mip_count {
            return false;
        }
        let slice_count = match desc.texture_type {
            TextureType::TextureCube => 6,
            TextureType::Texture2DArray => u32::from(desc.array_layer_count),
            _ => 1,
        };
        if self.slice >= slice_count {
            return false;
        }
        if self.store_action.needs_resolve_texture() {
            let Some(resolve) = &self.resolve_texture_desc else {
                return false;
            };
            if !resolve.is_valid() {
                return false;
            }
        }
        if self.resolve_texture_desc.is_some() && !self.store_action.needs_resolve_texture() {
            return false;
        }
        let storage_mode = self
            .resolve_texture_desc
            .as_ref()
            .map_or(desc.storage_mode, |r| r.storage_mode);
        if storage_mode == StorageMode::DeviceTransient {
            if self.load_action == LoadAction::Load {
                return false;
            }
            if self.store_action != StoreAction::DontCare {
                return false;
            }
        }
        true
    }
}

/// Shader reflection scalar type (`ShaderType` in `shader_types.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ShaderType {
    #[default]
    Unknown,
    Void,
    Boolean,
    SignedByte,
    UnsignedByte,
    SignedShort,
    UnsignedShort,
    SignedInt,
    UnsignedInt,
    SignedInt64,
    UnsignedInt64,
    AtomicCounter,
    HalfFloat,
    Float,
    Double,
    Struct,
    Image,
    SampledImage,
    Sampler,
}

/// Vertex attribute format derived from shader stage I/O slot (`VertexAttributeFormat` in `shader_types.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum VertexAttributeFormat {
    #[default]
    Invalid,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Float16,
    Float16x2,
    Float16x3,
    Float16x4,
    SInt8,
    SInt8x2,
    SInt8x3,
    SInt8x4,
    UInt8,
    UInt8x2,
    UInt8x3,
    UInt8x4,
    SInt16,
    SInt16x2,
    SInt16x3,
    SInt16x4,
    UInt16,
    UInt16x2,
    UInt16x3,
    UInt16x4,
    SInt32,
    SInt32x2,
    SInt32x3,
    SInt32x4,
    UInt32,
    UInt32x2,
    UInt32x3,
    UInt32x4,
}

/// Shader stage input/output slot metadata (`ShaderStageIOSlot` in `shader_types.h`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderStageIOSlot {
    pub name: String,
    pub location: usize,
    pub set: usize,
    pub binding: usize,
    pub shader_type: ShaderType,
    pub bit_width: usize,
    pub vec_size: usize,
    pub columns: usize,
    pub offset: usize,
    pub relaxed_precision: bool,
}

impl ShaderStageIOSlot {
    pub const fn get_vertex_attribute_format(&self) -> VertexAttributeFormat {
        if self.columns != 1 || self.vec_size < 1 || self.vec_size > 4 {
            return VertexAttributeFormat::Invalid;
        }
        const fn pick(
            vec_size: usize,
            x1: VertexAttributeFormat,
            x2: VertexAttributeFormat,
            x3: VertexAttributeFormat,
            x4: VertexAttributeFormat,
        ) -> VertexAttributeFormat {
            match vec_size {
                1 => x1,
                2 => x2,
                3 => x3,
                4 => x4,
                _ => VertexAttributeFormat::Invalid,
            }
        }
        match self.shader_type {
            ShaderType::Float if self.bit_width == 32 => pick(
                self.vec_size,
                VertexAttributeFormat::Float32,
                VertexAttributeFormat::Float32x2,
                VertexAttributeFormat::Float32x3,
                VertexAttributeFormat::Float32x4,
            ),
            ShaderType::HalfFloat if self.bit_width == 16 => pick(
                self.vec_size,
                VertexAttributeFormat::Float16,
                VertexAttributeFormat::Float16x2,
                VertexAttributeFormat::Float16x3,
                VertexAttributeFormat::Float16x4,
            ),
            ShaderType::SignedByte if self.bit_width == 8 => pick(
                self.vec_size,
                VertexAttributeFormat::SInt8,
                VertexAttributeFormat::SInt8x2,
                VertexAttributeFormat::SInt8x3,
                VertexAttributeFormat::SInt8x4,
            ),
            ShaderType::UnsignedByte if self.bit_width == 8 => pick(
                self.vec_size,
                VertexAttributeFormat::UInt8,
                VertexAttributeFormat::UInt8x2,
                VertexAttributeFormat::UInt8x3,
                VertexAttributeFormat::UInt8x4,
            ),
            ShaderType::SignedShort if self.bit_width == 16 => pick(
                self.vec_size,
                VertexAttributeFormat::SInt16,
                VertexAttributeFormat::SInt16x2,
                VertexAttributeFormat::SInt16x3,
                VertexAttributeFormat::SInt16x4,
            ),
            ShaderType::UnsignedShort if self.bit_width == 16 => pick(
                self.vec_size,
                VertexAttributeFormat::UInt16,
                VertexAttributeFormat::UInt16x2,
                VertexAttributeFormat::UInt16x3,
                VertexAttributeFormat::UInt16x4,
            ),
            ShaderType::SignedInt if self.bit_width == 32 => pick(
                self.vec_size,
                VertexAttributeFormat::SInt32,
                VertexAttributeFormat::SInt32x2,
                VertexAttributeFormat::SInt32x3,
                VertexAttributeFormat::SInt32x4,
            ),
            ShaderType::UnsignedInt if self.bit_width == 32 => pick(
                self.vec_size,
                VertexAttributeFormat::UInt32,
                VertexAttributeFormat::UInt32x2,
                VertexAttributeFormat::UInt32x3,
                VertexAttributeFormat::UInt32x4,
            ),
            _ => VertexAttributeFormat::Invalid,
        }
    }
}

/// Callback interface invoked when a `HostBuffer` is dropped (`IdleWaiter` in `idle_waiter.h`).
pub trait IdleWaiter: std::fmt::Debug + Send + Sync {
    fn wait_idle(&self);
}

/// Tracks GPU completion of submitted command buffers as a monotonic watermark (`GpuSubmissionTracker` in `gpu_submission_tracker.h`).
#[derive(Debug, Default)]
pub struct GpuSubmissionTracker {
    state: Mutex<GpuSubmissionTrackerState>,
}

#[derive(Debug, Default)]
struct GpuSubmissionTrackerState {
    last_id: u64,
    pending: Vec<u64>,
}

impl GpuSubmissionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_submission(&self) -> u64 {
        let mut guard = self.state.lock().unwrap();
        guard.last_id += 1;
        let id = guard.last_id;
        guard.pending.push(id);
        id
    }

    pub fn record_completion(&self, id: u64) {
        let mut guard = self.state.lock().unwrap();
        if let Ok(idx) = guard.pending.binary_search(&id) {
            guard.pending.remove(idx);
        }
    }

    pub fn completed_through(&self) -> u64 {
        let guard = self.state.lock().unwrap();
        match guard.pending.first() {
            Some(&first_pending) => first_pending - 1,
            None => guard.last_id,
        }
    }

    pub fn latest_submission(&self) -> u64 {
        let guard = self.state.lock().unwrap();
        guard.last_id
    }
}

/// Internal state snapshot of `HostBuffer` (`HostBuffer::TestStateQuery` in `host_buffer.h`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HostBufferTestState {
    pub current_frame: usize,
    pub current_buffer: usize,
    pub total_buffer_count: usize,
}

/// Multi-block, 4-frame ring arena allocator with alignment padding (`HostBuffer` in `host_buffer.h`).
#[derive(Clone, Debug)]
pub struct HostBuffer {
    arena: [Vec<DeviceBuffer>; K_HOST_BUFFER_ARENA_SIZE],
    one_off_buffers: Vec<DeviceBuffer>,
    entry_stamps: [u64; K_HOST_BUFFER_ARENA_SIZE],
    retired_buffers: Vec<(u64, Vec<DeviceBuffer>)>,
    frame_index: usize,
    current_buffer: usize,
    block_capacity: usize,
    minimum_uniform_alignment: usize,
    max_allocation_size: Option<usize>,
    idle_waiter: Option<Arc<dyn IdleWaiter>>,
    submission_tracker: Option<Arc<GpuSubmissionTracker>>,
}

impl Default for HostBuffer {
    fn default() -> Self {
        Self::new(K_DEFAULT_BLOCK_CAPACITY, 256)
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if let Some(waiter) = &self.idle_waiter {
            waiter.wait_idle();
        }
    }
}

impl HostBuffer {
    pub fn new(block_capacity: usize, minimum_uniform_alignment: usize) -> Self {
        let cap = if block_capacity == 0 {
            K_DEFAULT_BLOCK_CAPACITY
        } else {
            block_capacity
        };
        let arena = std::array::from_fn(|_| vec![DeviceBuffer::new_arena_block(cap)]);
        Self {
            arena,
            one_off_buffers: Vec::new(),
            entry_stamps: [0; K_HOST_BUFFER_ARENA_SIZE],
            retired_buffers: Vec::new(),
            frame_index: 0,
            current_buffer: 0,
            block_capacity: cap,
            minimum_uniform_alignment: minimum_uniform_alignment.max(1),
            max_allocation_size: None,
            idle_waiter: None,
            submission_tracker: None,
        }
    }

    pub fn with_tracker_and_waiter(
        block_capacity: usize,
        minimum_uniform_alignment: usize,
        idle_waiter: Option<Arc<dyn IdleWaiter>>,
        submission_tracker: Option<Arc<GpuSubmissionTracker>>,
    ) -> Self {
        let mut hb = Self::new(block_capacity, minimum_uniform_alignment);
        hb.idle_waiter = idle_waiter;
        hb.submission_tracker = submission_tracker;
        hb
    }

    pub fn set_max_allocation_size(&mut self, max_size: Option<usize>) {
        self.max_allocation_size = max_size;
    }

    pub const fn min_uniform_alignment(&self) -> usize {
        self.minimum_uniform_alignment
    }

    pub const fn get_minimum_uniform_alignment(&self) -> usize {
        self.minimum_uniform_alignment
    }

    pub const fn block_capacity(&self) -> usize {
        self.block_capacity
    }

    pub fn get_state_for_test(&self) -> HostBufferTestState {
        HostBufferTestState {
            current_frame: self.frame_index,
            current_buffer: self.current_buffer,
            total_buffer_count: self.arena[self.frame_index].len(),
        }
    }

    fn maybe_create_new_buffer(&mut self) -> bool {
        if let Some(max_size) = self.max_allocation_size {
            if self.block_capacity >= max_size {
                return false;
            }
        }
        self.current_buffer += 1;
        if self.current_buffer >= self.arena[self.frame_index].len() {
            let new_buf = DeviceBuffer::new_arena_block(self.block_capacity);
            self.arena[self.frame_index].push(new_buf);
        } else {
            self.arena[self.frame_index][self.current_buffer]
                .bytes
                .clear();
        }
        true
    }

    pub fn emplace_bytes(&mut self, bytes: &[u8], alignment: usize) -> BufferView {
        let length = bytes.len();
        if let Some(max_size) = self.max_allocation_size {
            if length >= max_size {
                return BufferView::empty();
            }
        }

        // If the requested allocation is larger than `block_capacity`, allocate a one-off buffer
        // without advancing `current_buffer` or disturbing the active arena block (`host_buffer.cc`).
        if length > self.block_capacity {
            let mut one_off = DeviceBuffer::new_arena_block(length);
            one_off.bytes.extend_from_slice(bytes);
            let idx = self.one_off_buffers.len();
            self.one_off_buffers.push(one_off);
            return BufferView {
                buffer_index: ONE_OFF_BUFFER_TAG | idx,
                range: Range::new(0, length),
            };
        }

        let align = alignment.max(1);
        let cur_len = self.arena[self.frame_index][self.current_buffer]
            .bytes
            .len();
        let rem = cur_len % align;
        let pad = if rem == 0 { 0 } else { align - rem };

        if cur_len + pad + length > self.block_capacity {
            if !self.maybe_create_new_buffer() {
                return BufferView::empty();
            }
        } else if pad > 0 {
            let new_len = cur_len + pad;
            self.arena[self.frame_index][self.current_buffer]
                .bytes
                .resize(new_len, 0);
        }

        let buf = &mut self.arena[self.frame_index][self.current_buffer].bytes;
        let offset = buf.len();
        buf.extend_from_slice(bytes);
        BufferView {
            buffer_index: self.current_buffer,
            range: Range::new(offset, length),
        }
    }

    /// Emplaces `length` bytes aligned to `alignment` and invokes `cb` to populate the slice in place.
    pub fn emplace_proc<F>(&mut self, length: usize, alignment: usize, cb: F) -> BufferView
    where
        F: FnOnce(&mut [u8]),
    {
        if let Some(max_size) = self.max_allocation_size {
            if length >= max_size {
                return BufferView::empty();
            }
        }

        if length > self.block_capacity {
            let mut one_off = DeviceBuffer::new_arena_block(length);
            one_off.bytes.resize(length, 0);
            cb(&mut one_off.bytes);
            let idx = self.one_off_buffers.len();
            self.one_off_buffers.push(one_off);
            return BufferView {
                buffer_index: ONE_OFF_BUFFER_TAG | idx,
                range: Range::new(0, length),
            };
        }

        let align = alignment.max(1);
        let cur_len = self.arena[self.frame_index][self.current_buffer]
            .bytes
            .len();
        let rem = cur_len % align;
        let pad = if rem == 0 { 0 } else { align - rem };

        if cur_len + pad + length > self.block_capacity {
            if !self.maybe_create_new_buffer() {
                return BufferView::empty();
            }
        } else if pad > 0 {
            let new_len = cur_len + pad;
            self.arena[self.frame_index][self.current_buffer]
                .bytes
                .resize(new_len, 0);
        }

        let buf = &mut self.arena[self.frame_index][self.current_buffer].bytes;
        let offset = buf.len();
        buf.resize(offset + length, 0);
        cb(&mut buf[offset..offset + length]);
        BufferView {
            buffer_index: self.current_buffer,
            range: Range::new(offset, length),
        }
    }

    pub fn emplace<T: Pod>(&mut self, slice: &[T]) -> BufferView {
        self.emplace_bytes(bytemuck::cast_slice(slice), std::mem::align_of::<T>())
    }

    pub fn emplace_slice<T: Pod>(&mut self, slice: &[T]) -> BufferView {
        self.emplace_bytes(bytemuck::cast_slice(slice), std::mem::align_of::<T>())
    }

    pub fn emplace_value<T: Pod>(&mut self, value: &T) -> BufferView {
        self.emplace_bytes(bytemuck::bytes_of(value), std::mem::align_of::<T>())
    }

    pub fn emplace_with_alignment<T: Pod>(&mut self, value: &T, alignment: usize) -> BufferView {
        self.emplace_bytes(
            bytemuck::bytes_of(value),
            alignment.max(std::mem::align_of::<T>()),
        )
    }

    pub fn emplace_uniform<T: Pod>(&mut self, value: &T) -> BufferView {
        let align = std::mem::align_of::<T>().max(self.minimum_uniform_alignment);
        self.emplace_bytes(bytemuck::bytes_of(value), align)
    }

    pub fn emplace_storage_buffer<T: Pod>(&mut self, value: &T) -> BufferView {
        let align = std::mem::align_of::<T>().max(self.minimum_uniform_alignment);
        self.emplace_bytes(bytemuck::bytes_of(value), align)
    }

    pub fn get_device_buffer(&self, view: BufferView) -> Option<&DeviceBuffer> {
        if view.is_empty() {
            return None;
        }
        if (view.buffer_index & ONE_OFF_BUFFER_TAG) != 0 {
            let idx = view.buffer_index & !ONE_OFF_BUFFER_TAG;
            self.one_off_buffers.get(idx)
        } else {
            self.arena[self.frame_index].get(view.buffer_index)
        }
    }

    pub fn get_slice(&self, view: BufferView) -> &[u8] {
        let buf = if (view.buffer_index & ONE_OFF_BUFFER_TAG) != 0 {
            let idx = view.buffer_index & !ONE_OFF_BUFFER_TAG;
            &self.one_off_buffers[idx].bytes
        } else {
            &self.arena[self.frame_index][view.buffer_index].bytes
        };
        &buf[view.range.offset..view.range.offset + view.range.length]
    }

    pub fn reset(&mut self) {
        while self.arena[self.frame_index].len() > self.current_buffer + 1 {
            self.arena[self.frame_index].pop();
        }

        if let Some(tracker) = &self.submission_tracker {
            self.entry_stamps[self.frame_index] = tracker.latest_submission();
        }

        self.current_buffer = 0;
        self.one_off_buffers.clear();
        self.frame_index = (self.frame_index + 1) % K_HOST_BUFFER_ARENA_SIZE;

        if let Some(tracker) = &self.submission_tracker {
            let completed = tracker.completed_through();
            self.retired_buffers
                .retain(|(stamp, _)| *stamp > completed);
            if self.entry_stamps[self.frame_index] > completed {
                let old_buffers = std::mem::take(&mut self.arena[self.frame_index]);
                self.retired_buffers
                    .push((self.entry_stamps[self.frame_index], old_buffers));
                self.arena[self.frame_index]
                    .push(DeviceBuffer::new_arena_block(self.block_capacity));
                self.entry_stamps[self.frame_index] = 0;
            }
        }

        for buf in &mut self.arena[self.frame_index] {
            buf.bytes.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
    struct Length2 {
        pad: [u8; 2],
    }

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
    struct Align16 {
        pad: [u8; 16],
    }

    #[derive(Debug, Default)]
    struct MockIdleWaiter {
        calls: AtomicUsize,
    }

    impl IdleWaiter for MockIdleWaiter {
        fn wait_idle(&self) {
            self.calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// C++ source: allocator_unittests.cc:TextureDescriptorCompatibility
    #[test]
    fn test_texture_descriptor_compatibility() {
        // Size.
        {
            let desc_a = TextureDescriptor {
                size: ISize::new(100, 100),
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                size: ISize::new(100, 100),
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                size: ISize::new(101, 100),
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
        // Storage Mode.
        {
            let desc_a = TextureDescriptor {
                storage_mode: StorageMode::DevicePrivate,
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                storage_mode: StorageMode::DevicePrivate,
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                storage_mode: StorageMode::HostVisible,
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
        // Format.
        {
            let desc_a = TextureDescriptor {
                format: PixelFormat::R8G8B8A8UNormInt,
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                format: PixelFormat::R8G8B8A8UNormInt,
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                format: PixelFormat::B10G10R10A10XR,
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
        // Sample Count.
        {
            let desc_a = TextureDescriptor {
                sample_count: SampleCount::Count4,
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                sample_count: SampleCount::Count4,
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                sample_count: SampleCount::Count1,
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
        // Texture Type.
        {
            let desc_a = TextureDescriptor {
                texture_type: TextureType::Texture2DMultisample,
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                texture_type: TextureType::Texture2DMultisample,
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                texture_type: TextureType::Texture2D,
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
        // Compression.
        {
            let desc_a = TextureDescriptor {
                compression_type: CompressionType::Lossless,
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                compression_type: CompressionType::Lossless,
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                compression_type: CompressionType::Lossy,
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
        // Mip Count.
        {
            let desc_a = TextureDescriptor {
                mip_count: 1,
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                mip_count: 1,
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                mip_count: 4,
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
        // Array layer count.
        {
            let desc_a = TextureDescriptor {
                texture_type: TextureType::Texture2DArray,
                array_layer_count: 4,
                ..Default::default()
            };
            let desc_b = TextureDescriptor {
                texture_type: TextureType::Texture2DArray,
                array_layer_count: 4,
                ..Default::default()
            };
            let desc_c = TextureDescriptor {
                texture_type: TextureType::Texture2DArray,
                array_layer_count: 8,
                ..Default::default()
            };
            assert_eq!(desc_a, desc_b);
            assert_ne!(desc_a, desc_c);
        }
    }

    /// C++ source: allocator_unittests.cc:TextureDescriptorArrayValidity
    #[test]
    fn test_texture_descriptor_array_validity() {
        let mut desc = TextureDescriptor {
            texture_type: TextureType::Texture2DArray,
            format: PixelFormat::R8G8B8A8UNormInt,
            size: ISize::new(16, 16),
            array_layer_count: 4,
            ..Default::default()
        };
        assert!(desc.is_valid());

        desc.array_layer_count = 0;
        assert!(!desc.is_valid());
        let err = desc.validate().unwrap_err();
        assert!(err.contains("layer"), "unexpected error: {err}");
    }

    /// C++ source: allocator_unittests.cc:RangeTest
    #[test]
    fn test_range_merge() {
        {
            let a = Range::new(0, 10);
            let b = Range::new(10, 20);
            let merged = a.merge(b);
            assert_eq!(merged.offset, 0);
            assert_eq!(merged.length, 30);
        }
        {
            let a = Range::new(0, 10);
            let b = Range::new(100, 20);
            let merged = a.merge(b);
            assert_eq!(merged.offset, 0);
            assert_eq!(merged.length, 120);
        }
        {
            let a = Range::new(0, 10);
            let b = Range::new(100, 20);
            let merged = b.merge(a);
            assert_eq!(merged.offset, 0);
            assert_eq!(merged.length, 120);
        }
        {
            let a = Range::new(0, 10);
            let b = Range::new(100, 0);
            let merged = b.merge(a);
            assert_eq!(merged.offset, 0);
            assert_eq!(merged.length, 10);
        }
        {
            let a = Range::new(0, 10);
            let b = Range::new(0, 10);
            let merged = b.merge(a);
            assert_eq!(merged.offset, 0);
            assert_eq!(merged.length, 10);
        }
    }

    /// C++ source: allocator_unittests.cc:CompressedFormatClassification
    #[test]
    fn test_compressed_format_classification() {
        assert!(!is_compressed(PixelFormat::R8G8B8A8UNormInt));
        assert!(!is_compressed(PixelFormat::R32Float));
        assert!(is_compressed(PixelFormat::BC1RGBAUNormInt));
        assert!(is_compressed(PixelFormat::ETC2RGBA8UNormInt));
        assert!(is_compressed(PixelFormat::ASTC8x8LDR));
        assert!(is_compressed(PixelFormat::ASTC4x4HDR));

        assert_eq!(
            compressed_texture_family_for_format(PixelFormat::BC7RGBAUNormInt),
            Some(CompressedTextureFamily::BC)
        );
        assert_eq!(
            compressed_texture_family_for_format(PixelFormat::ETC2RGB8UNormInt),
            Some(CompressedTextureFamily::ETC2)
        );
        assert_eq!(
            compressed_texture_family_for_format(PixelFormat::ASTC4x4LDR),
            Some(CompressedTextureFamily::ASTC)
        );
        assert_eq!(
            compressed_texture_family_for_format(PixelFormat::ASTC8x8HDR),
            Some(CompressedTextureFamily::ASTCHDR)
        );
    }

    /// C++ source: allocator_unittests.cc:CompressedFormatBlockMath
    #[test]
    fn test_compressed_format_block_math() {
        assert_eq!(
            compressed_block_width_for_pixel_format(PixelFormat::BC1RGBAUNormInt),
            4
        );
        assert_eq!(
            compressed_block_width_for_pixel_format(PixelFormat::ASTC8x8LDR),
            8
        );
        assert_eq!(
            compressed_block_height_for_pixel_format(PixelFormat::ASTC8x8LDR),
            8
        );
        assert_eq!(
            compressed_block_width_for_pixel_format(PixelFormat::ASTC4x4HDR),
            4
        );
        assert_eq!(
            compressed_block_height_for_pixel_format(PixelFormat::ASTC8x8HDR),
            8
        );
        assert_eq!(
            compressed_block_width_for_pixel_format(PixelFormat::R8G8B8A8UNormInt),
            1
        );

        assert_eq!(
            bytes_per_block_for_pixel_format(PixelFormat::BC1RGBAUNormInt),
            8
        );
        assert_eq!(
            bytes_per_block_for_pixel_format(PixelFormat::ETC2RGB8UNormInt),
            8
        );
        assert_eq!(
            bytes_per_block_for_pixel_format(PixelFormat::BC3RGBAUNormInt),
            16
        );
        assert_eq!(
            bytes_per_block_for_pixel_format(PixelFormat::ASTC4x4LDR),
            16
        );
        assert_eq!(
            bytes_per_block_for_pixel_format(PixelFormat::ASTC8x8HDR),
            16
        );
        assert_eq!(
            bytes_per_block_for_pixel_format(PixelFormat::R8G8B8A8UNormInt),
            4
        );
    }

    /// C++ source: allocator_unittests.cc:CompressedFormatRegionByteSizes
    #[test]
    fn test_compressed_format_region_byte_sizes() {
        assert_eq!(
            bytes_for_texture_region(PixelFormat::BC1RGBAUNormInt, 4, 4),
            8
        );
        assert_eq!(
            bytes_for_texture_region(PixelFormat::BC1RGBAUNormInt, 8, 8),
            32
        );
        assert_eq!(
            bytes_for_texture_region(PixelFormat::BC1RGBAUNormInt, 5, 5),
            32
        );
        assert_eq!(
            bytes_per_row_for_texture_width(PixelFormat::BC1RGBAUNormInt, 8),
            16
        );

        assert_eq!(bytes_for_texture_region(PixelFormat::ASTC8x8LDR, 8, 8), 16);
        assert_eq!(bytes_for_texture_region(PixelFormat::ASTC8x8LDR, 9, 9), 64);

        assert_eq!(
            bytes_for_texture_region(PixelFormat::R8G8B8A8UNormInt, 100, 100),
            40000
        );
        assert_eq!(
            bytes_per_row_for_texture_width(PixelFormat::R8G8B8A8UNormInt, 100),
            400
        );
    }

    /// C++ source: allocator_unittests.cc:CompressedTextureDescriptorByteSizes
    #[test]
    fn test_compressed_texture_descriptor_byte_sizes() {
        {
            let desc = TextureDescriptor {
                format: PixelFormat::BC1RGBAUNormInt,
                size: ISize::new(8, 8),
                ..Default::default()
            };
            assert_eq!(desc.get_byte_size_of_base_mip_level(), 32);
            assert_eq!(desc.get_bytes_per_row(), 16);
        }
        {
            let desc = TextureDescriptor {
                format: PixelFormat::ASTC8x8LDR,
                size: ISize::new(16, 16),
                ..Default::default()
            };
            assert_eq!(desc.get_byte_size_of_base_mip_level(), 64);
        }
    }

    /// C++ source: buffer_view_unittests.cc:Empty
    #[test]
    fn test_buffer_view_empty() {
        let buffer_view = BufferView::default();
        assert!(buffer_view.is_empty());
        assert!(!buffer_view.is_valid());
    }

    /// C++ source: buffer_view_unittests.cc:TakeRaw
    #[test]
    fn test_buffer_view_take_sub_view_and_slice() {
        let mut host_buffer = HostBuffer::default();
        let data: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let buffer_view = host_buffer.emplace_bytes(&data, 1);
        assert!(buffer_view.is_valid());
        assert_eq!(buffer_view.get_range(), Range::new(0, 8));
        assert_eq!(buffer_view.slice(&host_buffer), &data);

        let sub = buffer_view.take_sub_view(2, 4).expect("valid sub view");
        assert_eq!(sub.get_range(), Range::new(2, 4));
        assert_eq!(sub.slice(&host_buffer), &[30, 40, 50, 60]);
        assert!(buffer_view.take_sub_view(5, 4).is_none());
    }

    /// C++ source: host_buffer_unittests.cc:IdleWaiter
    #[test]
    fn test_host_buffer_idle_waiter() {
        let waiter = Arc::new(MockIdleWaiter::default());
        {
            let _buffer = HostBuffer::with_tracker_and_waiter(
                K_DEFAULT_BLOCK_CAPACITY,
                256,
                Some(waiter.clone()),
                None,
            );
            assert_eq!(waiter.calls.load(Ordering::SeqCst), 0);
        }
        assert_eq!(waiter.calls.load(Ordering::SeqCst), 1);
    }

    /// C++ source: host_buffer_unittests.cc:CanEmplace
    #[test]
    fn test_host_buffer_can_emplace() {
        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);
        for i in 0..12_500 {
            let view = buffer.emplace_value(&Length2 { pad: [1, 2] });
            assert!(view.is_valid());
            assert_eq!(
                view.get_range(),
                Range::new(i * std::mem::size_of::<Length2>(), 2)
            );
        }
    }

    /// C++ source: host_buffer_unittests.cc:CanEmplaceWithAlignment
    #[test]
    fn test_host_buffer_can_emplace_with_alignment() {
        assert_eq!(std::mem::size_of::<Length2>(), 2);
        assert_eq!(std::mem::align_of::<Align16>(), 16);
        assert_eq!(std::mem::size_of::<Align16>(), 16);

        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);

        {
            let view = buffer.emplace_value(&Length2::default());
            assert!(view.is_valid());
            assert_eq!(view.get_range(), Range::new(0, 2));
        }
        {
            let view = buffer.emplace_value(&Align16::default());
            assert!(view.is_valid());
            assert_eq!(view.get_range().offset, 16);
            assert_eq!(view.get_range().length, 16);
        }
        {
            let view = buffer.emplace_value(&Length2::default());
            assert!(view.is_valid());
            assert_eq!(view.get_range(), Range::new(32, 2));
        }
        {
            let view = buffer.emplace_value(&Align16::default());
            assert!(view.is_valid());
            assert_eq!(view.get_range().offset, 48);
            assert_eq!(view.get_range().length, 16);
        }
    }

    /// C++ source: host_buffer_unittests.cc:HostBufferInitialState
    #[test]
    fn test_host_buffer_initial_state() {
        let buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);
        let state = buffer.get_state_for_test();
        assert_eq!(state.current_buffer, 0);
        assert_eq!(state.current_frame, 0);
        assert_eq!(state.total_buffer_count, 1);
    }

    /// C++ source: host_buffer_unittests.cc:ResetIncrementsFrameCounter
    #[test]
    fn test_host_buffer_reset_increments_frame_counter() {
        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);
        assert_eq!(buffer.get_state_for_test().current_frame, 0);

        buffer.reset();
        assert_eq!(buffer.get_state_for_test().current_frame, 1);

        buffer.reset();
        assert_eq!(buffer.get_state_for_test().current_frame, 2);

        buffer.reset();
        assert_eq!(buffer.get_state_for_test().current_frame, 3);

        buffer.reset();
        assert_eq!(buffer.get_state_for_test().current_frame, 0);
    }

    /// C++ source: host_buffer_unittests.cc:EmplacingLargerThanBlockSizeCreatesOneOffBufferCallback
    #[test]
    fn test_emplacing_larger_than_block_size_creates_one_off_buffer_callback() {
        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);
        let view = buffer.emplace_proc(K_DEFAULT_BLOCK_CAPACITY + 10, 0, |data| {
            data[0] = 42;
            data[K_DEFAULT_BLOCK_CAPACITY + 9] = 99;
        });
        assert!(view.is_valid());
        assert_eq!(buffer.get_slice(view)[0], 42);
        assert_eq!(buffer.get_slice(view)[K_DEFAULT_BLOCK_CAPACITY + 9], 99);

        let state = buffer.get_state_for_test();
        assert_eq!(state.current_buffer, 0);
        assert_eq!(state.current_frame, 0);
        assert_eq!(state.total_buffer_count, 1);
    }

    /// C++ source: host_buffer_unittests.cc:EmplacingLargerThanBlockSizeCreatesOneOffBuffer
    #[test]
    fn test_emplacing_larger_than_block_size_creates_one_off_buffer() {
        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);
        let payload = vec![7u8; K_DEFAULT_BLOCK_CAPACITY + 10];
        let view = buffer.emplace_bytes(&payload, 0);
        assert!(view.is_valid());
        assert_eq!(buffer.get_slice(view).len(), K_DEFAULT_BLOCK_CAPACITY + 10);

        let state = buffer.get_state_for_test();
        assert_eq!(state.current_buffer, 0);
        assert_eq!(state.current_frame, 0);
        assert_eq!(state.total_buffer_count, 1);
    }

    /// C++ source: host_buffer_unittests.cc:UnusedBuffersAreDiscardedWhenResetting
    #[test]
    fn test_unused_buffers_are_discarded_when_resetting() {
        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);

        let _view_a = buffer.emplace_proc(1_020_000, 0, |_| {});
        let _view_b = buffer.emplace_proc(1_020_000, 0, |_| {});

        assert_eq!(buffer.get_state_for_test().current_buffer, 1);
        assert_eq!(buffer.get_state_for_test().total_buffer_count, 2);
        assert_eq!(buffer.get_state_for_test().current_frame, 0);

        for _ in 0..4 {
            buffer.reset();
        }

        assert_eq!(buffer.get_state_for_test().current_buffer, 0);
        assert_eq!(buffer.get_state_for_test().total_buffer_count, 2);
        assert_eq!(buffer.get_state_for_test().current_frame, 0);

        for _ in 0..4 {
            buffer.reset();
        }

        assert_eq!(buffer.get_state_for_test().current_buffer, 0);
        assert_eq!(buffer.get_state_for_test().total_buffer_count, 1);
        assert_eq!(buffer.get_state_for_test().current_frame, 0);
    }

    /// C++ source: host_buffer_unittests.cc:EmplaceWithProcIsAligned
    #[test]
    fn test_emplace_with_proc_is_aligned() {
        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);

        let view = buffer.emplace_value(&[0u8; 21]);
        assert_eq!(view.get_range(), Range::new(0, 21));

        let view = buffer.emplace_proc(64, 16, |slice| {
            slice.fill(0xAB);
        });
        assert_eq!(view.get_range(), Range::new(32, 64));
        assert!(buffer.get_slice(view).iter().all(|&b| b == 0xAB));
    }

    /// C++ source: host_buffer_unittests.cc:WatermarkHandlesOutOfOrderCompletion
    #[test]
    fn test_watermark_handles_out_of_order_completion() {
        let tracker = GpuSubmissionTracker::new();
        assert_eq!(tracker.completed_through(), 0);
        let a = tracker.record_submission();
        let b = tracker.record_submission();
        let c = tracker.record_submission();
        assert_eq!(tracker.latest_submission(), c);

        tracker.record_completion(b);
        assert_eq!(tracker.completed_through(), 0);
        tracker.record_completion(a);
        assert_eq!(tracker.completed_through(), b);
        tracker.record_completion(c);
        assert_eq!(tracker.completed_through(), c);
    }

    /// C++ source: host_buffer_unittests.cc:PendingEntriesAreReplacedNotReused
    #[test]
    fn test_pending_entries_are_replaced_not_reused() {
        let tracker = Arc::new(GpuSubmissionTracker::new());
        let mut buffer = HostBuffer::with_tracker_and_waiter(
            K_DEFAULT_BLOCK_CAPACITY,
            256,
            None,
            Some(tracker.clone()),
        );

        let first_view = buffer.emplace_uniform(&0.0f32);
        let first_entry_id = buffer.get_device_buffer(first_view).unwrap().id();
        let pending = tracker.record_submission();
        for _ in 0..K_HOST_BUFFER_ARENA_SIZE {
            buffer.reset();
        }

        let replaced_view = buffer.emplace_uniform(&0.0f32);
        let replaced_entry_id = buffer.get_device_buffer(replaced_view).unwrap().id();
        assert_ne!(replaced_entry_id, first_entry_id);

        tracker.record_completion(pending);
        let second_view = buffer.emplace_uniform(&0.0f32);
        let second_entry_id = buffer.get_device_buffer(second_view).unwrap().id();
        for _ in 0..K_HOST_BUFFER_ARENA_SIZE {
            buffer.reset();
        }
        let reused_view = buffer.emplace_uniform(&0.0f32);
        let reused_entry_id = buffer.get_device_buffer(reused_view).unwrap().id();
        assert_eq!(reused_entry_id, second_entry_id);
    }

    /// C++ source: host_buffer_unittests.cc:CompletedEntriesAreReusedWithoutTrackerChurn
    #[test]
    fn test_completed_entries_are_reused_without_tracker_churn() {
        let tracker = Arc::new(GpuSubmissionTracker::new());
        let mut buffer = HostBuffer::with_tracker_and_waiter(
            K_DEFAULT_BLOCK_CAPACITY,
            256,
            None,
            Some(tracker.clone()),
        );

        let view = buffer.emplace_uniform(&0.0f32);
        let entry_id = buffer.get_device_buffer(view).unwrap().id();
        for _ in 0..(K_HOST_BUFFER_ARENA_SIZE * 3) {
            let sub = tracker.record_submission();
            tracker.record_completion(sub);
            buffer.reset();
        }
        let after_view = buffer.emplace_uniform(&0.0f32);
        let after_entry_id = buffer.get_device_buffer(after_view).unwrap().id();
        assert_eq!(after_entry_id, entry_id);
    }

    /// C++ source: host_buffer_unittests.cc:EmplaceWithFailingAllocationDoesntCrash
    #[test]
    fn test_emplace_with_failing_allocation_doesnt_crash() {
        const MAGIC_FAILING_ALLOCATION: usize = K_DEFAULT_BLOCK_CAPACITY * 2;
        let mut buffer = HostBuffer::new(K_DEFAULT_BLOCK_CAPACITY, 256);
        buffer.set_max_allocation_size(Some(MAGIC_FAILING_ALLOCATION));

        let payload = vec![0u8; MAGIC_FAILING_ALLOCATION];
        let view = buffer.emplace_bytes(&payload, 0);
        assert!(buffer.get_device_buffer(view).is_none());
        assert_eq!(view.get_range().offset, 0);
        assert_eq!(view.get_range().length, 0);
    }

    /// C++ source: sampler_descriptor_unittests.cc:ToKeyIncludesAllFields
    #[test]
    fn test_sampler_descriptor_to_key_includes_all_fields() {
        let a = SamplerDescriptor::default();
        let mut b = SamplerDescriptor::default();
        assert_eq!(a.to_key(), b.to_key());

        b.min_filter = MinMagFilter::Linear;
        assert_ne!(a.to_key(), b.to_key());
        b = SamplerDescriptor::default();

        b.mag_filter = MinMagFilter::Linear;
        assert_ne!(a.to_key(), b.to_key());
        b = SamplerDescriptor::default();

        b.mip_filter = MipFilter::Linear;
        assert_ne!(a.to_key(), b.to_key());
        b = SamplerDescriptor::default();

        b.width_address_mode = SamplerAddressMode::Repeat;
        assert_ne!(a.to_key(), b.to_key());
        b = SamplerDescriptor::default();

        b.height_address_mode = SamplerAddressMode::Repeat;
        assert_ne!(a.to_key(), b.to_key());
        b = SamplerDescriptor::default();

        b.depth_address_mode = SamplerAddressMode::Repeat;
        assert_ne!(a.to_key(), b.to_key());
        b = SamplerDescriptor::default();

        b.max_anisotropy = 16;
        assert_ne!(a.to_key(), b.to_key());
        b.max_anisotropy = 1;
        assert_eq!(a.to_key(), b.to_key());
    }

    fn make_slot(
        shader_type: ShaderType,
        bit_width: usize,
        vec_size: usize,
        columns: usize,
    ) -> ShaderStageIOSlot {
        ShaderStageIOSlot {
            name: "test".to_string(),
            location: 0,
            set: 0,
            binding: 0,
            shader_type,
            bit_width,
            vec_size,
            columns,
            offset: 0,
            relaxed_precision: false,
        }
    }

    /// C++ source: shader_types_unittests.cc:VertexAttributeFormatFloat32
    #[test]
    fn test_vertex_attribute_format_float32() {
        assert_eq!(
            make_slot(ShaderType::Float, 32, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Float32
        );
        assert_eq!(
            make_slot(ShaderType::Float, 32, 2, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Float32x2
        );
        assert_eq!(
            make_slot(ShaderType::Float, 32, 3, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Float32x3
        );
        assert_eq!(
            make_slot(ShaderType::Float, 32, 4, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Float32x4
        );
    }

    /// C++ source: shader_types_unittests.cc:VertexAttributeFormatHalfFloat
    #[test]
    fn test_vertex_attribute_format_half_float() {
        assert_eq!(
            make_slot(ShaderType::HalfFloat, 16, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Float16
        );
        assert_eq!(
            make_slot(ShaderType::HalfFloat, 16, 4, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Float16x4
        );
    }

    /// C++ source: shader_types_unittests.cc:VertexAttributeFormatIntegers
    #[test]
    fn test_vertex_attribute_format_integers() {
        assert_eq!(
            make_slot(ShaderType::SignedByte, 8, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::SInt8
        );
        assert_eq!(
            make_slot(ShaderType::UnsignedByte, 8, 4, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::UInt8x4
        );
        assert_eq!(
            make_slot(ShaderType::SignedShort, 16, 2, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::SInt16x2
        );
        assert_eq!(
            make_slot(ShaderType::UnsignedShort, 16, 3, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::UInt16x3
        );
        assert_eq!(
            make_slot(ShaderType::SignedInt, 32, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::SInt32
        );
        assert_eq!(
            make_slot(ShaderType::UnsignedInt, 32, 4, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::UInt32x4
        );
    }

    /// C++ source: shader_types_unittests.cc:VertexAttributeFormatRejectsMatrices
    #[test]
    fn test_vertex_attribute_format_rejects_matrices() {
        assert_eq!(
            make_slot(ShaderType::Float, 32, 4, 4).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
    }

    /// C++ source: shader_types_unittests.cc:VertexAttributeFormatRejectsBadComponentCount
    #[test]
    fn test_vertex_attribute_format_rejects_bad_component_count() {
        assert_eq!(
            make_slot(ShaderType::Float, 32, 0, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
        assert_eq!(
            make_slot(ShaderType::Float, 32, 5, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
    }

    /// C++ source: shader_types_unittests.cc:VertexAttributeFormatRejectsBadBitWidth
    #[test]
    fn test_vertex_attribute_format_rejects_bad_bit_width() {
        assert_eq!(
            make_slot(ShaderType::Float, 16, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
        assert_eq!(
            make_slot(ShaderType::SignedInt, 16, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
    }

    /// C++ source: shader_types_unittests.cc:VertexAttributeFormatRejectsUnsupportedScalarKinds
    #[test]
    fn test_vertex_attribute_format_rejects_unsupported_scalar_kinds() {
        assert_eq!(
            make_slot(ShaderType::Boolean, 8, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
        assert_eq!(
            make_slot(ShaderType::Double, 64, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
        assert_eq!(
            make_slot(ShaderType::SignedInt64, 64, 1, 1).get_vertex_attribute_format(),
            VertexAttributeFormat::Invalid
        );
    }

    /// C++ source: pipeline_descriptor_unittests.cc:BlendModeColorAttachmentDescriptors
    #[test]
    fn test_color_attachment_descriptor_for_all_porter_duff_blend_modes() {
        let fmt = PixelFormat::R8G8B8A8UNormInt;
        let cases = [
            (
                BlendMode::Clear,
                BlendFactor::Zero,
                BlendFactor::Zero,
                BlendFactor::Zero,
                BlendFactor::Zero,
            ),
            (
                BlendMode::Src,
                BlendFactor::One,
                BlendFactor::Zero,
                BlendFactor::One,
                BlendFactor::Zero,
            ),
            (
                BlendMode::Dst,
                BlendFactor::Zero,
                BlendFactor::One,
                BlendFactor::Zero,
                BlendFactor::One,
            ),
            (
                BlendMode::SrcOver,
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
            ),
            (
                BlendMode::DstOver,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::One,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::One,
            ),
            (
                BlendMode::SrcIn,
                BlendFactor::DestinationAlpha,
                BlendFactor::Zero,
                BlendFactor::DestinationAlpha,
                BlendFactor::Zero,
            ),
            (
                BlendMode::DstIn,
                BlendFactor::Zero,
                BlendFactor::SourceAlpha,
                BlendFactor::Zero,
                BlendFactor::SourceAlpha,
            ),
            (
                BlendMode::SrcOut,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::Zero,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::Zero,
            ),
            (
                BlendMode::DstOut,
                BlendFactor::Zero,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::Zero,
                BlendFactor::OneMinusSourceAlpha,
            ),
            (
                BlendMode::SrcATop,
                BlendFactor::DestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::DestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
            ),
            (
                BlendMode::DstATop,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::SourceAlpha,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::SourceAlpha,
            ),
            (
                BlendMode::Xor,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::OneMinusDestinationAlpha,
                BlendFactor::OneMinusSourceAlpha,
            ),
            (
                BlendMode::Plus,
                BlendFactor::One,
                BlendFactor::One,
                BlendFactor::One,
                BlendFactor::One,
            ),
            (
                BlendMode::Modulate,
                BlendFactor::DestinationColor,
                BlendFactor::Zero,
                BlendFactor::DestinationAlpha,
                BlendFactor::Zero,
            ),
            (
                BlendMode::Screen,
                BlendFactor::One,
                BlendFactor::OneMinusSourceColor,
                BlendFactor::One,
                BlendFactor::OneMinusSourceAlpha,
            ),
        ];

        for (mode, sc, dc, sa, da) in cases {
            let desc = ColorAttachmentDescriptor::for_blend_mode(fmt, mode);
            assert_eq!(desc.format, fmt);
            assert!(desc.blending_enabled);
            assert_eq!(desc.src_color_blend_factor, sc, "mode={mode:?}");
            assert_eq!(desc.dst_color_blend_factor, dc, "mode={mode:?}");
            assert_eq!(desc.src_alpha_blend_factor, sa, "mode={mode:?}");
            assert_eq!(desc.dst_alpha_blend_factor, da, "mode={mode:?}");
            assert_eq!(desc.color_blend_op, BlendOperation::Add);
            assert_eq!(desc.alpha_blend_op, BlendOperation::Add);
            assert_eq!(desc.write_mask, ColorWriteMask::ALL);
        }
    }

    /// C++ source: allocator_unittests.cc:PixelFormatAndVertexBufferValidation
    #[test]
    fn test_pixel_format_index_type_and_vertex_buffer() {
        assert!(!PixelFormat::Unknown.is_color());
        assert!(!PixelFormat::Unknown.is_depth_stencil());
        assert!(PixelFormat::R8G8B8A8UNormInt.is_color());
        assert!(!PixelFormat::R8G8B8A8UNormInt.is_depth_stencil());
        assert!(PixelFormat::S8UInt.is_depth_stencil());
        assert!(!PixelFormat::S8UInt.is_depth_writable());
        assert!(PixelFormat::S8UInt.is_stencil_writable());
        assert!(PixelFormat::D24UnormS8Uint.is_depth_stencil());
        assert!(PixelFormat::D24UnormS8Uint.is_depth_writable());
        assert!(PixelFormat::D32FloatS8UInt.is_depth_stencil());
        assert!(!PixelFormat::D32FloatS8UInt.is_color());

        assert_eq!(IndexType::Index16.byte_size(), 2);
        assert_eq!(IndexType::Index32.byte_size(), 4);
        assert_eq!(IndexType::None.byte_size(), 0);
        assert_eq!(IndexType::Unknown.byte_size(), 0);

        let mut hb = HostBuffer::new(64, 16);
        let v_view = hb.emplace(&[1.0f32, 2.0, 3.0, 4.0]);
        let i_view = hb.emplace(&[0u16, 1, 2]);
        let vb = VertexBuffer::new(v_view, i_view, 3, IndexType::Index16);
        assert!(vb.is_valid());

        let unindexed = VertexBuffer::new(v_view, BufferView::empty(), 2, IndexType::None);
        assert!(unindexed.is_valid());

        let invalid_indexed = VertexBuffer::new(v_view, BufferView::empty(), 3, IndexType::Index16);
        assert!(!invalid_indexed.is_valid());
    }
}
