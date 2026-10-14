use std::collections::HashMap;

pub use etagere::AllocId;
use impeller_geometry::{IPoint, ISize, Scalar};

/// Dynamic shelf rectangle packer wrapping [`etagere::AtlasAllocator`], providing
/// Impeller's `RectanglePacker` / `SkylineRectanglePacker` API (`impeller/typographer/rectangle_packer.h`)
/// along with individual rectangle `deallocate(AllocId)` and dynamic `grow(ISize)`.
#[derive(Clone)]
pub struct RectanglePacker {
    size: ISize,
    regions: Vec<(IPoint, etagere::AtlasAllocator)>,
    allocations: HashMap<AllocId, (usize, AllocId, i64)>,
    next_alloc_id: u32,
    used_area: i64,
}

/// Alias matching C++ Impeller's `SkylineRectanglePacker` (`impeller/typographer/rectangle_packer.h`).
pub type SkylineRectanglePacker = RectanglePacker;

impl std::fmt::Debug for RectanglePacker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RectanglePacker")
            .field("size", &self.size)
            .field("regions_len", &self.regions.len())
            .field("allocations_len", &self.allocations.len())
            .field("used_area", &self.used_area)
            .finish()
    }
}

fn make_atlas_allocator(width: i64, height: i64) -> Option<etagere::AtlasAllocator> {
    if width <= 0 || height <= 0 {
        return None;
    }
    let w = i32::try_from(width).ok()?;
    let h = i32::try_from(height).ok()?;
    if w > i32::from(u16::MAX) || h > i32::from(u16::MAX) {
        return None;
    }
    w.checked_mul(h)?;
    Some(etagere::AtlasAllocator::new(etagere::size2(w, h)))
}

impl RectanglePacker {
    /// Creates a new shelf rectangle packer with the given dimensions in pixels.
    pub fn new(width: i64, height: i64) -> Self {
        let clamped_w = width.max(0);
        let clamped_h = height.max(0);
        let mut regions = Vec::new();
        if let Some(alloc) = make_atlas_allocator(clamped_w, clamped_h) {
            regions.push((IPoint::ZERO, alloc));
        }
        Self {
            size: ISize::new(clamped_w, clamped_h),
            regions,
            allocations: HashMap::new(),
            next_alloc_id: 1,
            used_area: 0,
        }
    }

    /// Factory helper matching C++ `RectanglePacker::Factory(width, height)`.
    pub fn factory(width: i64, height: i64) -> Self {
        Self::new(width, height)
    }

    /// Returns the current width of the packer in pixels.
    pub const fn width(&self) -> i64 {
        self.size.width
    }

    /// Returns the current height of the packer in pixels.
    pub const fn height(&self) -> i64 {
        self.size.height
    }

    /// Returns the current dimensions of the packer in pixels.
    pub const fn size(&self) -> ISize {
        self.size
    }

    /// Returns `true` if no rectangles are currently allocated.
    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Attempts to pack a `width x height` rectangle, returning its top-left [`IPoint`]
    /// on success (`impeller::RectanglePacker::AddRect`).
    pub fn add_rect(&mut self, width: i64, height: i64) -> Option<IPoint> {
        self.allocate(width, height).map(|(_, pt)| pt)
    }

    /// Attempts to pack a `width x height` rectangle, returning both its [`AllocId`]
    /// (for later [`Self::deallocate`]) and its top-left [`IPoint`].
    pub fn allocate(&mut self, width: i64, height: i64) -> Option<(AllocId, IPoint)> {
        if width <= 0 || height <= 0 || width > self.size.width || height > self.size.height {
            return None;
        }
        let req_w = i32::try_from(width).ok()?;
        let req_h = i32::try_from(height).ok()?;
        if req_w > i32::from(u16::MAX) || req_h > i32::from(u16::MAX) {
            return None;
        }
        let req_size = etagere::size2(req_w, req_h);

        for (region_idx, (origin, allocator)) in self.regions.iter_mut().enumerate() {
            if let Some(alloc) = allocator.allocate(req_size) {
                let outer_id = AllocId::deserialize(self.next_alloc_id);
                self.next_alloc_id = self.next_alloc_id.wrapping_add(1);
                if self.next_alloc_id == 0 {
                    self.next_alloc_id = 1;
                }
                let area = width.saturating_mul(height);
                self.allocations
                    .insert(outer_id, (region_idx, alloc.id, area));
                self.used_area = self.used_area.saturating_add(area);
                let pt = IPoint::new(
                    origin.x + i64::from(alloc.rectangle.min.x),
                    origin.y + i64::from(alloc.rectangle.min.y),
                );
                return Some((outer_id, pt));
            }
        }
        None
    }

    /// Deallocates a previously allocated rectangle by its [`AllocId`], reclaiming shelf space.
    pub fn deallocate(&mut self, id: AllocId) -> bool {
        let Some((region_idx, inner_id, area)) = self.allocations.remove(&id) else {
            return false;
        };
        if let Some((_, allocator)) = self.regions.get_mut(region_idx) {
            allocator.deallocate(inner_id);
        }
        self.used_area = (self.used_area - area).max(0);
        true
    }

    /// Dynamically grows the packer dimensions to `new_size` while preserving all existing
    /// packed rectangle coordinates and [`AllocId`] handles.
    pub fn grow(&mut self, new_size: ISize) -> bool {
        if new_size.width < self.size.width
            || new_size.height < self.size.height
            || new_size.width <= 0
            || new_size.height <= 0
        {
            return false;
        }
        if new_size == self.size {
            return true;
        }

        // Verify the grown bounds fit within etagere's u16/i32 limits.
        if i32::try_from(new_size.width)
            .ok()
            .and_then(|w| {
                i32::try_from(new_size.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h).map(|_| (w, h)))
            })
            .is_none_or(|(w, h)| w > i32::from(u16::MAX) || h > i32::from(u16::MAX))
        {
            return false;
        }

        if self.allocations.is_empty() {
            self.regions.clear();
            if let Some(alloc) = make_atlas_allocator(new_size.width, new_size.height) {
                self.regions.push((IPoint::ZERO, alloc));
            }
            self.size = new_size;
            return true;
        }

        let old_w = self.size.width;
        let old_h = self.size.height;
        let right_w = new_size.width - old_w;
        let bottom_h = new_size.height - old_h;

        if right_w > 0 && old_h > 0 {
            if let Some(right_alloc) = make_atlas_allocator(right_w, old_h) {
                self.regions.push((IPoint::new(old_w, 0), right_alloc));
            }
        }
        if bottom_h > 0 {
            if let Some(bottom_alloc) = make_atlas_allocator(new_size.width, bottom_h) {
                self.regions.push((IPoint::new(0, old_h), bottom_alloc));
            }
        }

        self.size = new_size;
        true
    }

    /// Returns the ratio `[0.0, 1.0]` of allocated rectangle area to total packer area.
    pub fn percent_full(&self) -> Scalar {
        let total = (self.size.width as f64) * (self.size.height as f64);
        if total <= 0.0 {
            0.0
        } else {
            ((self.used_area as f64) / total) as Scalar
        }
    }

    /// Clears all allocations and resets the shelf state while maintaining current dimensions.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.used_area = 0;
        self.regions.clear();
        if let Some(alloc) = make_atlas_allocator(self.size.width, self.size.height) {
            self.regions.push((IPoint::ZERO, alloc));
        }
    }
}
