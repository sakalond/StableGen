"""Preview gallery overlay for generated image selection."""

import math
import bpy  # pylint: disable=import-error
import blf  # pylint: disable=import-error
import gpu  # pylint: disable=import-error
from gpu_extras.batch import batch_for_shader  # pylint: disable=import-error
import numpy as np
from PIL import Image
import io

def _draw_rect_2d(x1, y1, x2, y2, color):
    """Draw a filled rectangle in 2D screen-space."""
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": ((x1, y1), (x2, y1), (x2, y2), (x1, y2))},
        indices=((0, 1, 2), (0, 2, 3)),
    )
    shader.bind()
    shader.uniform_float("color", color)
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _draw_texture_2d(texture, x1, y1, x2, y2):
    """Draw a textured rectangle in 2D screen-space."""
    shader = gpu.shader.from_builtin('IMAGE')
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
         "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))},
        indices=((0, 1, 2), (0, 2, 3)),
    )
    shader.bind()
    shader.uniform_sampler("image", texture)
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')


class _PreviewGalleryOverlay:
    """Viewport overlay that shows N generated images and lets the user pick one.

    Lifecycle:
    1. ``__init__`` — creates bpy.data.images + GPU textures, registers draw handler.
    2. ``handle_mouse_move`` / ``handle_click`` — called from Trellis2Generate.modal().
    3. ``update_images`` — called when "Generate More" produces a new batch.
    4. ``cleanup`` — removes draw handler and temp images.
    """

    def __init__(self, pil_images, seeds):
        """*pil_images*: list[PIL.Image]  *seeds*: list[int]"""
        self._pil_images = list(pil_images)
        self._seeds = list(seeds)
        self._n = len(pil_images)
        self._hover_idx = -1
        self._more_hover = False
        self._cancel_hover = False
        self._selected_idx = -1
        self.action = None  # 'select' | 'more' | 'cancel'
        self._cols = max(1, math.ceil(math.sqrt(self._n)))
        self._rows = max(1, math.ceil(self._n / self._cols))
        self._cell_rects: list[tuple] = []
        self._more_rect: tuple | None = None
        self._cancel_rect: tuple | None = None
        self._bpy_images: list = []
        self._textures: list = []
        self._setup_textures()
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (), 'WINDOW', 'POST_PIXEL')

    # ── texture management ──

    def _setup_textures(self):
        self._clear_textures()
        for i, pil_img in enumerate(self._pil_images):
            name = f"_sg_gallery_{i}"
            if name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])
            rgba = pil_img.convert('RGBA')
            w, h = rgba.size
            bpy_img = bpy.data.images.new(name, w, h, alpha=True)
            # Mark as Non-Color so gpu.texture.from_image() stores raw
            # sRGB values — the viewport's own output transform will
            # apply the single correct sRGB curve on display.
            bpy_img.colorspace_settings.name = 'Non-Color'
            # Blender images are stored bottom-to-top; PIL is top-to-bottom
            flipped = rgba.transpose(Image.FLIP_TOP_BOTTOM)
            flat = np.array(flipped, dtype=np.float32).ravel() / 255.0
            bpy_img.pixels.foreach_set(flat)
            bpy_img.pack()
            self._bpy_images.append(bpy_img)
            self._textures.append(gpu.texture.from_image(bpy_img))

    def _clear_textures(self):
        self._textures.clear()
        for img in self._bpy_images:
            if img.name in bpy.data.images:
                bpy.data.images.remove(img)
        self._bpy_images.clear()

    def update_images(self, pil_images, seeds):
        """Replace the gallery with a new batch."""
        self._pil_images = list(pil_images)
        self._seeds = list(seeds)
        self._n = len(pil_images)
        self._cols = max(1, math.ceil(math.sqrt(self._n)))
        self._rows = max(1, math.ceil(self._n / self._cols))
        self._hover_idx = -1
        self._more_hover = False
        self._cancel_hover = False
        self._selected_idx = -1
        self.action = None
        self._setup_textures()

    def cleanup(self):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        self._clear_textures()

    # ── interaction ──

    def handle_mouse_move(self, mx, my):
        """Update hover state. Returns True if display should redraw."""
        old = (self._hover_idx, self._more_hover, self._cancel_hover)
        self._hover_idx = -1
        self._more_hover = False
        self._cancel_hover = False
        for i, rect in enumerate(self._cell_rects):
            x1, y1, x2, y2 = rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                self._hover_idx = i
                break
        if self._more_rect:
            x1, y1, x2, y2 = self._more_rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                self._more_hover = True
        if self._cancel_rect:
            x1, y1, x2, y2 = self._cancel_rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                self._cancel_hover = True
        return old != (self._hover_idx, self._more_hover, self._cancel_hover)

    def handle_click(self, mx, my):
        """Returns 'select', 'more', 'cancel', or None."""
        if self._more_hover:
            return 'more'
        if self._cancel_hover:
            return 'cancel'
        if self._hover_idx >= 0:
            self._selected_idx = self._hover_idx
            return 'select'
        return None

    @property
    def selected_seed(self):
        if 0 <= self._selected_idx < len(self._seeds):
            return self._seeds[self._selected_idx]
        return None

    @property
    def selected_image_bytes(self):
        """Return PNG bytes of the selected image (for TRELLIS workflow)."""
        if 0 <= self._selected_idx < len(self._pil_images):
            buf = io.BytesIO()
            self._pil_images[self._selected_idx].save(buf, format='PNG')
            return buf.getvalue()
        return None

    # ── drawing ──

    def _draw(self):
        """POST_PIXEL callback — draws the full-screen gallery overlay."""
        context = bpy.context
        region = context.region
        if not region:
            return
        vw, vh = region.width, region.height
        if vw < 100 or vh < 100:
            return

        # Detect sidebar (N-panel) width so the grid avoids being covered.
        sidebar_w = 0
        # Headers drawn over the viewport (region overlap) would cover the title.
        top_inset = 0
        area = context.area
        if area:
            top = region.y + region.height
            for r in area.regions:
                if r.type == 'UI' and r.width > 1 and not sidebar_w:
                    sidebar_w = r.width
                if (r.type in {'HEADER', 'TOOL_HEADER'} and r.height > 1
                        and region.y < r.y < top):
                    top_inset = max(top_inset, top - r.y)

        # Layout constants
        pad = 20
        btn_h = 40
        title_h = 30
        bottom_area = btn_h + pad * 2
        usable_w = vw - sidebar_w  # exclude area behind the N-panel

        # --- dark backdrop ---
        _draw_rect_2d(0, 0, vw, vh, (0.08, 0.08, 0.08, 0.90))

        # --- title ---
        blf.size(0, 20)
        title = "Select an image — or generate more"
        tw, th = blf.dimensions(0, title)
        blf.position(0, (usable_w - tw) / 2, vh - top_inset - pad - th, 0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, title)

        grid_top = vh - top_inset - pad - title_h - pad
        available_w = usable_w - pad * 2
        available_h = grid_top - bottom_area - pad

        cell_w = max(1, available_w // self._cols)
        cell_h = max(1, available_h // self._rows)

        # Maintain image aspect ratio
        if self._pil_images:
            img_aspect = self._pil_images[0].width / max(1, self._pil_images[0].height)
            desired_h = cell_w / img_aspect
            if desired_h > cell_h:
                cell_w = int(cell_h * img_aspect)
            else:
                cell_h = int(desired_h)

        inner = 6
        total_grid_w = self._cols * cell_w
        offset_x = (usable_w - total_grid_w) / 2.0

        self._cell_rects.clear()

        for i in range(self._n):
            row = i // self._cols
            col = i % self._cols
            x1 = offset_x + col * cell_w + inner
            y2 = grid_top - row * cell_h - inner
            x2 = x1 + cell_w - inner * 2
            y1 = y2 - cell_h + inner * 2
            self._cell_rects.append((x1, y1, x2, y2))

            # Hover highlight ring
            if i == self._hover_idx:
                _draw_rect_2d(x1 - 3, y1 - 3, x2 + 3, y2 + 3, (0.35, 0.65, 1.0, 0.85))

            # Image texture
            if i < len(self._textures):
                _draw_texture_2d(self._textures[i], x1, y1, x2, y2)

            # Number badge
            blf.size(0, 15)
            label = str(i + 1)
            lw, lh = blf.dimensions(0, label)
            _draw_rect_2d(x1, y2 - lh - 8, x1 + lw + 12, y2, (0.0, 0.0, 0.0, 0.70))
            blf.position(0, x1 + 6, y2 - lh - 3, 0)
            blf.color(0, 1.0, 1.0, 1.0, 1.0)
            blf.draw(0, label)

            # Seed label
            blf.size(0, 11)
            seed_txt = f"Seed: {self._seeds[i]}"
            sw, sh = blf.dimensions(0, seed_txt)
            _draw_rect_2d(x1, y1, x1 + sw + 10, y1 + sh + 6, (0, 0, 0, 0.6))
            blf.position(0, x1 + 5, y1 + 3, 0)
            blf.color(0, 0.7, 0.7, 0.7, 1.0)
            blf.draw(0, seed_txt)

        # --- buttons ---
        btn_w = 160
        btn_y1 = pad
        btn_y2 = pad + btn_h

        # "Generate More"
        center_x = usable_w / 2.0
        mx1 = center_x - btn_w - 10
        mx2 = center_x - 10
        c = (0.25, 0.55, 0.85, 0.95) if self._more_hover else (0.20, 0.35, 0.55, 0.85)
        _draw_rect_2d(mx1, btn_y1, mx2, btn_y2, c)
        self._more_rect = (mx1, btn_y1, mx2, btn_y2)
        blf.size(0, 15)
        mt = "Generate More"
        mtw, mth = blf.dimensions(0, mt)
        blf.position(0, (mx1 + mx2 - mtw) / 2, (btn_y1 + btn_y2 - mth) / 2, 0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, mt)

        # "Cancel"
        cx1 = center_x + 10
        cx2 = center_x + btn_w + 10
        c = (0.75, 0.25, 0.25, 0.95) if self._cancel_hover else (0.45, 0.20, 0.20, 0.85)
        _draw_rect_2d(cx1, btn_y1, cx2, btn_y2, c)
        self._cancel_rect = (cx1, btn_y1, cx2, btn_y2)
        blf.size(0, 15)
        ct = "Cancel"
        ctw, cth = blf.dimensions(0, ct)
        blf.position(0, (cx1 + cx2 - ctw) / 2, (btn_y1 + btn_y2 - cth) / 2, 0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, ct)
