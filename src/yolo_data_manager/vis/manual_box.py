"""Interactive, non-persistent single-image box drawing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from yolo_data_manager.core.geometry import xywhn_to_xyxy, xyxy_to_xywhn
from yolo_data_manager.core.models import YoloDataset, YoloImage
from yolo_data_manager.io.layout import infer_label_path_from_image
from yolo_data_manager.io.loader import parse_label_file


_PREVIEW_COLORS = (
    (42, 220, 120),
    (255, 190, 40),
    (80, 170, 255),
    (220, 100, 255),
    (255, 100, 80),
)


@dataclass(frozen=True)
class ManualBoxResult:
    """Coordinates of the temporary box drawn by the user."""

    image_path: Path
    label_path: Path | None
    image_size: tuple[int, int]
    pixel_xyxy: tuple[int, int, int, int]
    yolo_xywhn: tuple[float, float, float, float]
    class_id: int | None = None
    precision: int = 6

    @property
    def yolo_line(self) -> str | None:
        if self.class_id is None:
            return None
        return format_yolo_line(self.class_id, self.yolo_xywhn, precision=self.precision)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for terminal/manual use."""

        width, height = self.image_size
        return {
            "image": str(self.image_path),
            "label": str(self.label_path) if self.label_path is not None else None,
            "image_size": {"width": width, "height": height},
            "pixel_xyxy": list(self.pixel_xyxy),
            "yolo_xywhn": list(self.yolo_xywhn),
            "class_id": self.class_id,
            "yolo_line": self.yolo_line,
        }


def format_yolo_line(
    class_id: int,
    yolo_xywhn: Sequence[float],
    *,
    precision: int = 6,
) -> str:
    """Format a class id and normalized ``cx cy width height`` as a YOLO row."""

    if len(yolo_xywhn) != 4:
        raise ValueError("YOLO detection coordinates must contain four values")
    if precision < 0:
        raise ValueError("precision must be non-negative")
    values = " ".join(f"{float(value):.{precision}f}" for value in yolo_xywhn)
    return f"{int(class_id)} {values}"


def find_dataset_image(dataset: YoloDataset, image_name: str | Path) -> YoloImage:
    """Find one dataset image by absolute path, relative path, name, or stem."""

    requested_text = str(image_name)
    requested = Path(image_name)
    exact: list[YoloImage] = []

    for image in dataset.images:
        if requested.is_absolute() and _same_path(image.path, requested):
            exact.append(image)
            continue
        if not requested.is_absolute():
            if _same_path(image.path, dataset.root / requested):
                exact.append(image)
                continue
            try:
                relative = image.path.resolve().relative_to(dataset.root.resolve())
            except ValueError:
                relative = None
            if relative is not None and relative == requested:
                exact.append(image)
                continue
        if image.file_name == requested_text:
            exact.append(image)

    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(_ambiguous_image_message(requested_text, exact))

    by_name = [image for image in dataset.images if image.path.name == requested.name]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise ValueError(_ambiguous_image_message(requested_text, by_name))

    if requested.suffix == "":
        by_stem = [image for image in dataset.images if image.stem == requested.name]
        if len(by_stem) == 1:
            return by_stem[0]
        if len(by_stem) > 1:
            raise ValueError(_ambiguous_image_message(requested_text, by_stem))

    raise ValueError(
        f"image not found in dataset: {requested_text!r}; "
        "pass an image filename or a path relative to the dataset root"
    )


def draw_manual_box(
    image_path: str | Path,
    *,
    label_path: str | Path | None = None,
    class_id: int | None = None,
    class_names: Sequence[str] | None = None,
    max_width: int = 1400,
    max_height: int = 900,
    min_pixels: int = 2,
    precision: int = 6,
    title: str | None = None,
) -> ManualBoxResult | None:
    """Show one image and let the user draw one temporary detection box.

    Existing YOLO annotations are drawn as a preview. The returned box is never
    written to ``label_path``. ``None`` means that the window was cancelled or
    closed without a completed box.
    """

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("max_width and max_height must be positive")
    if min_pixels <= 0:
        raise ValueError("min_pixels must be positive")
    if precision < 0:
        raise ValueError("precision must be non-negative")

    resolved_label_path = (
        Path(label_path) if label_path is not None else infer_label_path_from_image(image_path)
    )
    annotations = parse_label_file(resolved_label_path)
    with Image.open(image_path) as source:
        source_image = source.convert("RGB")

    width, height = source_image.size
    scale = min(max_width / width, max_height / height, 1.0)
    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))
    preview = source_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
    _draw_existing_annotations(
        preview,
        annotations,
        original_size=(width, height),
        scale=scale,
        class_names=class_names,
    )

    pixel_box = _run_box_window(
        preview,
        original_size=(width, height),
        display_size=(display_width, display_height),
        scale=scale,
        min_pixels=min_pixels,
        title=title or f"Manual box - {image_path.name}",
    )
    if pixel_box is None:
        return None

    yolo_box = xyxy_to_xywhn(*pixel_box, width, height)
    return ManualBoxResult(
        image_path=image_path,
        label_path=resolved_label_path,
        image_size=(width, height),
        pixel_xyxy=pixel_box,
        yolo_xywhn=yolo_box,
        class_id=class_id,
        precision=precision,
    )


def _draw_existing_annotations(
    preview: Image.Image,
    annotations: Sequence[Any],
    *,
    original_size: tuple[int, int],
    scale: float,
    class_names: Sequence[str] | None,
) -> None:
    width, height = original_size
    draw = ImageDraw.Draw(preview, "RGBA")
    for index, annotation in enumerate(annotations, start=1):
        box = annotation.geometry_box()
        if box is None:
            continue
        xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
        left = max(0.0, min(preview.width, xyxy.x1 * scale))
        top = max(0.0, min(preview.height, xyxy.y1 * scale))
        right = max(0.0, min(preview.width, xyxy.x2 * scale))
        bottom = max(0.0, min(preview.height, xyxy.y2 * scale))
        color = _PREVIEW_COLORS[(annotation.class_id or 0) % len(_PREVIEW_COLORS)]
        draw.rectangle((left, top, right, bottom), outline=(*color, 255), width=2)

        class_name = str(annotation.class_id)
        if class_names is not None and 0 <= annotation.class_id < len(class_names):
            class_name = f"{annotation.class_id}:{class_names[annotation.class_id]}"
        text = f"{annotation.line_no or index} {class_name}"
        text_box = draw.textbbox((left, top), text)
        pad = 2
        draw.rectangle(
            (
                text_box[0] - pad,
                text_box[1] - pad,
                text_box[2] + pad,
                text_box[3] + pad,
            ),
            fill=(*color, 220),
        )
        draw.text((left, top), text, fill=(0, 0, 0, 255))


def _run_box_window(
    preview: Image.Image,
    *,
    original_size: tuple[int, int],
    display_size: tuple[int, int],
    scale: float,
    min_pixels: int,
    title: str,
) -> tuple[int, int, int, int] | None:
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError as exc:  # pragma: no cover - depends on the Python install
        raise RuntimeError(
            "Tkinter is required for manual box drawing; install/enable Tk on this machine"
        ) from exc

    try:
        window = _ManualBoxWindow(
            tk,
            ImageTk,
            preview,
            original_size=original_size,
            display_size=display_size,
            scale=scale,
            min_pixels=min_pixels,
            title=title,
        )
        window.root.mainloop()
        return window.result
    except tk.TclError as exc:  # pragma: no cover - depends on the display server
        raise RuntimeError(
            "Tkinter could not open a display; run with a desktop display or X11 forwarding"
        ) from exc


class _ManualBoxWindow:
    def __init__(
        self,
        tk: Any,
        image_tk: Any,
        preview: Image.Image,
        *,
        original_size: tuple[int, int],
        display_size: tuple[int, int],
        scale: float,
        min_pixels: int,
        title: str,
    ) -> None:
        self.tk = tk
        self.original_width, self.original_height = original_size
        self.display_width, self.display_height = display_size
        self.scale = scale
        self.min_pixels = min_pixels
        self.start: tuple[float, float] | None = None
        self.rectangle: int | None = None
        self.result: tuple[int, int, int, int] | None = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.status = tk.StringVar(
            value="拖拽绘制一个 box；Enter 完成并输出，R 清除重画，Esc 取消"
        )
        tk.Label(self.root, textvariable=self.status, anchor="w", justify="left").pack(
            fill="x", padx=8, pady=(8, 4)
        )
        self.canvas = tk.Canvas(
            self.root,
            width=self.display_width,
            height=self.display_height,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack(padx=8, pady=4)
        self.photo = image_tk.PhotoImage(image=preview)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=(4, 8))
        tk.Button(controls, text="完成并输出", command=self.finish).pack(side="left")
        tk.Button(controls, text="清除", command=self.clear).pack(side="left", padx=6)
        tk.Button(controls, text="取消", command=self.cancel).pack(side="right")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Return>", self._finish_event)
        self.root.bind("<Escape>", self._cancel_event)
        self.root.bind("r", self._clear_event)
        self.root.bind("R", self._clear_event)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def on_press(self, event: Any) -> None:
        self.start = self._clamp_display_point(event.x, event.y)
        self.result = None
        if self.rectangle is not None:
            self.canvas.delete(self.rectangle)
        x, y = self.start
        self.rectangle = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#ff2020",
            width=2,
        )

    def on_drag(self, event: Any) -> None:
        if self.start is None or self.rectangle is None:
            return
        end = self._clamp_display_point(event.x, event.y)
        self.canvas.coords(self.rectangle, *self._display_xyxy(self.start, end))

    def on_release(self, event: Any) -> None:
        if self.start is None:
            return
        end = self._clamp_display_point(event.x, event.y)
        display_xyxy = self._display_xyxy(self.start, end)
        left, top, right, bottom = self._to_pixel_xyxy(display_xyxy)
        if right - left < self.min_pixels or bottom - top < self.min_pixels:
            self.status.set(f"box 太小，至少需要 {self.min_pixels} 像素；请重新拖拽")
            self.result = None
            return
        self.result = (left, top, right, bottom)
        if self.rectangle is not None:
            self.canvas.coords(
                self.rectangle,
                left * self.scale,
                top * self.scale,
                right * self.scale,
                bottom * self.scale,
            )
        self.status.set(
            "pixel xyxy={}；按 Enter 完成，或按 R 清除重画".format(self.result)
        )

    def clear(self) -> None:
        self.start = None
        self.result = None
        if self.rectangle is not None:
            self.canvas.delete(self.rectangle)
            self.rectangle = None
        self.status.set("已清除；请重新拖拽绘制一个 box")

    def finish(self) -> None:
        self.root.destroy()

    def cancel(self) -> None:
        self.result = None
        self.root.destroy()

    def _finish_event(self, _event: Any) -> str:
        self.finish()
        return "break"

    def _cancel_event(self, _event: Any) -> str:
        self.cancel()
        return "break"

    def _clear_event(self, _event: Any) -> str:
        self.clear()
        return "break"

    def _clamp_display_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            max(0.0, min(float(self.display_width), float(x))),
            max(0.0, min(float(self.display_height), float(y))),
        )

    @staticmethod
    def _display_xyxy(
        start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float, float, float]:
        return min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1])

    def _to_pixel_xyxy(
        self, display_xyxy: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = display_xyxy
        pixel_left = max(0, min(self.original_width, round(left / self.scale)))
        pixel_top = max(0, min(self.original_height, round(top / self.scale)))
        pixel_right = max(0, min(self.original_width, round(right / self.scale)))
        pixel_bottom = max(0, min(self.original_height, round(bottom / self.scale)))
        return pixel_left, pixel_top, pixel_right, pixel_bottom


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _ambiguous_image_message(requested: str, matches: Sequence[YoloImage]) -> str:
    choices = ", ".join(str(image.path) for image in matches[:5])
    suffix = "..." if len(matches) > 5 else ""
    return f"image name is ambiguous: {requested!r}; matches: {choices}{suffix}"
