"""Interactive, non-persistent single-image box drawing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

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
    show_existing: bool = True,
    mask_outside: bool = False,
    title: str | None = None,
) -> ManualBoxResult | None:
    """Show one image and let the user draw one temporary detection box.

    Existing YOLO annotations are shown as independent Matplotlib artists and
    can be toggled with the ``L`` key. The returned box is never written to
    ``label_path``. ``None`` means that the window was cancelled or closed
    without a completed box.
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

    pixel_box = _run_box_window(
        preview,
        annotations=annotations,
        class_names=class_names,
        show_existing=show_existing,
        original_size=(width, height),
        display_size=(display_width, display_height),
        scale=scale,
        min_pixels=min_pixels,
        mask_outside=mask_outside,
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


def _add_existing_annotation_artists(
    axes: Any,
    annotations: Sequence[Any],
    *,
    original_size: tuple[int, int],
    scale: float,
    class_names: Sequence[str] | None,
    display_size: tuple[int, int],
) -> list[Any]:
    from matplotlib.patches import Rectangle

    width, height = original_size
    display_width, display_height = display_size
    artists: list[Any] = []
    for index, annotation in enumerate(annotations, start=1):
        box = annotation.geometry_box()
        if box is None:
            continue
        xyxy = xywhn_to_xyxy(box.as_tuple(), width, height)
        left = max(0.0, min(display_width, xyxy.x1 * scale))
        top = max(0.0, min(display_height, xyxy.y1 * scale))
        right = max(0.0, min(display_width, xyxy.x2 * scale))
        bottom = max(0.0, min(display_height, xyxy.y2 * scale))
        color = _PREVIEW_COLORS[(annotation.class_id or 0) % len(_PREVIEW_COLORS)]
        color_float = tuple(channel / 255.0 for channel in color)
        rectangle = Rectangle(
            (left, top),
            right - left,
            bottom - top,
            fill=False,
            edgecolor=color_float,
            linewidth=1.5,
            zorder=3,
        )
        axes.add_patch(rectangle)
        artists.append(rectangle)

        class_name = str(annotation.class_id)
        if class_names is not None and 0 <= annotation.class_id < len(class_names):
            class_name = f"{annotation.class_id}:{class_names[annotation.class_id]}"
        text = f"{annotation.line_no or index} {class_name}"
        label = axes.text(
            left,
            top,
            text,
            color="black",
            fontsize=8,
            va="top",
            ha="left",
            zorder=4,
            bbox={
                "facecolor": color_float,
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 2,
            },
        )
        artists.append(label)
    return artists


def _run_box_window(
    preview: Image.Image,
    *,
    annotations: Sequence[Any],
    class_names: Sequence[str] | None,
    show_existing: bool,
    original_size: tuple[int, int],
    display_size: tuple[int, int],
    scale: float,
    min_pixels: int,
    mask_outside: bool,
    title: str,
) -> tuple[int, int, int, int] | None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector
    except ImportError as exc:  # pragma: no cover - depends on the Python install
        raise RuntimeError(
            "Matplotlib with an interactive GUI backend is required for manual box drawing"
        ) from exc

    backend = str(matplotlib.get_backend()).lower()
    if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"} or backend.startswith(
        "module://matplotlib_inline"
    ):
        raise RuntimeError(
            f"Matplotlib backend {backend!r} is non-interactive; "
            "use a GUI backend and a desktop display/X11 forwarding"
        )

    state: dict[str, tuple[int, int, int, int] | None] = {"result": None}
    figure = None
    try:
        figure, axes = plt.subplots(
            figsize=(display_size[0] / 100.0, display_size[1] / 100.0),
            dpi=100,
        )
        axes.imshow(
            preview,
            extent=(0, display_size[0], display_size[1], 0),
            origin="upper",
        )
        axes.set_xlim(0, display_size[0])
        axes.set_ylim(display_size[1], 0)
        axes.set_aspect("equal", adjustable="box")
        axes.axis("off")
        existing_artists = _add_existing_annotation_artists(
            axes,
            annotations,
            original_size=original_size,
            scale=scale,
            class_names=class_names,
            display_size=display_size,
        )
        existing_state = {"visible": bool(show_existing)}
        for artist in existing_artists:
            artist.set_visible(existing_state["visible"])
        mask_artists = _create_outside_mask_artists(
            axes,
            display_size=display_size,
        )
        figure.suptitle(
            f"{title}\nDrag box; wheel/+/- zoom; 0 reset; L labels; Enter save; R redraw; Esc cancel",
            fontsize=10,
        )
        status = figure.text(
            0.01,
            0.01,
            _status_text(existing_state["visible"], mask_outside),
            ha="left",
            va="bottom",
            fontsize=9,
            color="#cc0000",
        )

        def reset_view() -> None:
            axes.set_xlim(0, display_size[0])
            axes.set_ylim(display_size[1], 0)
            figure.canvas.draw_idle()

        def zoom_at(x: float | None, y: float | None, factor: float) -> None:
            _zoom_axes(
                axes,
                factor=factor,
                center=(x, y) if x is not None and y is not None else None,
                bounds=display_size,
            )
            figure.canvas.draw_idle()

        def on_select(eclick: Any, erelease: Any) -> None:
            if eclick.xdata is None or eclick.ydata is None:
                return
            if erelease.xdata is None or erelease.ydata is None:
                return
            display_xyxy = _display_xyxy(
                eclick.xdata,
                eclick.ydata,
                erelease.xdata,
                erelease.ydata,
            )
            pixel_xyxy = _display_to_pixel_xyxy(
                display_xyxy,
                original_size=original_size,
                display_size=display_size,
                scale=scale,
            )
            left, top, right, bottom = pixel_xyxy
            if right - left < min_pixels or bottom - top < min_pixels:
                state["result"] = None
                _set_outside_mask(mask_artists, None, display_size=display_size)
                status.set_text(f"box too small; minimum is {min_pixels} pixels")
            else:
                state["result"] = pixel_xyxy
                if mask_outside:
                    _set_outside_mask(
                        mask_artists,
                        display_xyxy,
                        display_size=display_size,
                    )
                status.set_text(f"pixel xyxy: {pixel_xyxy}; press Enter to save")
            figure.canvas.draw_idle()

        def on_key(event: Any) -> None:
            if event.key in {"enter", "return"}:
                plt.close(figure)
            elif event.key in {"escape", "q"}:
                state["result"] = None
                plt.close(figure)
            elif event.key in {"r", "c"}:
                state["result"] = None
                _set_outside_mask(mask_artists, None, display_size=display_size)
                status.set_text(_status_text(existing_state["visible"], mask_outside))
                figure.canvas.draw_idle()
            elif event.key in {"l", "L"}:
                existing_state["visible"] = not existing_state["visible"]
                for artist in existing_artists:
                    artist.set_visible(existing_state["visible"])
                status.set_text(_status_text(existing_state["visible"], mask_outside))
                figure.canvas.draw_idle()
            elif event.key in {"+", "="}:
                zoom_at(None, None, 0.8)
            elif event.key in {"-", "_"}:
                zoom_at(None, None, 1.25)
            elif event.key == "0":
                reset_view()

        def on_scroll(event: Any) -> None:
            if event.inaxes is not axes:
                return
            factor = 0.8 if event.button == "up" else 1.25
            zoom_at(event.xdata, event.ydata, factor)

        selector = RectangleSelector(
            axes,
            on_select,
            useblit=False,
            button=[1],
            minspanx=max(1.0, min_pixels * scale),
            minspany=max(1.0, min_pixels * scale),
            spancoords="pixels",
            interactive=False,
            props={"facecolor": "none", "edgecolor": "red", "linewidth": 2},
        )
        figure.canvas.mpl_connect("key_press_event", on_key)
        figure.canvas.mpl_connect("scroll_event", on_scroll)
        # Keep the selector alive for the whole blocking show call.
        figure._ydm_manual_box_selector = selector  # type: ignore[attr-defined]
        plt.show(block=True)
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - backend/display dependent
        raise RuntimeError(
            "Matplotlib could not open an interactive display; "
            "use a GUI backend with desktop display/X11 forwarding"
        ) from exc
    finally:
        if figure is not None and plt.fignum_exists(figure.number):
            plt.close(figure)
    return state["result"]


def _create_outside_mask_artists(
    axes: Any,
    *,
    display_size: tuple[int, int],
) -> list[Any]:
    """Create four hidden patches that can mask everything outside a box."""

    from matplotlib.patches import Rectangle

    width, height = display_size
    artists: list[Any] = []
    for _ in range(4):
        artist = Rectangle(
            (0, 0),
            0,
            0,
            facecolor="black",
            edgecolor="none",
            linewidth=0,
            zorder=5,
            visible=False,
        )
        axes.add_patch(artist)
        artists.append(artist)
    return artists


def _set_outside_mask(
    artists: Sequence[Any],
    display_xyxy: tuple[float, float, float, float] | None,
    *,
    display_size: tuple[int, int],
) -> None:
    """Update or hide the four black patches around the selected box."""

    if len(artists) != 4:
        raise ValueError("outside mask requires exactly four artists")

    if display_xyxy is None:
        for artist in artists:
            artist.set_visible(False)
        return

    width, height = display_size
    left, top, right, bottom = display_xyxy
    left = max(0.0, min(float(width), left))
    top = max(0.0, min(float(height), top))
    right = max(0.0, min(float(width), right))
    bottom = max(0.0, min(float(height), bottom))
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))

    rectangles = (
        (0.0, 0.0, left, float(height)),
        (right, 0.0, float(width) - right, float(height)),
        (left, 0.0, right - left, top),
        (left, bottom, right - left, float(height) - bottom),
    )
    for artist, (x, y, rectangle_width, rectangle_height) in zip(artists, rectangles):
        artist.set_xy((x, y))
        artist.set_width(max(0.0, rectangle_width))
        artist.set_height(max(0.0, rectangle_height))
        artist.set_visible(rectangle_width > 0 and rectangle_height > 0)


def _status_text(show_existing: bool, mask_outside: bool = False) -> str:
    existing = "shown" if show_existing else "hidden"
    mask = " | outside mask: on" if mask_outside else ""
    return f"pixel xyxy: draw a box | labels: {existing} (L){mask} | wheel/+/- zoom | 0 reset"


def _zoom_axes(
    axes: Any,
    *,
    factor: float,
    center: tuple[float | None, float | None] | None,
    bounds: tuple[int, int],
) -> None:
    """Zoom an image axes while keeping its coordinates in display pixels."""

    if factor <= 0:
        raise ValueError("zoom factor must be positive")
    width, height = bounds
    current_x = axes.get_xlim()
    current_y = axes.get_ylim()
    x_low, x_high = sorted(current_x)
    y_low, y_high = sorted(current_y)
    center_x = (x_low + x_high) / 2.0
    center_y = (y_low + y_high) / 2.0
    if center is not None:
        if center[0] is not None:
            center_x = float(center[0])
        if center[1] is not None:
            center_y = float(center[1])

    x_low, x_high = _zoom_interval(center_x, (x_high - x_low) * factor, width)
    y_low, y_high = _zoom_interval(center_y, (y_high - y_low) * factor, height)
    axes.set_xlim(x_low, x_high)
    axes.set_ylim(y_high, y_low)


def _zoom_interval(center: float, span: float, limit: int) -> tuple[float, float]:
    span = min(float(limit), max(1.0, float(span)))
    low = center - span / 2.0
    high = center + span / 2.0
    if low < 0:
        high -= low
        low = 0.0
    if high > limit:
        low -= high - limit
        high = float(limit)
    return max(0.0, low), min(float(limit), high)


def _display_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float, float, float]:
    return (
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    )


def _display_to_pixel_xyxy(
    display_xyxy: tuple[float, float, float, float],
    *,
    original_size: tuple[int, int],
    display_size: tuple[int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    """Convert Matplotlib display coordinates to clamped image pixel bounds."""

    left, top, right, bottom = display_xyxy
    width, height = original_size
    display_width, display_height = display_size
    left = max(0.0, min(float(display_width), left))
    top = max(0.0, min(float(display_height), top))
    right = max(0.0, min(float(display_width), right))
    bottom = max(0.0, min(float(display_height), bottom))
    return (
        max(0, min(width, round(left / scale))),
        max(0, min(height, round(top / scale))),
        max(0, min(width, round(right / scale))),
        max(0, min(height, round(bottom / scale))),
    )


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _ambiguous_image_message(requested: str, matches: Sequence[YoloImage]) -> str:
    choices = ", ".join(str(image.path) for image in matches[:5])
    suffix = "..." if len(matches) > 5 else ""
    return f"image name is ambiguous: {requested!r}; matches: {choices}{suffix}"
