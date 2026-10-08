"""Tools to read images.

- read_image: Loads an image file from disk, resizes it, and returns it as a multimodal
tool result so the model can see it in the current turn. Uses the same
resize pipeline as the screenshot tool to keep token usage reasonable.

- screenshot: Captures the current screen and returns it as a base64-encoded JPEG image.
Downscaled and compressed to reduce token usage. User confirmation is
required for privacy/security.
"""

from io import BytesIO
from pathlib import Path

from PIL import ImageGrab

from agent.output import get_output_or_ipc
from agent.tools import tool
from agent.utils import resize_image

# Refuse to slurp anything remotely huge before handing it to PIL.
MAX_IMAGE_BYTES = 50 * 1024 * 1024

@tool(
    name="read_image",
    description=(
        "Load an image file from disk and return it as an image result.\n"
        "Use this when the user asks you to look at an image on their\n"
        "filesystem, or when visual context is needed from a specific file.\n"
        "Takes a 'path' argument (absolute or relative path to the image).\n"
        "The image is attached to the current turn's tool result, so the\n"
        "model sees it immediately."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to the image (e.g., '/home/user/photo.jpg')",
            },
        },
        "required": ["path"],
    },
)
def read_image_handler(args):
    """Load an image file and return it as an image tool result."""
    path = args.get("path", "")
    output = get_output_or_ipc()

    if not path:
        output.err("Path not given :/")
        return {"error": "No file path provided"}

    # Expand ~ and resolve relative paths
    p = Path(path).expanduser().resolve()

    if not p.is_file():
        if p.is_dir():
            output.err(f"Not a file: {p}")
            return {"error": f"The path exists but is not a file: {p}"}
        output.err(f"File not found: {p}")
        return {"error": f"The file does not exist: {p}"}

    try:
        if p.stat().st_size > MAX_IMAGE_BYTES:
            output.err(f"Image file too large: {p}")
            return {
                "error": (
                    f"File is too large to load as an image: {p} "
                    f"({p.stat().st_size} bytes, limit {MAX_IMAGE_BYTES})"
                )
            }
        raw_bytes = p.read_bytes()
        result = resize_image(raw_bytes)
    except Exception as e:
        output.err(f"Failed to load image: {e}")
        return {"error": f"Could not process image: {e}"}

    # Return image_base64 + mime_type so core.py recognises it as an image
    # tool result and emits it immediately as a multimodal content block.
    # This lets the model see the image in the current turn.
    return {
        "image_base64": result["image_base64"],
        "mime_type": result["mime_type"],
        "path": str(p),
    }

@tool(
    name="screenshot",
    description="Capture a screenshot of the current screen and return it as an image. "
    "Use this when the user wants to see what's on the screen, or when visual context is needed.",
    parameters={
        "type": "object",
        "properties": {},
    },
)
def screenshot_handler(args):
    """Capture the current screen and return base64-encoded JPEG.

    Always prompts the user for confirmation before capturing,
    as screenshots contain sensitive information.
    """

    # User confirmation
    output = get_output_or_ipc()
    output.newline()
    output.print("📷 [warn]Screenshot requested[/warn]", indent=2)
    output.print("[weak]The agent wants to capture your current screen.[/weak]", indent=2)
    output.newline()

    confirmed = output.ask_confirm("[bold]Allow screenshot?[/bold]", default=False)

    if not confirmed:
        output.err("Cancelled by user", indent=2)
        return {
            "error": (
                "Screenshot was cancelled by the user. "
                "Explain to the user why you needed the screenshot "
                "and ask if they'd like to describe what's on screen instead."
            ),
            "user_cancelled": True,
        }

    output.ok("Capturing screenshot...", indent=2)

    # Capture
    img = ImageGrab.grab()

    # Convert RGBA to RGB for JPEG (screenshots typically don't need alpha)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Save to bytes then reuse the shared resize utility
    buf = BytesIO()
    img.save(buf, format="PNG")  # lossless intermediate
    raw_bytes = buf.getvalue()

    return resize_image(raw_bytes)
