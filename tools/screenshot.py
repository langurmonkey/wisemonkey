"""Screenshot tool.

Captures the current screen and returns it as a base64-encoded JPEG image.
Downscaled and compressed to reduce token usage. User confirmation is
required for privacy/security.
"""

import base64
from io import BytesIO
from PIL import Image, ImageGrab
from rich.prompt import Confirm as RichConfirm

from agent.tools import tool
from agent.console import print, newline


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
    newline()
    print("\U0001f4f8 [warn]Screenshot requested[/warn]")
    print("  [weak]The agent wants to capture your current screen.[/weak]")
    newline()

    confirmed = RichConfirm.ask("[bold]Allow screenshot?[/bold]", default=False)

    if not confirmed:
        print("  [err]\u2717[/err] Cancelled by user")
        return {
            "error": (
                "Screenshot was cancelled by the user. "
                "Explain to the user why you needed the screenshot "
                "and ask if they'd like to describe what's on screen instead."
            ),
            "user_cancelled": True,
        }

    print("  [ok]\u2713[/ok] Capturing screenshot...")

    # Capture
    img = ImageGrab.grab()

    # Convert RGBA to RGB for JPEG (screenshots typically don't need alpha)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Downscale: max dimension 1024px, maintain aspect ratio
    max_dim = 1024
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim / w, max_dim / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h))

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70, optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"image_base64": b64, "mime_type": "image/jpeg"}
