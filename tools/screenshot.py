"""Screenshot tool.

Captures the current screen and returns it as a base64-encoded JPEG image.
Downscaled and compressed to reduce token usage. User confirmation is
required for privacy/security.
"""

from io import BytesIO
from PIL import ImageGrab
from rich.prompt import Confirm as RichConfirm

from agent.tools import tool
from agent.console import print, newline, ok, err
from agent.utils import resize_image


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
    print("📷 [warn]Screenshot requested[/warn]")
    print("  [weak]The agent wants to capture your current screen.[/weak]")
    newline()

    confirmed = RichConfirm.ask("[bold]Allow screenshot?[/bold]", default=False)

    if not confirmed:
        err("Cancelled by user")
        return {
            "error": (
                "Screenshot was cancelled by the user. "
                "Explain to the user why you needed the screenshot "
                "and ask if they'd like to describe what's on screen instead."
            ),
            "user_cancelled": True,
        }

    ok("Capturing screenshot...")

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
