"""Read image tool.

Loads an image file from disk, resizes it, and attaches it to the next
prompt so the model can see it. Uses the same resize pipeline as the
screenshot tool to keep token usage reasonable.
"""

import os
from pathlib import Path

from agent.output import get_output_or_ipc
from agent.tools import tool
from agent.utils import resize_image


@tool(
    name="read_image",
    description=(
        "Load an image file from disk and attach it to the next prompt.\n"
        "Use this when the user asks you to look at an image on their\n"
        "filesystem, or when visual context is needed from a specific file.\n"
        "Takes a 'path' argument (absolute or relative path to the image).\n"
        "The image will be sent with your next message — call this tool,\n"
        "then ask the user for their question about the image."
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
    """Load an image file and attach it to the next turn."""
    path = args.get("path", "")
    output = get_output_or_ipc()

    if not path:
        output.err("Path not given :/")
        return {"error": "No file path provided"}

    # Expand ~ and resolve relative paths
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        output.err(f"File not found: {path}")
        return {"error": f"The file does not exist: {path}"}

    if not os.path.isfile(path):
        output.err(f"Not a file: {path}")
        return {"error": f"The path exists but is not a file: {path}"}

    try:
        raw_bytes = Path(path).read_bytes()
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
        "path": path,
    }
