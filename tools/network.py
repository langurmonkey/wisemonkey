"""Network access tool.

Allows the agent to access URLs and web pages on the host machine.
Usage: fetch_url(url='https://...')
"""

import urllib.request
import urllib.error
import re

from agent.tools import tool
from agent.console import print

def _strip_scripts_and_styles(html):
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    return html

def _decode_response(response):
    """Decode HTTP response body, handling various encodings."""
    raw = response.read()
    
    # Try to get charset from Content-Type header
    content_type = response.headers.get('Content-Type', '')
    charset = None
    
    # Extract charset from Content-Type (e.g., "text/html; charset=iso-8859-1")
    if 'charset=' in content_type:
        charset = content_type.split('charset=')[-1].split(';')[0].strip()
    
    if charset:
        try:
            return raw.decode(charset)
        except (UnicodeDecodeError, LookupError):
            pass  # Fall through to other methods
    
    # Try UTF-8 (most common)
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # Fall back to latin-1 (never fails — every byte is valid)
    return raw.decode('latin-1')

@tool(
    name="fetch_url",
    description=(
        "Browse a website given a URL and return the output.\n"
        "This tool provides access to websites via the HTTP and HTTPS protocols. The website HTML content is stripped of script and style tags."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the site to load",
            },
        },
        "required": ["url"],
    },
)
def fetch_url_handler(args):
    url = args.get("url", "")
    if not url:
        return {"error": "No URL provided"}
    try:
        print(f"  [weak]Accessing[/weak] [path]{url}[/path]")
        req = urllib.request.Request(url, headers={'User-Agent': 'Wisemonkey/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = _decode_response(response)
            return _strip_scripts_and_styles(html)
    except urllib.error.HTTPError as e:
        return {"error": str(e)}
    except urllib.error.URLError as e:
        return {"error": str(e)}

