"""Network access tool.

Allows the agent to access URLs and web pages on the host machine.
Usage: fetch_url(url='https://...')
"""

import urllib.request
import urllib.error
import urllib.parse
import html as html_lib
import re

from agent.tools import tool
from agent.output import get_output_or_ipc

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
        output = get_output_or_ipc()
        output.print(f"[weak]Accessing[/weak] [link]{url}[/link]", indent=2)
        req = urllib.request.Request(url, headers={'User-Agent': 'Wisemonkey/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = _decode_response(response)
            return _strip_scripts_and_styles(html)
    except urllib.error.HTTPError as e:
        return {"error": str(e)}
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _clean_text(fragment: str) -> str:
    """Strip HTML tags and unescape entities from a fragment of markup."""
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html_lib.unescape(fragment).strip()


def _decode_ddg_url(href: str) -> str:
    """DuckDuckGo wraps result URLs in a redirect like
    ``//duckduckgo.com/l/?uddg=<urlencoded>&rut=...``.
    Decode these back to the real target URL."""
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href or href.startswith("/l/"):
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        target = qs.get("uddg", [None])[0]
        if target:
            return target
    return href


def _parse_ddg_results(html: str, max_results: int) -> list[dict]:
    """Extract {title, url, snippet} dicts from a DuckDuckGo HTML results page."""
    results: list[dict] = []

    # Each organic result is an <a class="result__a" href="...">Title</a>
    # followed (possibly) by a <a class="result__snippet">Snippet</a>.
    link_re = re.compile(
        r'<a[^>]+class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_re = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    # The plain-text snippet variant uses a <div>/<span> instead of <a>.
    snippet_div_re = re.compile(
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</(?:a|div|span)>',
        re.DOTALL | re.IGNORECASE,
    )

    snippets = [m.group("snippet") for m in snippet_re.finditer(html)]
    if not snippets:
        snippets = [m.group("snippet") for m in snippet_div_re.finditer(html)]

    for i, m in enumerate(link_re.finditer(html)):
        if len(results) >= max_results:
            break
        url = _decode_ddg_url(m.group("href"))
        title = _clean_text(m.group("title"))
        snippet = _clean_text(snippets[i]) if i < len(snippets) else ""
        results.append({"title": title, "url": url, "snippet": snippet})

    return results


@tool(
    name="web_search",
    description=(
        "Search the web using DuckDuckGo and return a list of results.\\n"
        "Each result contains a title, URL, and short snippet. Use this to find "
        "pages, then call fetch_url on any result URL to read the full content. "
        "Useful for answering questions about current events or facts not in the model's knowledge."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
            },
        },
        "required": ["query"],
    },
)
def web_search_handler(args):
    query = args.get("query", "")
    if not query:
        return {"error": "No query provided"}
    try:
        max_results = int(args.get("max_results", 5))
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, min(max_results, 20))

    try:
        output = get_output_or_ipc()
        output.print(f"[weak]Searching the web for[/weak] [accent]{query}[/accent]", indent=2)

        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=data,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            page = _decode_response(response)

        results = _parse_ddg_results(page, max_results)
        if not results:
            return {"query": query, "results": [], "note": "No results found."}
        return {"query": query, "results": results}
    except urllib.error.HTTPError as e:
        return {"error": str(e)}
    except urllib.error.URLError as e:
        return {"error": str(e)}

