import os
import re
import requests
import urllib.parse


def main():
    body = os.environ["ISSUE_BODY"]
    issue_number = os.environ["ISSUE_NUMBER"]

    # Parse fields from issue body
    name = extract_field(body, "Institution name")
    short_name = extract_field(body, "Short name or acronym")
    logo_url = extract_image_url(body)

    if not name or not logo_url:
        print("ERROR: Could not parse institution name or logo URL")
        exit(1)

    # Export for use in later steps
    set_env("INSTITUTION_NAME", name)

    # Download logo
    filename = re.sub(r"[^a-zA-Z0-9]", "_", name) + get_extension(logo_url)
    logo_dir = os.path.join("docs", "source", "_static", "logos")
    os.makedirs(logo_dir, exist_ok=True)
    logo_path = os.path.join(logo_dir, filename)

    response = requests.get(logo_url)
    response.raise_for_status()
    with open(logo_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded logo to {logo_path}")

    # Update index.rst
    index_path = os.path.join("docs", "source", "index.rst")
    with open(index_path, "r") as f:
        content = f.read()

    # Build the new logo HTML line
    logo_html = (
        f'       <img src="_static/logos/{filename}" alt="{short_name}" title="{name}">'
    )

    # Insert before the duplicate comment in the first ribbon track
    # Find the first "<!-- Duplicate for seamless loop -->" and insert before it
    marker = "<!-- Duplicate for seamless loop -->"
    first_marker = content.find(marker)

    if first_marker == -1:
        print("ERROR: Could not find duplicate marker in index.rst")
        exit(1)

    # Determine which row has fewer logos (count img tags before each marker)
    second_marker = content.find(marker, first_marker + 1)
    row1_section = content[:first_marker]
    row2_section = content[first_marker:second_marker] if second_marker != -1 else ""

    row1_count = row1_section.count('<img src="_static/logos/')
    row2_count = row2_section.count('<img src="_static/logos/')

    if second_marker != -1 and row2_count < row1_count:
        # Add to second row
        insert_pos = second_marker
    else:
        # Add to first row
        insert_pos = first_marker

    # Insert the logo line (and its duplicate after the marker)
    indent_and_newline = "\n"
    insertion = logo_html + indent_and_newline + "       " + indent_and_newline

    # Find the marker after the chosen insert position
    target_marker_pos = content.find(
        marker, insert_pos if insert_pos == first_marker else second_marker
    )

    # Insert before marker
    content = content[:insert_pos] + logo_html + "\n       " + content[insert_pos:]

    # Also insert after the duplicate section (for seamless loop)
    # Find the closing </div> of the track after the marker
    after_marker = content.find(marker, insert_pos) + len(marker)
    # Find end of duplicated logos (next </div>)
    track_end = content.find("</div>", after_marker)

    # Insert before the </div>
    content = content[:track_end] + logo_html + "\n       " + content[track_end:]

    with open(index_path, "w") as f:
        f.write(content)

    print(f"Updated {index_path}")


def extract_field(body, label):
    """Extract a field value from the issue form body."""
    # GitHub issue forms format: ### Label\n\nValue\n
    pattern = rf"### {re.escape(label)}\s*\n\s*\n(.+?)(?=\n\s*\n###|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_image_url(body):
    """Extract the first image URL from the issue body."""
    # GitHub-hosted images
    pattern = r"https://(?:user-images\.githubusercontent\.com|github\.com)[^\s\)\]]*\.(?:png|jpg|jpeg|svg|gif)"
    match = re.search(pattern, body, re.IGNORECASE)
    if match:
        return match.group(0)

    # Also try markdown image syntax
    pattern = r"!\[.*?\]\((https?://[^\s\)]+)\)"
    match = re.search(pattern, body)
    if match:
        return match.group(1)

    return None


def get_extension(url):
    """Get file extension from URL."""
    path = urllib.parse.urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".svg", ".gif"):
        return ext
    return ".png"


def set_env(key, value):
    """Set environment variable for subsequent GitHub Actions steps."""
    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
