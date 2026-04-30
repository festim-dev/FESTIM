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

    print(f"Parsed name: {name}")
    print(f"Parsed short_name: {short_name}")
    print(f"Parsed logo_url: {logo_url}")

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

    response = requests.get(logo_url, allow_redirects=True)
    response.raise_for_status()
    with open(logo_path, "wb") as f:
        f.write(response.content)

    # Detect actual content type if URL had no extension
    if filename.endswith(".png"):
        content_type = response.headers.get("Content-Type", "")
        if "svg" in content_type:
            new_filename = filename[:-4] + ".svg"
            os.rename(logo_path, os.path.join(logo_dir, new_filename))
            filename = new_filename
            logo_path = os.path.join(logo_dir, filename)
        elif "jpeg" in content_type or "jpg" in content_type:
            new_filename = filename[:-4] + ".jpg"
            os.rename(logo_path, os.path.join(logo_dir, new_filename))
            filename = new_filename
            logo_path = os.path.join(logo_dir, filename)

    print(f"Downloaded logo to {logo_path}")

    # Update index.rst
    index_path = os.path.join("docs", "source", "index.rst")
    with open(index_path, "r") as f:
        content = f.read()

    # Build the new logo HTML line
    logo_html = (
        f'       <img src="_static/logos/{filename}" alt="{short_name}" title="{name}">'
    )

    # Find all duplicate markers
    marker = "       <!-- Duplicate for seamless loop -->"
    first_marker = content.find(marker)
    second_marker = content.find(marker, first_marker + 1)

    if first_marker == -1:
        print("ERROR: Could not find duplicate marker in index.rst")
        exit(1)

    # Count logos in each row to decide where to add
    row1_section = content[:first_marker]
    row1_count = row1_section.count('<img src="_static/logos/')

    if second_marker != -1:
        row2_section = content[first_marker:second_marker]
        row2_count = row2_section.count('<img src="_static/logos/')
    else:
        row2_count = row1_count  # fallback: add to row 1

    # Choose the row with fewer logos
    if second_marker != -1 and row2_count < row1_count:
        target_marker = second_marker
    else:
        target_marker = first_marker

    # Insert before the marker (original set)
    content = content[:target_marker] + logo_html + "\n" + content[target_marker:]

    # Recalculate marker position after insertion
    marker_pos = content.find(marker, target_marker)
    after_marker = marker_pos + len(marker)

    # Find the closing </div> of this track
    track_end_marker = "     </div>"
    track_end = content.find(track_end_marker, after_marker)

    # Insert before </div> (duplicate set)
    content = content[:track_end] + logo_html + "\n" + content[track_end:]

    with open(index_path, "w") as f:
        f.write(content)

    print(f"Updated {index_path}")


def extract_field(body, label):
    """Extract a field value from the issue form body."""
    pattern = rf"###\s*{re.escape(label)}\s*\n\s*\n(.+?)(?=\n\s*\n###|\n\s*\n\s*$|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        value = match.group(1).strip()
        # Remove any HTML tags (in case the value contains them)
        value = re.sub(r"<[^>]+>", "", value).strip()
        return value if value else None
    return None


def extract_image_url(body):
    """Extract the first image URL from the issue body."""
    # HTML img tag (GitHub drag-and-drop renders as <img> tags)
    pattern = r'<img[^>]+src=["\']([^"\']+)["\']'
    match = re.search(pattern, body, re.IGNORECASE)
    if match:
        return match.group(1)

    # GitHub user-attachments URLs (no file extension)
    pattern = r"https://github\.com/user-attachments/assets/[^\s\"\'\)<]+"
    match = re.search(pattern, body)
    if match:
        return match.group(0)

    # GitHub user-images URLs (with file extension)
    pattern = r"https://user-images\.githubusercontent\.com/[^\s\)\]]+"
    match = re.search(pattern, body)
    if match:
        return match.group(0)

    # Markdown image syntax
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
    # Default to .png for extensionless URLs (like user-attachments)
    return ".png"


def set_env(key, value):
    """Set environment variable for subsequent GitHub Actions steps."""
    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
