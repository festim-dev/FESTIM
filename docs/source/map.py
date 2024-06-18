import json
from folium.plugins import MarkerCluster
import folium

import requests
from PIL import Image
from io import BytesIO

LOGO_HEIGHT = 60


def get_image_dimensions_from_url(image_url):
    """Function to get the dimensions of an image from URL


    Args:
        image_url (str): the url

    Returns:
        tuple: the dimensions of the image
    """
    headers = {
        "User-Agent": "FESTIM (https://github.com/festim-dev/festim; remidm@mit.edu)"
    }

    try:
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        img = Image.open(BytesIO(response.content))
        return img.size
    except Exception as e:
        print(f"Error fetching or processing image: {e}")
        return None


def generate_map(clustered=True, draggable=False):
    # Load GeoJSON data
    with open("map.json") as f:
        data = json.load(f)

    # Create a map centered on a specific location
    m = folium.Map(
        location=[42.256349987281666, -37.61204625889672],
        zoom_start=3,
        tiles="cartodbpositron",
    )
    if clustered:
        marker_cluster = MarkerCluster().add_to(m)
    # Iterate over features in the GeoJSON data
    for feature in data["features"]:
        name = feature["properties"]["name"]
        print(name)
        url = feature["properties"]["url"]
        if url == "URL_PLACEHOLDER":
            url = "https://upload.wikimedia.org/wikipedia/commons/9/92/LOGO_CEA_ORIGINAL.svg"
        coordinates = feature["geometry"]["coordinates"]

        # Get the dimensions of the image from URL
        image_dimensions = get_image_dimensions_from_url(url)
        if image_dimensions:
            height_to_width_ratio = image_dimensions[1] / image_dimensions[0]
            image_dimensions = (LOGO_HEIGHT, int(LOGO_HEIGHT * height_to_width_ratio))
        else:
            image_dimensions = (LOGO_HEIGHT, LOGO_HEIGHT)
        # Create a marker with a custom icon and popup
        if coordinates != [0, 0]:
            icon = folium.CustomIcon(url, icon_size=image_dimensions)
            marker = folium.Marker(
                location=[coordinates[1], coordinates[0]],
                icon=icon,
                draggable=draggable,
            )
            if clustered:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)
        else:
            print("no coordinates for", name)
    return m


if __name__ == "__main__":
    m = generate_map(clustered=False, draggable=True)
    m.save("map.html")
    print("Map saved to map.html")
