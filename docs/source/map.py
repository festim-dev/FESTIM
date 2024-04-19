import json
from folium.plugins import MarkerCluster
import folium

import requests
from PIL import Image
from io import BytesIO

LOGO_HEIGHT = 80


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


def generate_map():
    # Load GeoJSON data
    with open("map.json") as f:
        data = json.load(f)

    # Create a map centered on a specific location
    m = folium.Map(location=[42, -71], zoom_start=5, tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(m)
    # Iterate over features in the GeoJSON data
    for feature in data["features"]:
        name = feature["properties"]["name"]
        url = feature["properties"]["url"]
        if url == "URL_PLACEHOLDER":
            url = "https://upload.wikimedia.org/wikipedia/commons/9/92/LOGO_CEA_ORIGINAL.svg"
        coordinates = feature["geometry"]["coordinates"]

        # Get the dimensions of the image from URL
        image_dimensions = get_image_dimensions_from_url(url)
        if image_dimensions:
            height_to_width_ratio = image_dimensions[1] / image_dimensions[0]
            image_dimensions = (int(LOGO_HEIGHT / height_to_width_ratio), LOGO_HEIGHT)
        else:
            image_dimensions = (LOGO_HEIGHT, LOGO_HEIGHT)
        # Create a marker with a custom icon and popup
        if coordinates != [0, 0]:
            icon = folium.CustomIcon(url, icon_size=image_dimensions)
            folium.Marker(
                location=[coordinates[1], coordinates[0]],
                icon=icon,
            ).add_to(marker_cluster)
        else:
            print("no coordinates for", name)
    return m
