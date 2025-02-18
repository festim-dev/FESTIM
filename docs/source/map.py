import json
from folium.plugins import MarkerCluster
import folium

import requests
from PIL import Image
from io import BytesIO

LOGO_HEIGHT = 60


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
        coordinates = feature["geometry"]["coordinates"]

        # Get the dimensions of the image from local images

        if "path" in feature["properties"]:
            path = feature["properties"]["path"]
            image_dimensions = Image.open(path).size

        if image_dimensions:
            height_to_width_ratio = image_dimensions[1] / image_dimensions[0]
            image_dimensions = (LOGO_HEIGHT, int(LOGO_HEIGHT * height_to_width_ratio))
        else:
            image_dimensions = (LOGO_HEIGHT, LOGO_HEIGHT)
        # Create a marker with a custom icon and popup
        if coordinates != [0, 0]:
            icon = folium.CustomIcon(path, icon_size=image_dimensions)
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
