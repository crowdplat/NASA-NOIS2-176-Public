import requests
import json
from dotenv import load_dotenv
import os
from loguru import logger

load_dotenv()

# ArcGIS Credentials and Endpoint
ARCGIS_CLIENT_ID = os.getenv("ARCGIS_CLIENT_ID")
ARCGIS_CLIENT_SECRET = os.getenv("ARCGIS_CLIENT_SECRET")
ARCGIS_FEATURE_LAYER_URL = os.getenv("ARCGIS_FEATURE_LAYER_URL")

def get_arcgis_token():
    token_url = "https://www.arcgis.com/sharing/rest/oauth2/token"
    logger.debug(f"ARCGIS CLIENT_ID: {ARCGIS_CLIENT_ID}")
    logger.debug(f"ARCGIS CLIENT_SECRET: {ARCGIS_CLIENT_SECRET}")
    payload = {
        "client_id": ARCGIS_CLIENT_ID,
        "client_secret": ARCGIS_CLIENT_SECRET,
        "grant_type": "client_credentials",
        "f": "json"
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"ArcGIS token error: {response.text}")

def send_to_arcgis(polygon_coords, incident_name, intensity):
    token = get_arcgis_token()
    logger.debug(f"ArcGIS token: {token}")
    geometry = {
        "rings": [polygon_coords],
        "spatialReference": {"wkid": 4326}
    }
    logger.debug(f"Geometry: {geometry}")
    attributes = {
        "IncidentName": incident_name,
        "Intensity": intensity
    }
    feature = {"attributes": attributes, "geometry": geometry}
    logger.debug(f"Feature: {feature}")
    params = {
        "f": "json",
        "token": token,
        "features": json.dumps([feature])
    }
    response = requests.post(ARCGIS_FEATURE_LAYER_URL, data=params)
    return response.json()
