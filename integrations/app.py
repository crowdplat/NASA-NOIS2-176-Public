from flask import Flask, request, jsonify
import asyncio
from arcgis_handler import send_to_arcgis
from tak_handler import send_and_confirm_tak_retry
from loguru import logger

app = Flask(__name__)

@app.route("/send-polygon", methods=["POST"])
def send_polygon():
    try:
        data = request.get_json()
        polygon = data.get("polygon")
        incident_name = data.get("incident_name", "Default_Incident")
        intensity = data.get("intensity", "Unknown")

        logger.debug(f"Received polygon: {polygon}")

        if not polygon:
            return jsonify({"error": "Missing 'polygon' field"}), 400

        arcgis_result = send_to_arcgis(polygon, incident_name, intensity)
        logger.debug(f"ArcGIS result: {arcgis_result}")
        tak_result = asyncio.run(send_and_confirm_tak_retry(polygon))
        logger.debug(f"TAK result: {tak_result}")

        return jsonify({
            "arcgis": arcgis_result,
            "tak": tak_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
