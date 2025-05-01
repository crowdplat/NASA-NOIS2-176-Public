import threading
import socket
import datetime
import time
import xml.etree.ElementTree as ET
import pytak
import uuid
from configparser import ConfigParser
from dotenv import load_dotenv
import os
import asyncio
from loguru import logger

load_dotenv()

TAK_COT_URL = os.getenv("TAK_COT_URL")
TAK_IP = os.getenv("TAK_IP")
TAK_PORT = int(os.getenv("TAK_PORT"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 10))
WAIT_SECONDS = int(os.getenv("WAIT_SECONDS", 8))

received_flag = threading.Event()

def start_receiver(expected_uid, timeout=WAIT_SECONDS):
    def listen():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((TAK_IP, TAK_PORT))
                s.sendall(create_dummy_message().encode("utf-8"))

                start_time = time.time()
                while time.time() - start_time < timeout:
                    data = s.recv(4096)
                    if expected_uid.encode() in data:
                        logger.debug(f"✅ Confirmed CoT message received with UID: {expected_uid}")
                        received_flag.set()
                        return
        except Exception as e:
            logger.error(f"⚠️ Receiver error: {e}")

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()

def create_dummy_message():
    now = datetime.datetime.utcnow()
    stale = now + datetime.timedelta(seconds=5)
    return f"""<?xml version="1.0"?>
<event version="2.0" uid="Receiver-Dummy" type="t-x-d-d" how="m-g"
       time="{now.isoformat()}Z" start="{now.isoformat()}Z" stale="{stale.isoformat()}Z">
    <point lat="37.7749" lon="-122.4194" hae="10.0" ce="5.0" le="5.0"/>
</event>"""

def gen_polygon_cot(polygon_coords, uid):
    root = ET.Element("event")
    root.set("version", "2.0")
    root.set("type", "a-f-G-U-C")
    root.set("uid", uid)
    root.set("how", "m-g")
    root.set("time", pytak.cot_time())
    root.set("start", pytak.cot_time())
    root.set("stale", pytak.cot_time(300))

    center_lat = sum(p[0] for p in polygon_coords) / len(polygon_coords)
    center_lon = sum(p[1] for p in polygon_coords) / len(polygon_coords)

    detail = ET.SubElement(root, "detail")
    shape = ET.SubElement(detail, "shape")
    polygon = ET.SubElement(shape, "polygon")
    for lat, lon in polygon_coords:
        ET.SubElement(polygon, "point", attrib={"lat": str(lat), "lon": str(lon)})

    ET.SubElement(detail, "strokeColor").text = "#FF0000"
    ET.SubElement(detail, "strokeWidth").text = "3"
    ET.SubElement(detail, "fillColor").text = "#FF000080"

    ET.SubElement(root, "point", attrib={
        "lat": str(center_lat), "lon": str(center_lon),
        "hae": "100", "ce": "5", "le": "5"
    })

    return ET.tostring(root)

class ConfirmingCoTSender(pytak.QueueWorker):
    async def handle_data(self, data):
        await self.put_queue(data)

    async def run(self):
        await self.handle_data(self.data_to_send)
        await asyncio.sleep(2)

    def set_data(self, data):
        self.data_to_send = data

async def send_and_confirm_tak_retry(polygon_coords, max_retries=MAX_RETRIES):
    config = ConfigParser()
    config["mycottool"] = {"COT_URL": TAK_COT_URL}
    config = config["mycottool"]

    clitool = pytak.CLITool(config)
    await clitool.setup()

    for attempt in range(1, max_retries + 1):
        uid = f"tak-{uuid.uuid4()}"
        received_flag.clear()
        logger.debug(f"\n🚀 Attempt {attempt}: Sending CoT with UID {uid}")
        start_receiver(uid)

        sender = ConfirmingCoTSender(clitool.tx_queue, config)
        sender.set_data(gen_polygon_cot(polygon_coords, uid))
        clitool.add_tasks({sender})
        await clitool.run()

        # Wait for confirmation
        received_flag.wait(timeout=WAIT_SECONDS)
        if received_flag.is_set():
            return {"success": True, "uid": uid, "attempt": attempt}

    return {"success": False, "error": "CoT message not confirmed after retries", "attempts": max_retries}
