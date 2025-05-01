import numpy as np
import pandas as pd
import math

# Constants
a = 6378137.0
f = 1.0 / 298.257223563
b = a * (1 - f)
e2 = 1 - (b * b) / (a * a)

def deg2rad(deg): return deg * math.pi / 180.0
def rad2deg(rad): return rad * 180.0 / math.pi

def lla_to_ecef(lat, lon, h):
    lat, lon = deg2rad(lat), deg2rad(lon)
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + h) * math.sin(lat)
    return np.array([x, y, z])

def ned_rotation(lat, lon):
    lat, lon = deg2rad(lat), deg2rad(lon)
    return np.array([
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        [-math.sin(lon), math.cos(lon), 0],
        [-math.cos(lat) * math.cos(lon), -math.cos(lat) * math.sin(lon), -math.sin(lat)]
    ])

def lla2ned(lla, origin_lla):
    ecef = lla_to_ecef(*lla)
    origin_ecef = lla_to_ecef(*origin_lla)
    R = ned_rotation(origin_lla[0], origin_lla[1])
    return R @ (ecef - origin_ecef)

def ned2lla(ned, origin_lla):
    origin_ecef = lla_to_ecef(*origin_lla)
    R = ned_rotation(origin_lla[0], origin_lla[1])
    ecef = origin_ecef + R.T @ ned
    x, y, z = ecef
    lon = math.atan2(y, x)
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(5):
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        lat = math.atan2(z + e2 * N * math.sin(lat), p)
    h = p / math.cos(lat) - N
    return np.array([rad2deg(lat), rad2deg(lon), h])

def rotvec2mat(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)

def fudge_alignment(derived, target):
    axis = np.cross(derived, target)
    angle = math.acos(np.clip(np.dot(derived, target), -1.0, 1.0))
    return rotvec2mat(axis, angle)

def load_metadata(metadata_file, systemtime):
    df = pd.read_csv(metadata_file)
    df["timestamp_diff"] = abs(df["UNIXTimeStamp___Microseconds"] - systemtime)
    row = df.loc[df["timestamp_diff"].idxmin()]
    return {
        "drone_lla": np.array([row["Sensor Latitude"], row["Sensor Longitude"], row["Sensor Ellipsoid Height"]]),
        "targ_lla": np.array([row["Frame Center Latitude"], row["Frame Center Longitude"], row["Frame Center Height Above Ellipsoid"]]),
        "drone_yaw": row["Platform Heading Angle"],
        "drone_pitch": row["Platform Pitch Angle (Full)"],
        "drone_roll": row["Platform Roll Angle (Full)"],
        "camera_HFOV": row["Sensor Horizontal Field of View"],
        "camera_VFOV": row["Sensor Vertical Field of View"],
        "camera_yaw": row["Sensor Relative Azimuth Angle"],
        "camera_pitch": row["Sensor Relative Elevation Angle"],
        "camera_roll": row["Sensor Relative Roll Angle"]
    }

def compute_world_coordinates(pixels, meta, numrow_pix=1080, numcol_pix=1350, cencol_pix=960, apply_fudge=True):
    cenrow_pix = numrow_pix / 2.0
    drone_lla = meta["drone_lla"]
    lla0 = meta["targ_lla"]
    xyzNED = lla2ned(drone_lla, lla0)
    target_vec = -xyzNED / np.linalg.norm(xyzNED)

    bsite = np.array([1, 0, 0])
    haxis = np.array([0, 1, 0])
    vaxis = np.array([0, 0, 1])

    R = rotvec2mat(np.array([0, 0, 1]), deg2rad(meta["drone_yaw"]))
    bsite = R @ bsite
    haxis = R @ haxis
    vaxis = R @ vaxis

    pitch_axis = rotvec2mat(np.array([0, 0, 1]), math.pi / 2.0) @ bsite
    R = rotvec2mat(pitch_axis, deg2rad(meta["drone_pitch"]))
    bsite = R @ bsite
    haxis = R @ haxis
    vaxis = R @ vaxis

    R = rotvec2mat(bsite, deg2rad(meta["drone_roll"]))
    bsite = R @ bsite
    haxis = R @ haxis
    vaxis = R @ vaxis

    R = rotvec2mat(vaxis, deg2rad(meta["camera_yaw"]))
    bsite = R @ bsite
    haxis = R @ haxis
    vaxis = R @ vaxis

    R = rotvec2mat(haxis, deg2rad(meta["camera_pitch"]))
    bsite = R @ bsite
    haxis = R @ haxis
    vaxis = R @ vaxis

    R = rotvec2mat(bsite, deg2rad(meta["camera_roll"]))
    bsite = R @ bsite
    haxis = R @ haxis
    vaxis = R @ vaxis

    if apply_fudge:
        R = fudge_alignment(bsite, target_vec)
        bsite = R @ bsite
        haxis = R @ haxis
        vaxis = R @ vaxis

    results = []
    for row_pix, col_pix in pixels:
        yaw_pix = meta["camera_HFOV"] * (col_pix - cencol_pix) / numcol_pix
        pitch_pix = -meta["camera_VFOV"] * (row_pix - cenrow_pix) / numrow_pix
        ray = rotvec2mat(vaxis, deg2rad(yaw_pix)) @ bsite
        ray = rotvec2mat(haxis, deg2rad(pitch_pix)) @ ray
        n = np.array([0, 0, -1])
        t = -(n @ xyzNED) / (n @ ray)
        I = xyzNED + t * ray
        geodetic = ned2lla(I, lla0)
        results.append({
            "row_pix": row_pix,
            "col_pix": col_pix,
            "latitude": geodetic[0],
            "longitude": geodetic[1],
            "altitude": geodetic[2]
        })
    return pd.DataFrame(results)

# Example Main execution
if __name__ == "__main__":
    # These are the pixel coordinates of the convex hull of the fire boundary.
    pixels = [(914,875),(899,890),(854,930),(839,942),(704,1031),(614,1079),
              (608,1079),(458,980),(120,674),(117,670),(30,466),(30,460),
              (342,121),(629,121),(869,207),(873,211),(914,874)]
    
    # Place the path to your metadata file that consists all the camera and drone prameters as above 
    metadata_file = "your_metadata_file.csv"
    # Here we are selecting rows based on system_time column for 1-to-1 mapping of image and its metadata
    system_time = 1729041510838477  
    meta = load_metadata(metadata_file, system_time)
    df = compute_world_coordinates(pixels, meta)
    df.to_csv("output_latlon_.csv", index=False)
    print("Saved geodetic results to 'output_latlon_ignore.csv'")
