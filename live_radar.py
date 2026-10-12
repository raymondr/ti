import serial
import serial.tools.list_ports
import struct
import math
import sys
import time
import argparse
import collections
import os
import re

from classifier import KnightlyClassifier

# ==============================================================================
# Python-Configurable Radar Settings
# Set or modify sensorPosition and BoundingBox parameters here or via functions / CLI
# ==============================================================================

class SensorPosition:
    """Represents sensor position and tilt: [x, y, z, az_tilt, el_tilt]"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 1.0, azimuth_tilt: float = 0.0, elevation_tilt: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.azimuth_tilt = azimuth_tilt
        self.elevation_tilt = elevation_tilt

    def to_params(self) -> str:
        return f"{self.x:g} {self.y:g} {self.z:g} {self.azimuth_tilt:g} {self.elevation_tilt:g}"

    def __repr__(self):
        return f"SensorPosition(x={self.x}, y={self.y}, z={self.z}, azimuth_tilt={self.azimuth_tilt}, elevation_tilt={self.elevation_tilt})"


class BoundingBox:
    """Represents a 3D bounding box: [minX, maxX, minY, maxY, minZ, maxZ]"""
    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z

    def to_params(self) -> str:
        return f"{self.min_x:g} {self.max_x:g} {self.min_y:g} {self.max_y:g} {self.min_z:g} {self.max_z:g}"

    def __repr__(self):
        return f"BoundingBox(min_x={self.min_x}, max_x={self.max_x}, min_y={self.min_y}, max_y={self.max_y}, min_z={self.min_z}, max_z={self.max_z})"


# Default Radar Configuration Parameters
DEFAULT_SENSOR_HEIGHT = 1.0  # Default sensor height above ground in meters (z value)
DEFAULT_SENSOR_POSITION = [0, 0, DEFAULT_SENSOR_HEIGHT, 0, 0]  # [x, y, z (height in m), azimuth_tilt (deg), elevation_tilt (deg)]

DEFAULT_BOUNDING_BOXES = {
    "boundaryBox": [-4, 4, 0, 7.5, -0.5, 3],          # [minX, maxX, minY, maxY, minZ, maxZ]
    "staticBoundaryBox": [-3.5, 3.5, 0.5, 7, 0, 3],
    "presenceBoundaryBox": [-3.5, 3.5, 0.5, 7, 0, 3],
}


def normalize_box_name(name: str) -> str:
    """
    Normalizes bounding box parameter names into standard TI radar command names (camelCase with boundaryBox).
    Handles snake_case, camelCase, and variations of 'bounding' vs 'boundary'.
    e.g.:
        'boundary_box' -> 'boundaryBox'
        'boundingBox' -> 'boundaryBox'
        'static_bounding_box' -> 'staticBoundaryBox'
        'presence_bounding_box' -> 'presenceBoundaryBox'
        'occupancyBoundingBox' -> 'occupancyBoundaryBox'
    """
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    s2 = s1.replace("bounding", "boundary")
    parts = [p for p in s2.split('_') if p]
    if not parts:
        return name
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


def format_param_value(val) -> str:
    """Formats values (list, tuple, dict, str, SensorPosition, BoundingBox) to a space-delimited string."""
    if val is None:
        return ""
    if hasattr(val, 'to_params'):
        return val.to_params()
    if isinstance(val, (list, tuple)):
        items = []
        for v in val:
            if isinstance(v, str):
                items.extend(v.split())
            elif isinstance(v, float):
                items.append(f"{v:g}")
            else:
                items.append(str(v))
        return " ".join(items)
    if isinstance(val, dict):
        return " ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in val.values())
    if isinstance(val, str):
        return " ".join(val.strip().split())
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def prepare_config_commands(
    config_file='object.cfg',
    sensor_height=None,
    boundary_box=None,
    static_boundary_box=None,
    presence_boundary_box=None,
    bounding_boxes=None,
    **extra_boxes
) -> list:
    """
    Reads a TI radar config file and overrides or injects sensorPosition and any BoundingBox commands
    using Python-configured parameters.
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()

    # Determine effective sensor position (uses sensor_height for z, defaults for x, y, az, el)
    eff_height = sensor_height if sensor_height is not None else DEFAULT_SENSOR_HEIGHT
    eff_sensor_pos = [DEFAULT_SENSOR_POSITION[0], DEFAULT_SENSOR_POSITION[1], eff_height, DEFAULT_SENSOR_POSITION[3], DEFAULT_SENSOR_POSITION[4]]

    # Build dictionary of effective bounding boxes starting from defaults
    eff_boxes = dict(DEFAULT_BOUNDING_BOXES)
    if bounding_boxes:
        for k, v in bounding_boxes.items():
            eff_boxes[normalize_box_name(k)] = v
    if boundary_box is not None:
        eff_boxes["boundaryBox"] = boundary_box
    if static_boundary_box is not None:
        eff_boxes["staticBoundaryBox"] = static_boundary_box
    if presence_boundary_box is not None:
        eff_boxes["presenceBoundaryBox"] = presence_boundary_box
    for k, v in extra_boxes.items():
        eff_boxes[normalize_box_name(k)] = v

    commands = []
    applied_boxes = set()
    applied_sensor_pos = False

    for line in lines:
        cmd_line = line.strip()
        if cmd_line.startswith('%') or not cmd_line:
            continue
        tokens = cmd_line.split()
        cmd_name = tokens[0]

        if cmd_name == 'sensorPosition':
            if eff_sensor_pos is not None:
                commands.append(f"sensorPosition {format_param_value(eff_sensor_pos)}")
                applied_sensor_pos = True
            else:
                commands.append(cmd_line)
        elif normalize_box_name(cmd_name) in eff_boxes or "boundarybox" in cmd_name.lower() or "boundingbox" in cmd_name.lower():
            box_key = normalize_box_name(cmd_name)
            if box_key in eff_boxes:
                commands.append(f"{cmd_name} {format_param_value(eff_boxes[box_key])}")
                applied_boxes.add(box_key)
            else:
                commands.append(cmd_line)
        else:
            commands.append(cmd_line)

    # Insert any configured parameters that weren't present in the file before sensorStart or baudRate
    insert_idx = len(commands)
    for i, cmd in enumerate(commands):
        if cmd.startswith("sensorStart") or cmd.startswith("baudRate"):
            insert_idx = i
            break

    unapplied = []
    if eff_sensor_pos is not None and not applied_sensor_pos:
        unapplied.append(f"sensorPosition {format_param_value(eff_sensor_pos)}")
    for box_key, val in eff_boxes.items():
        if box_key not in applied_boxes:
            unapplied.append(f"{box_key} {format_param_value(val)}")

    for cmd in reversed(unapplied):
        commands.insert(insert_idx, cmd)

    return commands


def send_config(
    cfg_port_name,
    config_file='object.cfg',
    sensor_height=None,
    boundary_box=None,
    static_boundary_box=None,
    presence_boundary_box=None,
    bounding_boxes=None,
    **extra_boxes
):
    print(f"Opening CFG port {cfg_port_name}...")
    try:
        # Check if sensor is already in high baud rate or stopped
        cli_port = serial.Serial(cfg_port_name, 115200, timeout=0.1)
        cli_port.write(b'\r\nsensorStop 0\r\n')
        time.sleep(0.1)
        resp = cli_port.read(cli_port.in_waiting or 0)
        
        # If no response at 115200, it might already be at 1250000
        if not resp:
            cli_port.baudrate = 1250000
            cli_port.write(b'\r\nsensorStop 0\r\n')
            time.sleep(0.1)
            cli_port.read(cli_port.in_waiting or 0)
            cli_port.write(b'baudRate 115200\r\n')
            time.sleep(0.1)
            cli_port.baudrate = 115200

        commands = prepare_config_commands(
            config_file=config_file,
            sensor_height=sensor_height,
            boundary_box=boundary_box,
            static_boundary_box=static_boundary_box,
            presence_boundary_box=presence_boundary_box,
            bounding_boxes=bounding_boxes,
            **extra_boxes
        )

        print(f"Sending configuration from {config_file} (parameterized from Python)...")
        for cmd in commands:
            if cmd.startswith("sensorPosition") or "boundarybox" in cmd.lower() or "boundingbox" in cmd.lower():
                print(f"  🔧 {cmd}")

        for cmd in commands:
            cli_port.write((cmd + '\r').encode('utf-8'))
            if cmd.startswith("baudRate"):
                time.sleep(0.1)
                try:
                    new_baud = int(cmd.split()[1])
                    cli_port.baudrate = new_baud
                except Exception:
                    pass
            elif "sensorStart" in cmd:
                time.sleep(1.5)
            else:
                time.sleep(0.05)
                    
        cli_port.close()
        print("✅ Configuration sent successfully!\n")
    except serial.SerialException as e:
        print(f"❌ Error opening CFG port {cfg_port_name}: {e}")
        sys.exit(1)

def select_best_track(targets: list, metric: str = "persistence_confidence") -> dict | None:
    """
    Filters a list of tracks/targets to return the single track with the highest persistence/confidence.

    :param targets: List of target dictionaries.
    :param metric: 'persistence_confidence' (primary: persistence, secondary: confidence),
                   'confidence' (primary: confidence, secondary: persistence),
                   or 'persistence' (strictly persistence).
    :return: The target dictionary with highest persistence/confidence, or None if targets is empty.
    """
    if not targets:
        return None
    if len(targets) == 1:
        return targets[0]

    if metric == "confidence":
        return max(targets, key=lambda t: (t.get("confidence", 0.0), t.get("persistence", 0)))
    elif metric == "persistence":
        return max(targets, key=lambda t: (t.get("persistence", 0), t.get("confidence", 0.0)))
    else:  # "persistence_confidence"
        return max(targets, key=lambda t: (t.get("persistence", 0), t.get("confidence", 0.0)))


def run_live_radar(
    data_port_name,
    baudrate,
    cfg_port_name=None,
    on_target=None,
    filter_highest=True,
    filter_metric="persistence_confidence",
    sensor_height=DEFAULT_SENSOR_HEIGHT,
    only_on_change=True,
    bed_location=None,
    window_size=3,
    **kwargs
):
    if "only_on_posture_change" in kwargs:
        only_on_change = kwargs["only_on_posture_change"]

    classifier = KnightlyClassifier(bed_location=bed_location)
    eff_sensor_height = sensor_height if sensor_height is not None else DEFAULT_SENSOR_HEIGHT
    magic = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    buffer = b''
    track_persistence = {}
    last_seen = {}
    last_state = {}
    state_history = {}
    
    ports_to_try = [data_port_name]
    if cfg_port_name and cfg_port_name != data_port_name:
        ports_to_try.append(cfg_port_name)

    for port in ports_to_try:
        print(f"Connecting to DATA port {port} at {baudrate} baud...")
        try:
            with serial.Serial(port, baudrate, timeout=1.0) as ser:
                print(f"Waiting for radar data stream on {port}...")
                start_t = time.time()
                got_magic = False
                while time.time() - start_t < 3.0:
                    d = ser.read(ser.in_waiting or 1)
                    if d:
                        buffer += d
                        if magic in buffer:
                            got_magic = True
                            break
                if got_magic:
                    print(f"✅ Connected to {port}! Streaming live tracking data from the TI Radar...\n")
                    while True:
                        new_data = ser.read(ser.in_waiting or 1)
                        if not new_data:
                            continue
                        
                        buffer += new_data
                        
                        idx = buffer.find(magic)
                        if idx != -1:
                            buffer = buffer[idx:]
                            if len(buffer) >= 40:
                                header = buffer[:40]
                                m, ver, plen, plat, frameNum, cpuCyc, numObj, numTLV, subFrame = struct.unpack('<8sIIIIIIII', header)
                                
                                if len(buffer) >= plen:
                                    packet = buffer[:plen]
                                    buffer = buffer[plen:]
                                    
                                    offset = 40
                                    frame_targets = []
                                    for i in range(numTLV):
                                        if offset + 8 > len(packet): break
                                        tlv_type, tlv_len = struct.unpack('<II', packet[offset:offset+8])
                                        
                                        if tlv_type == 308:
                                            num_targets = tlv_len // 112
                                            for t in range(num_targets):
                                                t_off = offset + 8 + (t * 112)
                                                if t_off + 28 > len(packet): break
                                                
                                                tid, x, y, z, vx, vy, vz = struct.unpack('<Iffffff', packet[t_off:t_off+28])
                                                if eff_sensor_height is not None:
                                                    z += eff_sensor_height
                                                velocity = math.sqrt(vx*vx + vy*vy + vz*vz)
                                                
                                                # Unpack confidenceLevel at offset 108 if 112-byte struct
                                                confidence = 0.0
                                                if t_off + 112 <= len(packet):
                                                    confidence = struct.unpack('<f', packet[t_off+108:t_off+112])[0]
                                                
                                                # Track persistence (frame lifetime count)
                                                persistence = track_persistence.get(tid, 0) + 1
                                                track_persistence[tid] = persistence
                                                last_seen[tid] = frameNum

                                                classification = classifier.classify_target(x=x, y=y, z=z, velocity=velocity, vy=vy)
                                                target_type = classification["type"]
                                                posture = classification["posture"]
                                                
                                                target_dict = {
                                                    "tid": tid,
                                                    "x": x,
                                                    "y": y,
                                                    "z": z,
                                                    "vx": vx,
                                                    "vy": vy,
                                                    "vz": vz,
                                                    "velocity": velocity,
                                                    "confidence": confidence,
                                                    "persistence": persistence,
                                                    "type": target_type,
                                                    "posture": posture,
                                                    "classification": classification,
                                                }
                                                frame_targets.append(target_dict)
                                                
                                        offset += 8 + tlv_len

                                    # Clean up stale tracks not seen in last 15 frames
                                    for t_id in list(track_persistence.keys()):
                                        if frameNum - last_seen.get(t_id, frameNum) > 15:
                                            track_persistence.pop(t_id, None)
                                            last_seen.pop(t_id, None)
                                            last_state.pop(t_id, None)
                                            state_history.pop(t_id, None)

                                    if frame_targets:
                                        if filter_highest:
                                            best_tgt = select_best_track(frame_targets, metric=filter_metric)
                                            selected_targets = [best_tgt] if best_tgt else []
                                        else:
                                            selected_targets = frame_targets

                                        for tgt in selected_targets:
                                            tid = tgt['tid']
                                            posture = tgt['posture']
                                            current_state = (posture)

                                            eff_window = max(1, window_size)
                                            if tid not in state_history:
                                                state_history[tid] = collections.deque(maxlen=eff_window)
                                            state_history[tid].append(current_state)

                                            prev_state = last_state.get(tid)

                                            # Require state to be sustained over window_size frames before reporting change
                                            is_sustained = (
                                                len(state_history[tid]) >= eff_window
                                                and all(s == current_state for s in state_history[tid])
                                            )

                                            should_print = False
                                            if not only_on_change:
                                                should_print = True
                                            elif is_sustained and current_state != prev_state:
                                                should_print = True
                                                last_state[tid] = current_state

                                            if should_print:
                                                print(
                                                    f"[Frame {frameNum}] Live Target [ID: {tgt['tid']}] -> "
                                                    f"X: {tgt['x']:5.2f}m | Y: {tgt['y']:5.2f}m | Z: {tgt['z']:5.2f}m | "
                                                    f"Speed: {tgt['velocity']:5.2f}m/s | Type: {tgt['type']} | "
                                                    f"Posture: {tgt['posture']} |"
                                                    f"Persistence: {tgt['persistence']:2d} | "
                                                    f"Conf: {tgt['confidence']:.2f}"
                                                )
                                            if on_target:
                                                on_target(tgt)
                        
                        # Prevent buffer from growing infinitely if baud rate is wrong and magic is never found
                        if len(buffer) > 10000:
                            buffer = buffer[-100:]
                    return
                else:
                    if len(ports_to_try) > 1 and port != ports_to_try[-1]:
                        print(f"ℹ️ No data detected on {port} (IWRL6432 streams data over {cfg_port_name}). Switching to {ports_to_try[-1]}...")
                        buffer = b''
                        continue
        except serial.SerialException as e:
            print(f"\n❌ ERROR: Could not open {port}: {e}")
            if len(ports_to_try) > 1 and port != ports_to_try[-1]:
                print(f"ℹ️ Trying {ports_to_try[-1]} instead...")
                continue
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TI mmWave Live Radar Streamer & Classifier")
    parser.add_argument('--config', type=str, help="Path to config file (e.g. object.cfg)")
    
    # Set different defaults depending on OS
    if sys.platform.startswith('win'):
        default_cfg = 'COM5'
        default_data = 'COM6'
    else:
        default_cfg = '/dev/tty.usbmodem0000RI2A1'
        default_data = '/dev/tty.usbmodem0000RI2A4'
        
    parser.add_argument('--cfg-port', type=str, default=default_cfg, help=f"CFG Serial port (default: {default_cfg})")
    parser.add_argument('--data-port', type=str, default=default_data, help=f"DATA Serial port (default: {default_data})")
    parser.add_argument('--baud', type=int, default=1250000, help="Data port baud rate (default: 1250000)")
    
    # Parameterized Radar Options
    parser.add_argument('--sensor-height', type=float, default=None, help="Sensor height above ground in meters (z value, e.g. 0.61)")
    parser.add_argument('--boundary-box', nargs='*', help="Main boundary box limits: minX maxX minY maxY minZ maxZ")
    parser.add_argument('--static-boundary-box', nargs='*', help="Static boundary box limits: minX maxX minY maxY minZ maxZ")
    parser.add_argument('--presence-boundary-box', nargs='*', help="Presence boundary box limits: minX maxX minY maxY minZ maxZ")
    parser.add_argument('--bed-location', nargs='*', help="Bed bounding box limits: minX maxX minY maxY [minZ maxZ]")
    parser.add_argument('--window-size', type=int, default=3, help="Number of consecutive frames required for a sustained state before printing (default: 3)")
    parser.add_argument('--filter-metric', type=str, default='persistence_confidence', choices=['persistence_confidence', 'confidence', 'persistence'], help="Metric to rank tracks when multiple exist in a frame (default: persistence_confidence)")
    parser.add_argument('--all-tracks', action='store_true', help="Output all tracks instead of filtering to the highest persistence/confidence track")
    parser.add_argument('--print-all-frames', action='store_true', help="Print target details on every frame instead of only when posture or door approach changes")
    
    args = parser.parse_args()

    cfg_file = args.config
    has_custom_params = any([
        args.sensor_height is not None,
        args.boundary_box,
        args.static_boundary_box,
        args.presence_boundary_box,
    ])
    if not cfg_file and has_custom_params:
        if os.path.exists("object.cfg"):
            cfg_file = "object.cfg"

    effective_sensor_height = args.sensor_height if args.sensor_height is not None else DEFAULT_SENSOR_HEIGHT

    bed_location = None
    if args.bed_location:
        raw_bed = []
        for item in args.bed_location:
            if isinstance(item, str):
                raw_bed.extend(item.replace(',', ' ').split())
            else:
                raw_bed.append(item)
        try:
            bed_location = [float(v) for v in raw_bed]
        except ValueError:
            bed_location = raw_bed

    if cfg_file:
        send_config(
            args.cfg_port,
            config_file=cfg_file,
            sensor_height=args.sensor_height,
            boundary_box=args.boundary_box,
            static_boundary_box=args.static_boundary_box,
            presence_boundary_box=args.presence_boundary_box,
        )
        
    run_live_radar(
        args.data_port,
        args.baud,
        args.cfg_port,
        sensor_height=effective_sensor_height,
        filter_highest=not args.all_tracks,
        filter_metric=args.filter_metric,
        only_on_change=not args.print_all_frames,
        bed_location=bed_location,
        window_size=args.window_size,
    )

