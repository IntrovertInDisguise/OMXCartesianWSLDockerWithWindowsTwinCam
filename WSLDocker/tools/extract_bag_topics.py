#!/usr/bin/env python3
"""
Extract selected topics from a rosbag2 sqlite storage into CSV files.

Usage:
  python3 tools/extract_bag_topics.py <bag_dir> <out_dir>

This uses rosbag2_py.SequentialReader and rclpy.serialization.deserialize_message
to decode messages into Python ROS message objects and write simple CSVs.
"""
import sys
import os
import csv
import importlib
from pathlib import Path

def msg_type_to_module_class(type_str: str):
    # e.g. "geometry_msgs/msg/Pose" -> ("geometry_msgs.msg", "Pose")
    if '/' in type_str:
        pkg, rest = type_str.split('/', 1)
        # rest like 'msg/Pose'
        if rest.startswith('msg/'):
            cls = rest.split('/', 1)[1]
            mod = pkg + '.msg'
            return mod, cls
    # fallback
    parts = type_str.split('/')
    return ('.'.join(parts[:-1]), parts[-1])

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main():
    if len(sys.argv) < 3:
        print("Usage: extract_bag_topics.py <bag_dir> <out_dir>")
        sys.exit(1)
    bag_dir = sys.argv[1]
    out_dir = sys.argv[2]
    ensure_dir(out_dir)

    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        import rclpy
    except Exception as e:
        print("Required ROS2 Python packages not available:", e)
        sys.exit(2)

    # topics we care about and desired CSV headers
    TOPICS = {
        '/robot1/robot1_variable_stiffness/cartesian_pose_desired': ['time', 'x', 'y', 'z'],
        '/robot1/robot1_variable_stiffness/end_effector_position': ['time', 'x', 'y', 'z'],
        '/robot1/robot1_variable_stiffness/contact_wrench': ['time', 'fx', 'fy', 'fz'],
        '/robot2/robot2_variable_stiffness/cartesian_pose_desired': ['time', 'x', 'y', 'z'],
        '/robot2/robot2_variable_stiffness/end_effector_position': ['time', 'x', 'y', 'z'],
        '/robot2/robot2_variable_stiffness/contact_wrench': ['time', 'fx', 'fy', 'fz'],
    }

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topics_and_types}

    # prepare output containers
    rows = {t: [] for t in TOPICS.keys()}

    # init rclpy for deserialize_message
    rclpy.init()

    count = 0
    try:
        while reader.has_next():
            bag_msg = reader.read_next()
            # bag_msg often is a tuple (topic, serialized, timestamp)
            try:
                topic, serdata, ts = bag_msg
            except Exception:
                # try attribute access
                topic = getattr(bag_msg, 'topic_name', None)
                serdata = getattr(bag_msg, 'serialized_data', None)
                ts = getattr(bag_msg, 'timestamp', None)

            if topic not in TOPICS:
                continue

            msg_type = topic_type_map.get(topic)
            if msg_type is None:
                continue

            mod_name, cls_name = msg_type_to_module_class(msg_type)
            try:
                mod = importlib.import_module(mod_name)
                msg_cls = getattr(mod, cls_name)
            except Exception:
                # could not import message type
                continue

            # serdata may be bytes or object; try to get raw bytes
            if hasattr(serdata, 'serialized_data'):
                raw = bytes(serdata.serialized_data)
            elif isinstance(serdata, (bytes, bytearray)):
                raw = bytes(serdata)
            else:
                # some rosbag2_py variants return SerializedBagMessage with 'serialized_data'
                # if unknown, skip
                continue

            try:
                msg = deserialize_message(raw, msg_cls)
            except Exception:
                # skip messages we can't deserialize
                continue

            # timestamp in nanoseconds -> seconds
            t_s = float(ts) * 1e-9 if ts is not None else 0.0

            if 'cartesian_pose_desired' in topic:
                if hasattr(msg, 'position'):
                    x = msg.position.x
                    y = msg.position.y
                    z = msg.position.z
                elif hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
                    x = msg.pose.position.x
                    y = msg.pose.position.y
                    z = msg.pose.position.z
                else:
                    x = getattr(msg, 'x', 0.0)
                    y = getattr(msg, 'y', 0.0)
                    z = getattr(msg, 'z', 0.0)
                rows[topic].append([t_s, x, y, z])

            elif 'end_effector_position' in topic:
                x = getattr(msg, 'x', None)
                if x is None and hasattr(msg, 'point'):
                    x = msg.point.x
                    y = msg.point.y
                    z = msg.point.z
                else:
                    y = getattr(msg, 'y', 0.0)
                    z = getattr(msg, 'z', 0.0)
                rows[topic].append([t_s, x, y, z])

            elif 'contact_wrench' in topic:
                if hasattr(msg, 'wrench') and hasattr(msg.wrench, 'force'):
                    fx = msg.wrench.force.x
                    fy = msg.wrench.force.y
                    fz = msg.wrench.force.z
                else:
                    fx = getattr(msg, 'fx', 0.0)
                    fy = getattr(msg, 'fy', 0.0)
                    fz = getattr(msg, 'fz', 0.0)
                rows[topic].append([t_s, fx, fy, fz])

            count += 1
    finally:
        rclpy.shutdown()

    # write CSVs
    for topic, hdr in TOPICS.items():
        fname = topic.replace('/', '_').lstrip('_') + '.csv'
        outp = os.path.join(out_dir, fname)
        write_csv(outp, hdr, rows[topic])

    print(f"Extracted {sum(len(v) for v in rows.values())} messages into {out_dir}")

if __name__ == '__main__':
    main()
