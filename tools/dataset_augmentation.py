from absl import flags, app, logging
from typing import DefaultDict
import random
import os
import sys
import csv
import shutil
import xml.etree.ElementTree as ET
from absl.flags import FLAGS
from pathlib import Path

from yolov3_tf2.io_utils import get_directory_xml_files, extract_bounding_boxes, get_img_files

flags.DEFINE_string(
    'set_name', None, 'Name of the new set'
)
flags.DEFINE_string(
    'base_set_path', None, 'Path to the dataset'
)
flags.DEFINE_string(
    'set_output_folder', None, 'Path to the dataset output'
)
flags.DEFINE_string(
    'image_format', 'jpg', 'Images format, Defaults jpg'
)

flags.mark_flags_as_required(['set_name', 'base_set_path'])

def main(argv):

    del argv

    set_path = Path(FLAGS.base_set_path)
    output_path = Path(FLAGS.set_output_folder)

    if not set_path.is_dir():
        logging.error(f'{set_path} is not a folder. For this tool to work you must specify it\nExiting...')
        sys.exit(1)

    output_path.mkdir(exist_ok=True)

    xml_files = get_directory_xml_files(set_path)
    bounding_boxes = extract_bounding_boxes(xml_files, set_path)
    # img_files = get_img_files(xml_files, FLAGS.img_format)


if '__main__' == __name__:

    app.run(main)
