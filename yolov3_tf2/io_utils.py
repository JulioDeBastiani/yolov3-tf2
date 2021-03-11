import os
import xml.etree.ElementTree as ET
from typing import List
import shutil
from imgaug import BoundingBox


def get_directory_xml_files(directory_path):

    set_files: list = []

    for _, _, files in os.walk(directory_path, topdown=False):
        for file in files:
            if file.split('.')[1] == 'xml':
                set_files.append(file)

    return set_files

def extract_bounding_boxes(xml_files, set_path):

    bounding_boxes: list = []

    for xml_file in xml_files:
        tree = ET.parse(os.path.join(set_path, xml_file))
        root = tree.getroot()
        for xml_annotation in root.findall('object'):

            single_file_bounding_boxes: List[BoundingBox] = []

            for bound_box in xml_annotation.findall('bndbox'):

                xmin: list = []
                ymin: list = []
                xmax: list = []
                ymax: list = []

                for xminimun in bound_box.findall('xmin'):
                    xmin.append(int(xminimun.text))

                for yminimun in bound_box.findall('ymin'):
                    ymin.append(int(yminimun.text))

                for xmaximun in bound_box.findall('xmax'):
                    xmax.append(int(xmaximun.text))

                for ymaximun in bound_box.findall('ymax'):
                    ymax.append(int(ymaximun.text))
            for i in range(len(xmin)):
                single_file_bounding_boxes.append(BoundingBox(
                    x1=xmin[i], y1=ymin[i], x2=xmax[i], y2=ymax[i]
                ))

        bounding_boxes.append(single_file_bounding_boxes)

    return bounding_boxes

def get_img_files(xml_files, img_format):
    pass
