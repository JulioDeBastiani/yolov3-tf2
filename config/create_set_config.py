import configparser

parser = configparser.RawConfigParser()
parser.read('config/config.cfg')

base_set_path: str = parser.get('create_set', 'base_set_path')
frozen_set_name: str = parser.get('create_set', 'frozen_set_name')
set_output_folder: str = parser.get('create_set', 'set_output_folder')
set_dev_percentage: float = float(parser.get('create_set', 'set_dev_percentage'))
set_train_percentage: float = float(parser.get('create_set', 'set_train_percentage'))
set_val_percentage: float = float(parser.get('create_set', 'set_val_percentage'))
set_xml_path: str = parser.get('create_set', 'set_xml_path')
