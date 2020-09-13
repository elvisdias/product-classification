import configparser, os

config_nfe = configparser.ConfigParser()
config_file=os.path.join(os.path.dirname(__file__), 'config.ini')
config_nfe.read(config_file)