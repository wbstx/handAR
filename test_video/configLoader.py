import configparser
from ast import literal_eval


class ConfigLoader:
	def __init__(self):
		self.config = configparser.ConfigParser()
		self.config.read('hand.config')
		self.current_section = self.config['DEFAULT']['current_section']

	@property
	def get_image_scale(self):
		return literal_eval(self.config['DEFAULT']['image_scale'])
	
	@property
	def get_green_background(self):
		return literal_eval(self.config['DEFAULT']['green_background'])
	
	@property
	def get_object_mask(self):
		return literal_eval(self.config['DEFAULT']['object_mask'])
	
	@property
	def get_path(self):
		return self.config[self.current_section]['path']
	
	@property
	def get_object_name(self):
		return self.config[self.current_section]['object_name']
