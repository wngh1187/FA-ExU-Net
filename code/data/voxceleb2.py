import os
from dataclasses import dataclass

@dataclass
class TrainItem:
	path: str
	speaker: str

@dataclass
class EnrollmentItem:
	key: str
	path: str

@dataclass
class TestTrial:
	key1: str
	key2: str
	label: str

class VoxCeleb2:
	@property
	def train_set(self):
		return self.__train_set

	@property
	def train_speakers(self):
		return self.__train_speakers


	def __init__(self, path_train):
		# train_set
		self.__train_set = []
		for root, _, files in os.walk(path_train):
			for file in files:
				if '.wav' in file:
					temp = os.path.join(root, file)
					self.__train_set.append(
						TrainItem(
							path=temp,
							speaker=temp.split('/')[-3]
						)
					)

		# train_speakers
		temp = {}
		for item in self.train_set:
			try:
				temp[item.speaker]
			except:
				temp[item.speaker] = None
		self.__train_speakers = temp.keys()

				
		# error check
		# assert len(self.train_set) == 148642, f'len(train_set): {len(self.train_set)}'
		# assert len(self.train_speakers) == 6112, f'len(train_speakers): {len(self.train_speakers)}'

	def _parse_trials(self, path):
		trials = []

		f = open(path) 
		for line in f.readlines():
			strI = line.split(' ')
			trials.append(
				TestTrial(
					key1=strI[1].replace('\n', ''),
					key2=strI[2].replace('\n', ''),
					label=strI[0]
				)
			)
		return trials