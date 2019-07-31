from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

def train_nlu(data, configs, model_dir):
	training_data = load_data(data)
	trainer = Trainer(config.load(configs))
	trainer.train(training_data)
	model_directory = trainer.persist(model_dir, fixed_model_name = 'trainedNlu')
    
def run_nlu():
	interpreter = Interpreter.load('./models/nlu/default/trainedNlu')
	print(interpreter.parse(u"I am planning my holiday to Shanghai. What's the weather out there?"))

if __name__ == '__main__':
	train_nlu('./data/training_data.json', 'config_spacy.json', './models/nlu')
	run_nlu()
