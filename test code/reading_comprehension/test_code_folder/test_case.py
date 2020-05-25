from RC.renders import QATrain, QAPredict
from RC.models import QANet
from RC.preproc import Preproc
import config
import torch

def test_case():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("test 1: Processing data...")
	p=Preproc()
	p.process_train_file(config)
	p.process_predict_file(config)

    print("test 2: Read Comprehension algorithm")
    print("1 Building model...")
    model = QANet(config).to(device)

    print("2 Training model...")
    QATrain(config, model, device)

    print("3 Predicting answer...")
    QAPredict(config, model, device)
  
if __name__ == '__main__':
  	test_case()