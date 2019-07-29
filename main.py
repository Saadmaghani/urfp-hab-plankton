from preprocessing import Preprocessor
from training import Trainer, Metrics
import torch.nn as nn
import torch.optim as optim
from models.first_CNN import firstCNN
from configuration import Hyperparameters as HP

years = [str(y) for y in range(2006, 2015)]
classes = ["detritus", "Leptocylindrus", "Chaetoceros", "Rhizosolenia", "Guinardia_delicatula", "Cerataulina",
           "Cylindrotheca", "Skeletonema", "Dactyliosolen", "Thalassiosira", "Dinobryon", "Corethron", "Thalassionema",
           "Ditylum", "pennate", "Prorocentrum", "Pseudonitzschia", "Tintinnid", "Guinardia_striata", "Phaeocystis"]
pp = Preprocessor(years, include_classes=classes, train_eg_per_class=HP.number_of_images_per_class)

pp.create_datasets([0.6,0.2,0.2])

trainLoader = pp.get_loaders('train', 128)
validLoader = pp.get_loaders('validation', 128)

trainer = Trainer(HP_version = HP.version, epochs = HP.number_of_epochs, loss_fn = HP.loss_function, 
	optimizer = HP.optimizer, lr = HP.learning_rate, momentum = HP.momentum, useCuda=True)

model = firstCNN()

trainAcc, validAcc = trainer.train(model, trainLoader, validLoader)

# - or -

#model = trainer.load_full_model(model, "./models/firstCNN-1.0.pth")

testLoader = pp.get_loaders('test', 128)
pred, target = trainer.test(model, testLoader)

met = Metrics(target, pred)
met.accuracy()
met.recall()
met.f_score()
#met.plot_CM()

f= open("stats-"+str(model)+"-"+str(HP.version)+".txt","w+")
str_to_write = "{TrainAcc: "+ str(trainAcc)+", ValidAcc: "+str(validAcc)+", TestAcc: "+str(met.accuracy())+"}"
f.write(str_to_write)
f.close()
