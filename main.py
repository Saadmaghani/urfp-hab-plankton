from preprocessing import Preprocessor
from training import Trainer, Metrics
import torch.nn as nn
import torch.optim as optim
from models.first_CNN import firstCNN

years = [str(y) for y in range(2006, 2015)]
classes = ["detritus", "Leptocylindrus", "Chaetoceros", "Rhizosolenia", "Guinardia_delicatula", "Cerataulina",
           "Cylindrotheca", "Skeletonema", "Dactyliosolen", "Thalassiosira", "Dinobryon", "Corethron", "Thalassionema",
           "Ditylum", "pennate", "Prorocentrum", "Pseudonitzschia", "Tintinnid", "Guinardia_striata", "Phaeocystis"]
pp = Preprocessor(years, include_classes=classes, train_eg_per_class=1000)

pp.create_datasets([0.6,0.2,0.2])

trainLoader = pp.get_loaders('train', 128)
validLoader = pp.get_loaders('validation', 128)

trainer = Trainer(epochs = 10, loss_fn = nn.MSELoss, optimizer = optim.SGD, lr = 0.01, momentum = 0.9, useCuda=True)

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

f= open("stats-"+str(model)+".txt","w+")
f.write("{TrainAcc:",trainAcc,", ValidAcc:",validAcc,", TestAcc:",met.accuracy(),"}")
f.close()
