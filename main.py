from preprocessing import Preprocessor
from training import Trainer
from metrics import Metrics
import torch.nn as nn
import torch.optim as optim
from models.vgg_TL import VGG, GoogleNet, ResNet
from configuration import Hyperparameters as HP
import torch

years = [str(y) for y in range(2006, 2015)]

classes = ["detritus", "Leptocylindrus", "Chaetoceros", "Rhizosolenia", "Guinardia_delicatula", "Cerataulina", "Cylindrotheca",
    "Skeletonema", "Dactyliosolen", "Thalassiosira", "Dinobryon", "Corethron", "Thalassionema", "Ditylum", "pennate", "Prorocentrum",
    "Pseudonitzschia", "Tintinnid", "Guinardia_striata", "Phaeocystis"]

all_classes = ["mix", "detritus", "Leptocylindrus", "mix_elongated", "Chaetoceros", "dino30", "Rhizosolenia", "Guinardia_delicatula", 
"Cerataulina", "Cylindrotheca", "Skeletonema", "Ciliate_mix", "Dactyliosolen", "Thalassiosira", "bad", "Dinobryon", "Corethron", 
"DactFragCerataul", "Thalassionema", "Ditylum", "pennate", "Prorocentrum", "Pseudonitzschia", "Mesodinium_sp", "G_delicatula_parasite", 
"Tintinnid", "Guinardia_striata", "Phaeocystis", "Dictyocha", "Pleurosigma", "Eucampia", "Thalassiosira_dirty", 
"Asterionellopsis", "flagellate_sp3", "Laboea_strobila", "Chaetoceros_didymus_flagellate", "Heterocapsa_triquetra", "Guinardia_flaccida", 
"Chaetoceros_pennate", "Ceratium", "Euglena", "Coscinodiscus", "Strombidium_morphotype1", "Paralia", "Gyrodinium", "Ephemera", "Pyramimonas_longicauda", 
"Proterythropsis_sp", "Gonyaulax", "kiteflagellates", "Chrysochromulina", "Chaetoceros_didymus", "bead", "Katodinium_or_Torodinium", "Leptocylindrus_mediterraneus", 
"spore", "Tontonia_gracillima", "Delphineis", "Dinophysis", "Strombidium_morphotype2", "Licmophora", "Lauderia", "clusterflagellate", "Strobilidium_morphotype1", 
"Leegaardiella_ovalis", "pennate_morphotype1", "amoeba", "Strombidium_inclinatum", "Pseudochattonella_farcimen", "Amphidinium_sp", "dino_large1", 
"Strombidium_wulffi", "Chaetoceros_flagellate", "Strombidium_oculatum", "Cerataulina_flagellate", "Emiliania_huxleyi", "Pleuronema_sp", "Strombidium_conicum",
 "Odontella", "Protoperidinium", "zooplankton", "Stephanopyxis", "Tontonia_appendiculariformis", "Strombidium_capitatum", "Bidulphia", "Euplotes_sp", 
 "Parvicorbicula_socialis", "bubble", "Hemiaulus", "Didinium_sp", "pollen", "Tiarina_fusus", "Bacillaria", "Cochlodinium", "Akashiwo", "Karenia"]

classes_30 = ["Asterionellopsis", "bad", "Chaetoceros", "Chaetoceros_flagellate", "Ciliate_mix", "Corethron", "Cylindrotheca", "Dictyocha","dino30", "detritus",
	"Dinobryon", "Ditylum", "Eucampia", "flagellate_sp3", "Guinardia_delicatula", "Guinardia_flaccida", "Guinardia_striata", "Heterocapsa_triquetra", "Laboea_strobila", "Leptocylindrus",
	"pennate", "Phaeocystis", "Pleurosigma", "Prorocentrum", "Pseudonitzschia", "Skeletonema", "Thalassionema", "Thalassiosira", "Thalassiosira_dirty", "Tintinnid"]

print(len(classes_30))

#pp = Preprocessor(years, include_classes=classes, train_eg_per_class=HP.number_of_images_per_class)
#pp = Preprocessor(years, include_classes=all_classes, train_eg_per_class=HP.number_of_images_per_class, thresholding=HP.thresholding)
pp = Preprocessor(years, include_classes=classes_30, train_eg_per_class=HP.number_of_images_per_class, thresholding=HP.thresholding, maxN = HP.maxN, minimum = HP.minimum)


pp.create_datasets([0.6,0.2,0.2])

trainLoader = pp.get_loaders('train', HP.batch_size)
validLoader = pp.get_loaders('validation', HP.batch_size)
testLoader = pp.get_loaders('test', HP.batch_size)


trainer = Trainer(HP_version = HP.version, epochs = HP.number_of_epochs, loss_fn = HP.loss_function, 
	optimizer = HP.optimizer, scheduler = HP.scheduler, lr = HP.learning_rate, momentum = HP.momentum, useCuda=True)

model = GoogleNet()

trainAcc, validAcc, epochs = trainer.train(model, trainLoader, validLoader, earlyStopping = HP.es)

# - or -
#path_to_statedict = "../GoogleNet_1.2-4.0.tar"

#checkpoint = torch.load(path_to_statedict)
#model.load_state_dict(checkpoint['model_state_dict'])

#model.eval()

test_pred, test_target, test_fnames = trainer.test(model, testLoader)
valid_pred, valid_target, valid_fnames = trainer.test(model, validLoader)
train_pred, train_target, train_fnames = trainer.test(model, trainLoader)

test_met = Metrics(test_target, test_pred)
#valid_met = Metrics(valid_target, valid_pred)
#train_met = Metrics(train_target, train_pred)

print(test_met.accuracy())
time = trainer.getTime()
print(time)

f= open("stats-"+str(model)+"-"+str(HP.version)+".json","w+")
str_to_write = "{\"Time\": "+ time +",\n \"Epochs\": "+str(epochs)+ ",\n \"TrainAcc\": "+ str(trainAcc)+",\n \"ValidAcc\": "+str(validAcc)+",\n \"TestAcc\": "+str(test_met.accuracy()) + \
",\n \"Train_Pred\": " + str(list(train_pred.cpu().numpy())) + ",\n \"Train_Target\": " + str(list(train_target.cpu().numpy())) + ",\n \"Train_fnames\": " + str(train_fnames) + \
",\n \"Valid_Pred\": " + str(list(valid_pred.cpu().numpy())) + ",\n \"Valid_Target\": " + str(list(valid_target.cpu().numpy())) + ",\n \"Valid_fnames\": " + str(valid_fnames) + \
",\n \"Test_Pred\": " + str(list(test_pred.cpu().numpy())) + ",\n \"Test_Target\": " + str(list(test_target.cpu().numpy())) + ",\n \"Test_fnames\": " + str(test_fnames) +"}"
f.write(str_to_write)
f.close()


