import torch
from preprocessing import Preprocessor
from training import Trainer, load_partial_model, load_full_model
from metrics import Metrics
import torch.nn as nn
from models.vgg_TL import GoogleNet
from configuration import Hyperparameters as HP
import json
import sys

years = [str(y) for y in range(2006, 2015)]

classes_20 = ["detritus", "Leptocylindrus", "Chaetoceros", "Rhizosolenia", "Guinardia_delicatula", "Cerataulina", "Cylindrotheca",
    "Skeletonema", "Dactyliosolen", "Thalassiosira", "Dinobryon", "Corethron", "Thalassionema", "Ditylum", "pennate", "Prorocentrum",
    "Pseudonitzschia", "Tintinnid", "Guinardia_striata", "Phaeocystis"]

classes_all = ["detritus", "Leptocylindrus", "mix_elongated", "Chaetoceros", "dino30", "Rhizosolenia", "Guinardia_delicatula", 
    "Cerataulina", "Cylindrotheca", "Skeletonema", "Ciliate_mix", "Dactyliosolen", "Thalassiosira", "bad", "Dinobryon", "Corethron", 
    "DactFragCerataul", "Thalassionema", "Ditylum", "pennate", "Prorocentrum", "Pseudonitzschia", "Mesodinium_sp", "G_delicatula_parasite", 
    "Tintinnid", "Guinardia_striata", "Phaeocystis", "Dictyocha", "Pleurosigma", "Eucampia", "Thalassiosira_dirty", "Asterionellopsis", 
    "flagellate_sp3", "Laboea_strobila", "Chaetoceros_didymus_flagellate", "Heterocapsa_triquetra", "Guinardia_flaccida", "Chaetoceros_pennate",
    "Ceratium", "Euglena", "Coscinodiscus", "Strombidium_morphotype1", "Paralia", "Gyrodinium", "Ephemera", "Pyramimonas_longicauda", 
    "Proterythropsis_sp", "Gonyaulax", "kiteflagellates", "Chrysochromulina", "Chaetoceros_didymus", "bead", "Katodinium_or_Torodinium", 
    "Leptocylindrus_mediterraneus", "spore", "Tontonia_gracillima", "Delphineis", "Dinophysis", "Strombidium_morphotype2", "Licmophora", 
    "Lauderia", "clusterflagellate", "Strobilidium_morphotype1", "Leegaardiella_ovalis", "pennate_morphotype1", "amoeba", "Strombidium_inclinatum", 
    "Pseudochattonella_farcimen", "Amphidinium_sp", "dino_large1", "Strombidium_wulffi", "Chaetoceros_flagellate", "Strombidium_oculatum", 
    "Cerataulina_flagellate", "Emiliania_huxleyi", "Pleuronema_sp", "Strombidium_conicum","Odontella", "Protoperidinium", "zooplankton", 
    "Stephanopyxis", "Tontonia_appendiculariformis", "Strombidium_capitatum", "Bidulphia", "Euplotes_sp", "Parvicorbicula_socialis", 
    "bubble", "Hemiaulus", "Didinium_sp", "pollen", "Tiarina_fusus", "Bacillaria", "Cochlodinium", "Akashiwo", "Karenia"]

classes_31 = ["Asterionellopsis", "bad", "Chaetoceros", "Ciliate_mix", "Corethron", "Cylindrotheca", "Dictyocha","dino30", "detritus", 
    "Mesodinium_sp", "Chaetoceros_flagellate", "Dinobryon", "Ditylum", "Eucampia", "flagellate_sp3", "Guinardia_delicatula", "Guinardia_flaccida", 
    "Guinardia_striata", "Heterocapsa_triquetra", "Laboea_strobila", "Leptocylindrus","pennate", "Phaeocystis", "Pleurosigma", "Prorocentrum", 
    "Pseudonitzschia", "Skeletonema", "Thalassionema", "Thalassiosira", "Thalassiosira_dirty", "Tintinnid"]

classes_30_cf = ["Asterionellopsis", "bad", "Chaetoceros", "Ciliate_mix", "Corethron", "Cylindrotheca", "Dictyocha","dino30", "detritus", 
    "Chaetoceros_flagellate", "Dinobryon", "Ditylum", "Eucampia", "flagellate_sp3", "Guinardia_delicatula", "Guinardia_flaccida", "Guinardia_striata",
    "Heterocapsa_triquetra", "Laboea_strobila", "Leptocylindrus","pennate", "Phaeocystis", "Pleurosigma", "Prorocentrum", "Pseudonitzschia", 
    "Skeletonema", "Thalassionema", "Thalassiosira", "Thalassiosira_dirty", "Tintinnid"]

classes_30_ms = ["Asterionellopsis", "bad", "Chaetoceros", "Ciliate_mix", "Corethron", "Cylindrotheca", "Dictyocha","dino30", "detritus", 
    "Mesodinium_sp", "Dinobryon", "Ditylum", "Eucampia", "flagellate_sp3", "Guinardia_delicatula", "Guinardia_flaccida", "Guinardia_striata", 
    "Heterocapsa_triquetra", "Laboea_strobila", "Leptocylindrus","pennate", "Phaeocystis", "Pleurosigma", "Prorocentrum", "Pseudonitzschia", 
    "Skeletonema", "Thalassionema", "Thalassiosira", "Thalassiosira_dirty", "Tintinnid"]

classes_vae = ['detritus']

classes_hkust = ["Akashiwo", "Asterionellopsis glacialis", "Cerataulina", "Ceratoperidinium", "Chaetoceros", "Ciliophora", 
"Corethron criophilum", "Cryptophyceae", "Cyanobacteria", "Dactyliosolen", "Dactyliosolen phuketensis or Guinardia striata", 
"Detonula_spp", "Dictyocha fibula", "Dictyocha Octonaria", "Dinophysis", "Disc_shape_Diatom", "Eucampia_spp", "Gonyaulax", 
"Guinardia", "Heterosigma akashiwo", "Lauderia annualata", "Leptocylindrus", "Mesodinium", "Nanoflagellate", "Neodelphineis", 
"OtherDiatom_Chain", "OtherDino_L10um", "OtherPennate_Single", "Prorocentrum", "Prorocentrum gracile", "Prorocentrum triestinum", 
"Psedonitzschia", "Pseudochattonella", "Rhizosolenia imbricata", "Scrippsiella", "Skeletonema", "Thalassionema", "Triposfurca", 
"Triposfusus", "Triposmuelleri", "Tripos_spp", "Vicicitu"]
    
print(len(classes_hkust))


#pp = Preprocessor(years, include_classes=classes, train_eg_per_class=HP.number_of_images_per_class)
#pp = Preprocessor(years, include_classes=all_classes, train_eg_per_class=HP.number_of_images_per_class, thresholding=HP.thresholding)
pp = Preprocessor(years, include_classes=classes_hkust, strategy = HP.pp_strategy, train_eg_per_class=HP.number_of_images_per_class, maxN=HP.maxN,
    minimum=HP.minimum, transformations = HP.transformations, database="HKUST")


pp.create_datasets(HP.data_splits)

trainLoader = pp.get_loaders('train', HP.batch_size)
validLoader = pp.get_loaders('validation', HP.batch_size)
testLoader = pp.get_loaders('test', HP.batch_size)

trainer = Trainer(HP_version=HP.version, epochs=HP.number_of_epochs, loss_fn=HP.loss_function, optimizer=HP.optimizer, scheduler=HP.scheduler, lr=HP.learning_rate, momentum=HP.momentum, autoencoder=HP.train_AE)


# training autoencoder
#model = Simple_AE()
#model = CNN_VAE()

# training normal model
model = GoogleNet()

# training autoencoder + model
"""
ae_model = Simple_AE()
path_to_ae = "models/Simple_AE_3.0-10.2.pth"
if ".tar" in path_to_ae:
    ae_model = load_partial_model(ae_model, path_to_ae)
else:
    ae_model = load_full_model(ae_model, path_to_ae)
model = GoogleNet(autoencoder = ae_model)
"""

# training
trainAcc = []
validAcc = [] 
epochs = 0 
other_stats = {}
trainAcc, validAcc, epochs, other_stats = trainer.train(model, trainLoader, validLoader, earlyStopping = HP.es)

# Just Testing
# model = GoogleNet(v=5.3)
# path_to_statedict = "models/GoogleNet_5.3-13.31.pth"

# if ".tar" in path_to_statedict:
#     model = load_partial_model(model, path_to_statedict)
# else:
#     model = load_full_model(model, path_to_statedict)

# # further training of model
# trainAcc, validAcc, epochs = trainer.train(model, trainLoader, validLoader, earlyStopping=HP.es)
#
#
# #testing autoencoder
#
# #test_sumsqs, test_fnames = trainer.test_autoencoder(model, testLoader)
# #test_acc = test_sumsqs.tolist()
# #test_acc = torch.mean(test_sumsqs).tolist()
#
#
# testing normal model
test_pred, test_target, test_fnames, _ = trainer.test(model, testLoader)
valid_pred, valid_target, valid_fnames, _ = trainer.test(model, validLoader)
train_pred, train_target, train_fnames, _ = trainer.test(model, trainLoader)


test_met = Metrics(test_target, test_pred)
valid_met = Metrics(valid_target, valid_pred)
train_met = Metrics(train_target, train_pred)

test_acc = test_met.accuracy()

print(test_acc)

time = trainer.getTime()
print(time)

f = open("./stats/stats-" + str(model) + "-" + str(HP.version) + ".json", "w+")


str_to_write = "{\"Time\": \"" + time + "\",\n \"Epochs\": " + str(epochs) + ",\n \"TrainAcc\": " + str(trainAcc) + ",\n \"ValidAcc\": " + str(validAcc) + ",\n \"TestAcc\": " + str(test_acc) + \
               ",\n \"Test_Pred\": " + str(list(test_pred.cpu().numpy())) + ",\n \"Test_Target\": " + str(list(test_target.cpu().numpy())) + ",\n \"Test_fnames\": " + json.dumps(test_fnames) + \
               "}"

# ",\n \"Train_Pred\": " + str(train_pred.tolist()) + ",\n \"Train_Target\": " + str(list(train_target.cpu().numpy())) + ",\n \"Train_fnames\": " + json.dumps(train_fnames) + ",\n \"Train_dropped_fnames\": " + json.dumps(train_dropped_fnames) + \
# ",\n \"Valid_Pred\": " + str(list(valid_pred.cpu().numpy())) + ",\n \"Valid_Target\": " + str(list(valid_target.cpu().numpy())) + ",\n \"Valid_fnames\": " + json.dumps(valid_fnames) + ",\n \"Valid_dropped_fnames\": " + json.dumps(valid_dropped_fnames) + \
# ",\n \"Test_dropped_fnames\": " + json.dumps(test_dropped_fnames) + ",\n \"Test_dropped_outs\": " + str(test_extras['all_outs'][1].tolist()) + ",\n \"Test_dropped_confs\": " + str(test_extras['all_confs'][1].tolist()) + \


# ",\n \"Tr_Trgt_Time\": "+ str(other_stats["Tr_Targ_time"]) + ",\n \"Tr_Pred_Time\": "+ str(other_stats["Tr_Pred_time"]) + \

# ",\n \"loss\": "+ str(other_stats["loss"]) + ",\n \"class_loss\": "+ str(other_stats["class_loss"]) + \
# ",\n \"avg_confidence\": " + str(other_stats["avg_confidence"]) + ",\n \"train_drop\": " + str(other_stats["train_drop"])+ ",\n \"valid_drop\": " + str(other_stats["valid_drop"]) + \

f.write(str_to_write)
f.close()

# for training lots of models - use bash script caller
# if len(sys.argv) == 3:

#     i = int(sys.argv[2])
#     thresh = float(sys.argv[1])

#     print("tresh:", thresh, " | i:", i)

#     config_version = str(HP.version)
#     config_version = config_version.replace("121", "12" + str(i + 1))
# else:
#     config_version = str(HP.version)
#     thresh = HP.model_conf

# model.threshold = thresh
# #testing confidenceloss version:
# test_pred, test_target, test_fnames, test_extras = trainer.test(model, testLoader, return_softmax=True, return_confs=True)
# test_fnames, test_dropped_fnames = test_fnames
# # valid_pred, valid_target, valid_fnames, _ = trainer.test(model, validLoader)
# # valid_fnames, valid_dropped_fnames = valid_fnames
# # train_pred, train_target, train_fnames, _ = trainer.test(model, trainLoader)
# # train_fnames, train_dropped_fnames = train_fnames

# # cf_train_pp = Preprocessor(strategy="confident_images", conf_fnames=train_fnames, transformations=HP.transformations)
# # cf_valid_pp = Preprocessor(strategy="confident_images", conf_fnames=valid_fnames, transformations=HP.transformations)
# cf_test_pp = Preprocessor(strategy="confident_images", conf_fnames=test_fnames, transformations=HP.transformations)

# # cf_train_pp.create_cf_datasets()
# # cf_train_loader = cf_train_pp.get_cf_loaders(HP.batch_size)
# #
# # cf_valid_pp.create_cf_datasets()
# # cf_valid_loader = cf_valid_pp.get_cf_loaders(HP.batch_size)

# cf_test_pp.create_cf_datasets()
# cf_test_loader = cf_test_pp.get_cf_loaders(HP.batch_size)


# new_model = GoogleNet(v=1.2)
# path_to_statedict = "models/GoogleNet_1.2-4.2.pth"

# if ".tar" in path_to_statedict:
#     new_model = load_partial_model(new_model, path_to_statedict)
# else:
#     new_model = load_full_model(new_model, path_to_statedict)


# # cf_trainer = Trainer(HP_version=config_version, epochs=HP.number_of_epochs, loss_fn=nn.BCELoss, optimizer=HP.optimizer, scheduler=HP.scheduler, lr=HP.learning_rate, momentum=HP.momentum)
# # trainAcc, validAcc, epochs, other_stats = cf_trainer.train(new_model, cf_train_loader, cf_valid_loader, earlyStopping=HP.es, save=False)


# # testing normal model
# test_pred, test_target, test_fnames, _ = trainer.test(new_model, cf_test_loader)
# # cf_test_pred, cf_test_target, cf_test_fnames, _ = cf_trainer.test(new_model, cf_test_loader)

# test_met = Metrics(test_target, test_pred)
# # cf_test_met = Metrics(cf_test_target, cf_test_pred)

# test_acc = test_met.accuracy()
# # cf_test_acc = cf_test_met.accuracy()

# print(test_acc)
# # print(cf_test_acc)

# time = trainer.getTime()
# print(time)

# f = open("./stats/stats-"+str(new_model)+"-"+str(config_version)+".json", "w+")

# str_to_write = "{\"Time\": \""+ time +"\",\n \"Epochs\": "+str(epochs)+ ",\n \"TrainAcc\": "+ str(trainAcc)+",\n \"ValidAcc\": "+str(validAcc)+",\n \"TestAcc\": " + str(test_acc)  + \
# ",\n \"Test_Pred\": " + str(list(test_pred.cpu().numpy())) + ",\n \"Test_Target\": " + str(list(test_target.cpu().numpy())) + ",\n \"Test_fnames\": " + json.dumps(test_fnames) + \
# "}"

# # + ",\n \"CF_TestAcc\": " + str(cf_test_acc)
# # ",\n \"CF_Test_Pred\": " + str(list(cf_test_pred.cpu().numpy())) + ",\n \"CF_Test_Target\": " + str(list(cf_test_target.cpu().numpy())) + ",\n \"CF_Test_fnames\": " + json.dumps(cf_test_fnames) + \


# f.write(str_to_write)
# f.close()


# new line: to release cuda memory.
torch.cuda.empty_cache()

