import torch
from preprocessing import Preprocessor
from training import Trainer, load_partial_model, load_full_model
from metrics import Metrics
import torch.nn as nn
import torch.optim as optim
#from models.triple_arch import N_Parallel_Models
from models.vgg_TL import GoogleNet
#from models.autoencoders import Simple_AE, CNN_VAE
from configuration import Hyperparameters as HP
import json
import math

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

print(len(classes_all))


#pp = Preprocessor(years, include_classes=classes, train_eg_per_class=HP.number_of_images_per_class)
#pp = Preprocessor(years, include_classes=all_classes, train_eg_per_class=HP.number_of_images_per_class, thresholding=HP.thresholding)
pp = Preprocessor(years, include_classes=classes_30_cf, strategy = HP.pp_strategy, train_eg_per_class=HP.number_of_images_per_class, maxN = HP.maxN, 
    minimum =  HP.minimum, transformations = HP.transformations)


pp.create_datasets(HP.data_splits)

trainLoader = pp.get_loaders('train', HP.batch_size)
validLoader = pp.get_loaders('validation', HP.batch_size)
testLoader = pp.get_loaders('test', HP.batch_size)

trainer = Trainer(HP_version = HP.version, epochs = HP.number_of_epochs, loss_fn = HP.loss_function, optimizer = HP.optimizer, 
    scheduler = HP.scheduler, lr = HP.learning_rate, momentum = HP.momentum, autoencoder=HP.train_AE)


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
#trainAcc, validAcc, epochs, other_stats = trainer.train(model, trainLoader, validLoader, earlyStopping = HP.es)

# Just Testing
model = GoogleNet()
path_to_statedict = "models/GoogleNet_5.3-13.31.pth"

if ".tar" in path_to_statedict:
    model = load_partial_model(model, path_to_statedict)
else:
    model = load_full_model(model, path_to_statedict)

# further training of model
#trainAcc, validAcc, epochs = trainer.train(model, trainLoader, validLoader, earlyStopping = HP.es)


#testing autoencoder

#test_sumsqs, test_fnames = trainer.test_autoencoder(model, testLoader)
#test_acc = test_sumsqs.tolist()
#test_acc = torch.mean(test_sumsqs).tolist()


if isinstance(HP.model_conf, list):
    for i, thresh in enumerate(HP.model_conf):

        model.threshold = thresh
        #testing confidenceloss version:
        test_pred, test_target, test_fnames, test_extras = trainer.test(model, testLoader, return_softmax=True, return_confs=True)
        test_fnames, test_dropped_fnames = test_fnames
        valid_pred, valid_target, valid_fnames, _ = trainer.test(model, validLoader)
        valid_fnames, valid_dropped_fnames = valid_fnames
        train_pred, train_target, train_fnames, _ = trainer.test(model, trainLoader)
        train_fnames, train_dropped_fnames = train_fnames

        # testing normal model
        #test_pred, test_target, test_fnames = trainer.test(model, testLoader)
        #valid_pred, valid_target, valid_fnames = trainer.test(model, validLoader)
        #train_pred, train_target, train_fnames = trainer.test(model, trainLoader)


        test_met = Metrics(test_target, test_pred)
        #valid_met = Metrics(valid_target, valid_pred)
        #train_met = Metrics(train_target, train_pred)

        test_acc = test_met.accuracy()

        print(test_acc)

        time = trainer.getTime()
        print(time)
        config_version = str(HP.version)
        config_version = config_version.replace("121", "12"+str(i+1))
        f = open("./stats/stats-"+str(model)+"-"+str(config_version)+".json","w+")

        #str(test_met.accuracy()) + \

        str_to_write = "{\"Time\": \""+ time +"\",\n \"Epochs\": "+str(epochs)+ ",\n \"TrainAcc\": "+ str(trainAcc)+",\n \"ValidAcc\": "+str(validAcc)+",\n \"TestAcc\": "+ str(test_acc) + \
        ",\n \"Train_Pred\": " + str(train_pred.tolist()) + ",\n \"Train_Target\": " + str(list(train_target.cpu().numpy())) + ",\n \"Train_fnames\": " + json.dumps(train_fnames) + ",\n \"Train_dropped_fnames\": " + json.dumps(train_dropped_fnames) + \
        ",\n \"Valid_Pred\": " + str(list(valid_pred.cpu().numpy())) + ",\n \"Valid_Target\": " + str(list(valid_target.cpu().numpy())) + ",\n \"Valid_fnames\": " + json.dumps(valid_fnames) + ",\n \"Valid_dropped_fnames\": " + json.dumps(valid_dropped_fnames) + \
        ",\n \"Test_Pred\": " + str(list(test_pred.cpu().numpy())) + ",\n \"Test_Target\": " + str(list(test_target.cpu().numpy())) + ",\n \"Test_fnames\": " + json.dumps(test_fnames) + \
        ",\n \"Test_dropped_fnames\": " + json.dumps(test_dropped_fnames) + ",\n \"Test_dropped_outs\": " + str(test_extras['all_outs'][1].tolist()) + ",\n \"Test_dropped_confs\": " + str(test_extras['all_confs'][1].tolist()) + \
        "}"

        #",\n \"Tr_Trgt_Time\": "+ str(other_stats["Tr_Targ_time"]) + ",\n \"Tr_Pred_Time\": "+ str(other_stats["Tr_Pred_time"]) + \

        #",\n \"loss\": "+ str(other_stats["loss"]) + ",\n \"class_loss\": "+ str(other_stats["class_loss"]) + \
        #",\n \"avg_confidence\": " + str(other_stats["avg_confidence"]) + ",\n \"train_drop\": " + str(other_stats["train_drop"])+ ",\n \"valid_drop\": " + str(other_stats["valid_drop"]) + \

        f.write(str_to_write)
        f.close()
else:
    model.threshold = HP.model_conf
    #testing confidenceloss version:
    test_pred, test_target, test_fnames, test_extras = trainer.test(model, testLoader, return_softmax=True, return_confs=True)
    test_fnames, test_dropped_fnames = test_fnames
    valid_pred, valid_target, valid_fnames, _ = trainer.test(model, validLoader)
    valid_fnames, valid_dropped_fnames = valid_fnames
    train_pred, train_target, train_fnames, _ = trainer.test(model, trainLoader)
    train_fnames, train_dropped_fnames = train_fnames

    if HP.version >= 15:
        new_model = GoogleNet(v=1.2)

        conf_trainLoader = pp.confident_imgs(train_fnames, HP.batch_size, transformations = HP.transformations)
        conf_validLoader = pp.confident_imgs(valid_fnames, HP.batch_size, transformations = HP.transformations)
        conf_testLoader = pp.confident_imgs(test_fnames, HP.batch_size, transformations = HP.transformations)


        conf_trainer = Trainer(HP_version = HP.version, epochs = HP.number_of_epochs, loss_fn = nn.BCELoss, optimizer = HP.optimizer, scheduler = HP.scheduler, lr = HP.learning_rate, momentum = HP.momentum)



        trainAcc, validAcc, epochs, other_stats = conf_trainer.train(new_model, conf_trainLoader, conf_validLoader, earlyStopping = HP.es)


        # testing normal model
        test_pred, test_target, test_fnames, _ = trainer.test(model, conf_testLoader)
        #valid_pred, valid_target, valid_fnames = trainer.test(model, validLoader)
        #train_pred, train_target, train_fnames = trainer.test(model, trainLoader)


        test_met = Metrics(test_target, test_pred)
        #valid_met = Metrics(valid_target, valid_pred)
        #train_met = Metrics(train_target, train_pred)

        test_acc = test_met.accuracy()

        print(test_acc)

        time = trainer.getTime()
        print(time)

        f = open("./stats/stats-"+str(model)+"-"+str(HP.version)+".json","w+")

        #str(test_met.accuracy()) + \

        str_to_write = "{\"Time\": \""+ time +"\",\n \"Epochs\": "+str(epochs)+ ",\n \"TrainAcc\": "+ str(trainAcc)+",\n \"ValidAcc\": "+str(validAcc)+",\n \"TestAcc\": "+ str(test_acc) + \
        ",\n \"Test_Pred\": " + str(list(test_pred.cpu().numpy())) + ",\n \"Test_Target\": " + str(list(test_target.cpu().numpy())) + ",\n \"Test_fnames\": " + json.dumps(test_fnames) + \
        "}"
        
        #",\n \"Train_Pred\": " + str(train_pred.tolist()) + ",\n \"Train_Target\": " + str(list(train_target.cpu().numpy())) + ",\n \"Train_fnames\": " + json.dumps(train_fnames) + ",\n \"Train_dropped_fnames\": " + json.dumps(train_dropped_fnames) + \
        #",\n \"Valid_Pred\": " + str(list(valid_pred.cpu().numpy())) + ",\n \"Valid_Target\": " + str(list(valid_target.cpu().numpy())) + ",\n \"Valid_fnames\": " + json.dumps(valid_fnames) + ",\n \"Valid_dropped_fnames\": " + json.dumps(valid_dropped_fnames) + \
        #",\n \"Test_dropped_fnames\": " + json.dumps(test_dropped_fnames) + ",\n \"Test_dropped_outs\": " + str(test_extras['all_outs'][1].tolist()) + ",\n \"Test_dropped_confs\": " + str(test_extras['all_confs'][1].tolist()) + \
        

        #",\n \"Tr_Trgt_Time\": "+ str(other_stats["Tr_Targ_time"]) + ",\n \"Tr_Pred_Time\": "+ str(other_stats["Tr_Pred_time"]) + \

        #",\n \"loss\": "+ str(other_stats["loss"]) + ",\n \"class_loss\": "+ str(other_stats["class_loss"]) + \
        #",\n \"avg_confidence\": " + str(other_stats["avg_confidence"]) + ",\n \"train_drop\": " + str(other_stats["train_drop"])+ ",\n \"valid_drop\": " + str(other_stats["valid_drop"]) + \

        f.write(str_to_write)
        f.close()

# new line: to release cuda memory.
torch.cuda.empty_cache()

