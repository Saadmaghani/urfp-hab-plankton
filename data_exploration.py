import os

data_folder = "data"
years = [str(i) for i in range(2006, 2014)]
class_names = ["Akashiwo", "Amphidinium sp", "Chrysochromulina", "Cochlodinium", 
	"Dinophysis", "Gonyaulax", "Guinardia_delicatula", "Guinardia_striata", "Gyrodinium", 
	"Heterocapsa_triquetra", "Karenia", "Phaeocystis", "Prorocentrum", "Pseudochattonella_farcimen", 
	"non-HAB-causing"]

ignored_classes = ["Chaetoceros_other", "diatom_flagellate", "G_delicatula_detritus", 
	"other_interaction", "pennates_on_diatoms",]


class_stats = {c_name:{ year:[0] for year in years} for c_name in class_names}

#another approach is to get all classes and get its stats
all_classes = [name for name in os.listdir(data_folder+"/"+years[0]) if os.path.isdir(data_folder+"/"+years[0]+"/"+name)]
classB_stats = {name: {year: [0] for year in years} for name in all_classes}

for year in years:
	data_path = data_folder+"/"+year

	if os.path.isdir(data_path):
		non_hab_causing = [0]
		for class_name in os.listdir(data_path) and class_name not in ignored_classes:
			c_path = data_path + "/"+class_name


			if os.path.isdir(c_path):
				if class_name in class_names:
					class_stats[class_name][year][0] = len([x for x in os.listdir(c_path) if ".png" in x]) 
				else:
					non_hab_causing[0] += len([x for x in os.listdir(c_path) if ".png" in x]) 

				if class_name in all_classes:
					classB_stats[class_name][year][0] = len([x for x in os.listdir(c_path) if ".png" in x]) 

		class_stats["non-HAB-causing"][year] = non_hab_causing

print(class_stats)

print("#############################")
print(classB_stats)