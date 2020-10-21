import datetime
import pathlib
import pandas as pd

def create_experiment_folder():
	now = datetime.datetime.now()
	year = '{:04d}'.format(now.year)
	month = '{:02d}'.format(now.month)
	day = '{:02d}'.format(now.day)
	minute = '{:02d}'.format(now.minute)
	hour = '{:02d}'.format(now.hour)
	timestamp = year + "_" + month + "_" + day + "_" + hour + "_" + minute
	foldername = timestamp
	directory_name = "results/" + foldername 
	pathlib.Path(directory_name).mkdir(parents=True, exist_ok=True) 
	return directory_name


def setup_folders_and_summary_files(exp, datasets, embedding_methods):
	result_folder = create_experiment_folder()

	save_experiment_summary(result_folder,exp,datasets, embedding_methods)

	if(exp["link_prediction"]):
		link_prediction_folder = result_folder + "/link_prediction"
		pathlib.Path(link_prediction_folder).mkdir(parents=True, exist_ok=True)

		col_names_link_prediction =  ['embedding_method', 'dataset', 'run_number', 'auc_score']
		df  = pd.DataFrame(columns = col_names_link_prediction)
		df.to_csv(f'{link_prediction_folder}/link_prediction_results.csv',index=False)

	if(exp["node_classification"]):
		node_classification_folder = result_folder + "/node_classification"
		pathlib.Path(node_classification_folder).mkdir(parents=True, exist_ok=True)

		col_names_node_classification =  ['embedding_method', 'dataset', 'run_number', 'acc_score']
		df  = pd.DataFrame(columns = col_names_node_classification)
		df.to_csv(f'{node_classification_folder}/node_classification_results.csv',index=False)

	if(exp["node_clustering"]):
		node_clustering_folder = result_folder + "/node_clustering"
		pathlib.Path(node_clustering_folder).mkdir(parents=True, exist_ok=True)

		col_names_node_clustering =  ['embedding_method', 'dataset', 'run_number', 'nmi_score']
		df  = pd.DataFrame(columns = col_names_node_clustering)
		df.to_csv(f'{node_clustering_folder}/node_clustering_results.csv',index=False)

	return result_folder

def save_experiment_summary(result_folder,exp, datasets, embedding_methods):
	with open(result_folder + '/experiment_setup_summary.txt', 'w') as file:
		file.write("## Experiment Setup Summary ## \n\n")
		file.write("Datasets used:\n")
		for ds in datasets:
			file.write(f'- {ds}\n')
		file.write("\n")
		file.write("Embedding methods used:\n")
		for em in embedding_methods:
			file.write(f'{em.get_method_summary()}\n')
		file.write("\n")
		file.write("Experiment Parameters:\n")
		for key, value in exp.items():
			file.write(f'- {key}: {value}\n')
