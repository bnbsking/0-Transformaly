### Modes:
+ Annotation
	+ D1: _data/example/train/
	+ D2: _data/example/train/
	+ S1: train.py: train model M, get distribution-1 features
	+ S2: eval.py: use M to get distribution-2 features, compute score for D1,D2,D1*D2
	+ S1 and S2: trainset (=valset) normal, testset all
+ Overall flow:
	+ Developing: D1S1 -> D1S2 -> D2S1 -> D2S2
	+ Production: D2S1 -> D2S2

### File structures
+ _code/
	+ run.sh # run D1S1 or D1S2 or D2S1 or D2S2
	+ result.ipynb
+ _data/
	+ example/
		+ train/ # D1
			+ cats/*.jpg # abnormal
			+ dogs/*.jpg # normal
		+ test/ # D2
			+ cats/*.jpg # abnormal
			+ dogs/*.jpg # normal
+ experiments/ # auto-generated results by S1 & S2
	+ multimodal/cats_vs_dogs/
	+ class_0/
	+ extracted_features_d1/ # D1S1
				+ train_pretrained_ViT_features.npy      # normal features pretrained
				+ test_pretrained_Vit_features.npy       # all features pretrained
				+ test_pretrained_samples_likelihood.npy # distribution-1 scores
			+ feature_distance_d1/  # D1S2
				+ train_finetuned_features.npy           # normal pretrained CE(student,teacher)
				+ full_test_finetuned_score.npy          # all pretrained CE(student,teacher)
				+ test_pretrained_samples_likelihood.npy # distribution-2 scores
			+ extracted_features_d2/*npy # D2S1, same as above
			+ feature_distance_d2/*.npy  # D2S2, same as above
			+ model/ # D1S1
				+ best_full_finetuned_model_state_dict.pkl # best model and loaded by S2 for feature_distance
				+ last_full_finetuned_model_state_dict.pkl
				+ *_recon_model_state_dict.pkl
				+ *_graph.png  # loss plot
				+ *_losses.pkl # loss record
			+ *_outputs.log # log for S1,S2
		+ summarize_results/cats_vs_dogs/ # eval
		+ d1_cats_vs_dogs_results.csv # D1S2
			+ d2_cats_vs_dogs_results.csv # D2S2
+ train.py # S1
+ eval.py  # S2

### Modifications
+ train.py
+ line 77 -> iterate 1 only. only class0 is abnormal, train once only
+ line 171,217-218 -> no replicate a model for the best, save best directly (utils.py line 430-433) instead
+ line 207-210 -> add arg path_to_save
+ utils.py
+ line 109 -> add D1 or D2 dataset_index
+ line 130-133 -> _classes = [0] (anomaly class), iterate 1 class only
	+ line 219 -> add arg dataset_index
	+ line 252 -> continue skip KNN
	+ line 392, 400 -> skip validation loop since validation=train always
	+ line 430-433 -> save best model
	+ line 760-764 -> arg to_show = False
+ eval.py
	+ line 28-29 -> add arg dataset_index
	+ line 121,126 -> dataset_index
	+ line 158-159 -> save pretrained score
	+ line 179,202-204,209,218,237,240 -> dataset_index
	+ line 257-258 -> save finetuned score
	+ line 287 -> dataset_index
