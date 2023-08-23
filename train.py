"""
Transformaly Training Script
"""
import argparse
import logging
import pickle
import os

import torch.nn
from utils import print_and_add_to_log, get_datasets_for_ViT, \
    Identity, freeze_finetuned_model, train, plot_graphs, \
    extract_fetures
from os.path import join
from pytorch_pretrained_vit.model import AnomalyViT, ViT

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset',default='cifar10')
    parser.add_argument('--data_path', default='./data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('--lr', default=0.0001,
                        help='Learning rate value')
    parser.add_argument('--eval_every', type=int, default=2,
                        help='Will evaluate the model ever <eval_every> epochs')
    parser.add_argument('--unimodal', default=False, action='store_true',
                        help='Use the unimodal settings')
    parser.add_argument('--plot_every_layer_summarization', default=False, action='store_true',
                        help='plot the per layer AUROC')
    parser.add_argument('--dataset_index', type=str, default="d1",
                        help='dataset index')
    parser_args = parser.parse_args()
    args = vars(parser_args)


    args['use_layer_outputs'] = list(range(2, 12))
    args['use_imagenet'] = True
    BASE_PATH = 'experiments'

    if args['dataset'] == 'cifar10':
        _classes = range(10)
    elif args['dataset'] == 'fmnist':
        _classes = range(10)
    elif args['dataset'] == 'cifar100':
        _classes = range(20)
    elif args['dataset'] == 'cats_vs_dogs':
        _classes = range(2)
    elif args['dataset'] == 'dior':
        _classes = range(19)
    else:
        raise ValueError(f"Does not support the {args['dataset']} dataset")
    # create the relevant directories
    if not os.path.exists(
            join(BASE_PATH,
                 f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}')):
        os.makedirs(join(BASE_PATH,
                         f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}'))

    logging.basicConfig(
        filename=join(BASE_PATH,
                      f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}',
                      f'Train_{args["dataset"]}_Transformaly_outputs.log'), level=logging.DEBUG)

    print_and_add_to_log("========================================================", logging)
    print_and_add_to_log("Args are:", logging)
    print_and_add_to_log(args, logging)
    print_and_add_to_log("========================================================", logging)
    results = {'class': [],
               'pretrained_AUROC_scores': [],
               'all_layers_finetuned_AUROC_scores': [],
               'pretrained_and_finetuned_AUROC_scores': []}



    #for _class in _classes: # alternate each class to be abnormal for multi-modal # so modify class 0 as abnormal
    for _class in _classes[:1]:
        print_and_add_to_log("===================================", logging)
        print_and_add_to_log(f"Class is : {_class}", logging)
        print_and_add_to_log("===================================", logging)
        args['_class'] = _class
        base_feature_path = join(
            BASE_PATH,
            f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}/class_{_class}')
        model_path = join(base_feature_path, 'model')

        args['base_feature_path'] = base_feature_path # .../experiments/multimodal/cifar10/class_i
        args['model_path'] = model_path               # .../experiments/multimodal/cifar10/class_i/model

        # create the relevant directories
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if args['unimodal']:
            anomaly_classes = [i for i in _classes if i != args['_class']]
        else:
            anomaly_classes = [args['_class']]

        print_and_add_to_log(
            "====================================================================",
            logging)
        print_and_add_to_log(
            "Start Training", logging)
        print_and_add_to_log(
            "====================================================================",
            logging)

        trainset, testset = get_datasets_for_ViT(dataset=args['dataset'],          # "cifar10"
                                                 data_path=args['data_path'],      # "./_data/cifar10/train"
                                                 one_vs_rest=args['unimodal'],     # False
                                                 _class=args['_class'],            # 0
                                                 normal_test_sample_only=True,     # True
                                                 use_imagenet=args['use_imagenet'] # True
                                                 ) # get normal only # trainset==testset if cats_vs_dogs

        _, ood_test_set = get_datasets_for_ViT(dataset=args['dataset'],
                                               data_path=args['data_path'],
                                               one_vs_rest=not args['unimodal'], # only diff
                                               _class=args['_class'],
                                               normal_test_sample_only=True,
                                               use_imagenet=args['use_imagenet']
                                               ) # get abnormal only

        print_and_add_to_log("---------------", logging)
        print_and_add_to_log(f'Class size: {args["_class"]}', logging) # abnormal class # 0
        print_and_add_to_log(f'Trainset size: {len(trainset)}', logging) # 200
        print_and_add_to_log(f'Testset size: {len(testset)}', logging) # 200
        print_and_add_to_log(f'OOD testset size: {len(ood_test_set)}', logging) # 20
        print_and_add_to_log("---------------", logging)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                                 shuffle=False)
        ood_val_loader = torch.utils.data.DataLoader(ood_test_set, batch_size=args['batch_size'],
                                                     shuffle=False)

        dataloaders = {'training': train_loader,
                       'val': val_loader,
                       'test': ood_val_loader
                       }

        # Build model
        if args['use_imagenet']:
            VIT_MODEL_NAME = 'B_16_imagenet1k'
        else:
            VIT_MODEL_NAME = 'B_16'

        if 1 and args['dataset_index']!="d1":
            model = ViT(VIT_MODEL_NAME, pretrained=True)
            model.fc = Identity()
            model.eval()
            model = model.to('cuda')
            extract_fetures(base_path=BASE_PATH,
                        data_path=args['data_path'],
                        datasets=[args['dataset']],
                        model=model,
                        logging=logging,
                        calculate_features=True,
                        unimodal_vals=[args['unimodal']],
                        manual_class_num_range=[_class],
                        output_train_features=True,
                        output_test_features=True,
                        use_imagenet=args['use_imagenet'], di=args['dataset_index'])
            break

        # Build model
        model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
        model.fc = Identity()
        # Build model for best instance
        best_model = AnomalyViT(VIT_MODEL_NAME, pretrained=True) if 0 else torch.nn.Module()
        best_model.fc = Identity()
        if 0:
            print( "*"*10 )
            for name,weight in model.named_parameters():
                print( (name, weight.shape, weight.numel()) )
            print( "*"*10 )

        model.to('cuda')
        best_model.to('cuda')

        model_checkpoint_path = join(model_path, 'last_full_finetuned_model_state_dict.pkl')
        if os.path.exists(model_checkpoint_path): # False
            model_state_dict = torch.load(model_checkpoint_path)
            model.load_state_dict(model_state_dict)
            print_and_add_to_log("model loadded from checkpoint here:", logging)
            print_and_add_to_log(model_checkpoint_path, logging)

        # freeze the model
        freeze_finetuned_model(model)
        model, best_model, cur_acc_loss = train(model=model,
                                                best_model=best_model,
                                                args=args,
                                                dataloaders=dataloaders,
                                                output_path=model_path, # .../experiments/multimodal/cifar10/class_i/model
                                                device='cuda',
                                                seed=42,
                                                model_checkpoint_path=model_checkpoint_path, # .../experiments/multimodal/cifar10/class_i/model/last_full_finetuned_model_state_dict.pkl
                                                anomaly_classes=anomaly_classes # [0]
                                                )
        print("AAA---trainFinished---AAA")
        training_losses = cur_acc_loss['training_losses']
        val_losses = cur_acc_loss['val_losses']
        print( type(training_losses), len(training_losses) ) # list # ep # () # ()
        print( type(val_losses), len(val_losses) ) # list # ep # () # ()
        try:
            if True:
                plot_graphs(training_losses, val_losses, training_losses, val_losses, path_to_save=model_path)
            else:
                plot_graphs(training_losses, val_losses, training_losses, val_losses)

        except Exception as e:
            print_and_add_to_log('raise error:', logging)
            print_and_add_to_log(e, logging)
        print("BBB---finishPlot---BBB")
        # save models
        if 0:
            torch.save(best_model.state_dict(), join(model_path,
                                                 'best_full_finetuned_model_state_dict.pkl'))
        torch.save(model.state_dict(), join(model_path,
                                            'last_full_finetuned_model_state_dict.pkl'))

        # save losses
        with open(join(model_path, 'full_finetuned_training_losses.pkl'), 'wb') as f:
            pickle.dump(training_losses, f)
        with open(join(model_path, 'full_finetuned_val_losses.pkl'), 'wb') as f:
            pickle.dump(val_losses, f)
        if 0:
            continue

        if args['use_imagenet']:
            MODEL_NAME = 'B_16_imagenet1k'
        else:
            MODEL_NAME = 'B_16'

        model = ViT(MODEL_NAME, pretrained=True)
        model.fc = Identity()
        model.eval()
        print(f"CCC---newModel {_class}---CCC")
        if 0:
            print( "*"*10 )
            for name,weight in model.named_parameters():
                print( (name, weight.shape, weight.numel()) )
            print( "*"*10 )

        extract_fetures(base_path=BASE_PATH,
                        data_path=args['data_path'],
                        datasets=[args['dataset']],
                        model=model,
                        logging=logging,
                        calculate_features=True,
                        unimodal_vals=[args['unimodal']],
                        manual_class_num_range=[_class],
                        output_train_features=True,
                        output_test_features=True,
                        use_imagenet=args['use_imagenet'])
    print(f"DDD---END of training ---DDD")
