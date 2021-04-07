import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from model_files.EfficientNetModel import *
from utils.utils import *
from torch.utils.data import *
import time
import torch.utils.data.distributed

def train (model, train_loader, valid_loader, optimizer, loss_obj, stochastic=False, run_name=""):
    ## Main training function

    ## Set mixed precision mode
    mixed_precision = True

    ## set warmup / patience
    warmup = 15

    ## learning rate decay
    decay = 0.9

    ## create manual learning rate scheduler
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay, verbose=True)

    print('########################################### Training Run' + run_name +  ' ###########################################')

    ## declare scaler for mixed precision training and inference
    scaler = torch.cuda.amp.GradScaler()


    ## predeclare result containers ##
    train_loss_batch_metric = np.array(0)
    batch_metric = np.array(0)
    batch_count = 0

    train_roc_metric = np.array(0)
    train_loss_metric = np.array(0)
    val_roc_metric = np.array(0)
    val_loss_metric = np.array(0)
    epoch_metric = np.array(0)

    ## training loop for each epoch
    for epoch in range(0,50):
        epoch_time = time.time()

        ### Train for one epoch ###
        total_loss = 0
        total_count = 0

        ## training loop
        model.train() #calc gradients and bn enabled

        ## save all predictions to calculate ROC AUC
        probability_preds = np.array(0)
        true_labels = np.array(0)

        ## train for one epoch
        for idx, batch in enumerate(train_loader):

            batch_count += 1
            batch_size = 0

            ## if training the stochastic models, the generator will create two images of different resolutions
            if stochastic:
                images_lr, image_hr, y = batch
                batch_size = images_lr.shape[0]
                images = (images_lr.cuda(non_blocking=True), image_hr.cuda(non_blocking=True))
            else:
                images, y = batch
                batch_size = images.shape[0]

            true_labels = np.append(true_labels, y.cpu().detach().numpy())
            y = torch.unsqueeze(y, 1)

            ## transfer batch to gpu
            if not stochastic:
                images = images.cuda(non_blocking=True)  # apparently non_blocking makes it faster
            labels = y.cuda(non_blocking=True)

            optimizer.zero_grad()  # set grads to zero before backprop

            ## forward and backpropagation step
            if mixed_precision: ##mixed precision mode
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = loss_obj(output, labels)

                scaler.scale(loss).backward() # backprop
                scaler.step(optimizer)  # step the grads update the weights
                scaler.update()
            else:
                output = model(images)
                loss = loss_obj(output, labels)

                loss.backward() #backprop
                optimizer.step() #step the grads update the weights

            ### store metrics for the batch ##
            probability_preds = np.append(probability_preds, torch.sigmoid(output).cpu().detach().numpy())
            train_loss_batch_metric = np.append(train_loss_batch_metric, loss.item()/batch_size)
            batch_metric = np.append(batch_metric, batch_count)

            total_loss += loss.item()
            total_count += batch_size

        ### store train metrics for the epoch ###
        total_loss = total_loss/float(total_count)
        train_roc_metric = np.append(train_roc_metric, roc_auc_score(true_labels, probability_preds))
        train_loss_metric = np.append(train_loss_metric, total_loss)
        epoch_metric = np.append(epoch_metric, epoch)

        if epoch > warmup:
            lr_sched.step() #step the lr scheduler

        ## Print Performance
        print('Epoch ' + str(epoch) + ' train loss:{loss:.6f}'.format(loss=total_loss) + ' train RoC:{RoC:.4f}'.format(RoC=roc_auc_score(true_labels, probability_preds)) )



        ###### Eval Model ######
        ## Eval against validation set ##

        probability_preds = np.array(0)
        true_labels = np.array(0)

        ## same loop as above ##
        model.eval() ## set model to eval mode
        total_count = 0
        total_loss = 0
        with torch.no_grad(): #dont calc gradients for validation
            for idx, batch in enumerate(valid_loader):
                batch_start = time.time()
                batch_size = 0
                if stochastic:
                    images_lr, image_hr, y = batch
                    batch_size = images_lr.shape[0]
                    images = (images_lr.cuda(non_blocking=True), image_hr.cuda(non_blocking=True))
                else:
                    images, y = batch
                    batch_size = images.shape[0]


                true_labels = np.append(true_labels, y.cpu().detach().numpy())
                y = torch.unsqueeze(y, 1)

                if not stochastic:
                    images = images.cuda(non_blocking=True)  # apparently non_blocking makes it faster

                labels = y.cuda(non_blocking=True)

                optimizer.zero_grad()  # set grads to zero before backprop

                if mixed_precision:  ##mixed precision mode
                    with torch.cuda.amp.autocast():
                        output = model(images)
                        loss = loss_obj(output, labels)
                else:
                    output = model(images)
                    loss = loss_obj(output, labels)

                probability_preds = np.append(probability_preds, torch.sigmoid(output).cpu().detach().numpy())
                total_loss += loss.item()
                total_count += batch_size


        ## Save results for epoch to dataframe ##

        total_loss = total_loss / float(total_count)
        print('Epoch ' + str(epoch) + ' val loss:{loss:.6f}'.format(loss=total_loss) + ' val RoC:{RoC:.4f}'.format(
            RoC=roc_auc_score(true_labels, probability_preds)))

        val_roc_metric = np.append(val_roc_metric, roc_auc_score(true_labels, probability_preds))
        val_loss_metric = np.append(val_loss_metric, total_loss)

        per_epoch_results_dict = {'Epoch': epoch_metric, 'Train Loss':train_loss_metric, 'Val Loss':val_loss_metric,
                                  'Train RoC':train_roc_metric, 'Val RoC':val_roc_metric}
        per_epoch_df = pd.DataFrame(per_epoch_results_dict)
        per_epoch_df.to_csv(run_name + '_per_epoch.csv',index_label=False)

        per_batch_results_dict = {'Batch': batch_metric, 'Train Loss': train_loss_batch_metric}
        per_batch_df = pd.DataFrame(per_batch_results_dict)
        per_batch_df.to_csv(run_name + '_per_batch.csv', index_label=False)


if __name__ == '__main__':
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv('data/1024x1024_train.csv')

    from sklearn.model_selection import train_test_split


    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=1337)

    train_dataset = MelaDS(image_folder='data/1024x1024/', dataframe=train_df, global_res=380, high_res=1024)
    valid_dataset = MelaDS(image_folder='data/1024x1024/', dataframe=valid_df, global_res=380, high_res=1024)
    train_dataset = DataLoader(train_dataset, batch_size=100, num_workers=10, shuffle=True)
    valid_dataset = DataLoader(valid_dataset, batch_size=200, num_workers=10, shuffle=True)

    train_dataset_lr = MelaDS(image_folder='data/1024x1024/', dataframe=train_df, global_res=380, high_res=None)
    valid_dataset_lr = MelaDS(image_folder='data/1024x1024/', dataframe=valid_df, global_res=380, high_res=None)
    train_dataset_lr = DataLoader(train_dataset_lr, batch_size=100, num_workers=12, shuffle=True)
    valid_dataset_lr = DataLoader(valid_dataset_lr, batch_size=100, num_workers=12, shuffle=True)

    print(train_df.describe())
    print(train_df['target'].sum())
    print(valid_df.describe())
    print(valid_df['target'].sum())


    train_iterations = 5

    torch.backends.cudnn.benchmark = True
    for i in range(0, train_iterations):

        ## create model
        model = EffNetb4Stochastic(dropout=0.3, image_size=(380, 380), image_size_hr=(1024, 1024))

        ## declare optimizer
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0,
                                       momentum=0.0, centered=False)

        ## declare loss object
        loss_obj = nn.BCEWithLogitsLoss()

        ## distribute model
        model = nn.DataParallel(model).cuda()

        ## call the training routine
        train(model, train_dataset, valid_dataset, optimizer, loss_obj, stochastic=True,
              run_name='1024x1024ResultsFinal/b4_stochastic_vs_base/b4_stochastic' + str(i))

        ## save model
        torch.save(model,'b4_stochastic_full.pth.tar')

        ## delete model from memory
        del model









