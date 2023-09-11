import time
from ist_utilis import *
from random import shuffle
import torch.nn as nn
import argparse
import numpy as np
import google_speech_data_loader as speech_dataset
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import idr_torch

class DNNGoogleSpeechBatchNorm3Layer(nn.Module):
    def __init__(self, partition_num=1, sample_size=4096, model_size=4096, label_num=35):
        super(DNNGoogleSpeechBatchNorm3Layer, self).__init__()
        device = torch.device('cuda')
        self.partition_num = partition_num
        self.partition_dim = model_size // partition_num
        self.temp_hidden_layer_index1 = [i for i in range(model_size)]
        self.temp_hidden_layer_index2 = [i for i in range(model_size)]
        self.fc1 = nn.Linear(sample_size, model_size, False).to(device)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False).to(device)
        self.fc2 = nn.Linear(model_size, model_size, False).to(device)
        self.bn2 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False).to(device)
        self.fc3 = nn.Linear(model_size, label_num, False).to(device)
        self.bn3 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False).to(device)
        # The following is used for distributed training.
        if partition_num != 1:
            self.hidden_layer_index_log1 = []
            self.fc1_weight_partition = []
            self.bn1_weight_partition = []
            self.bn1_bias_partition = []
            self.hidden_layer_index_log2 = []
            self.fc2_weight_partition = []
            self.bn2_weight_partition = []
            self.bn2_bias_partition = []
            self.fc3_weight_partition = []

        self.s1=[None for i in range(self.partition_num)]
        self.s2=[None for i in range(self.partition_num)]
        self.s3=[None for i in range(self.partition_num)]
        self.s4=[None for i in range(self.partition_num)]
        self.s5=[None for i in range(self.partition_num)]
        self.s6=[None for i in range(self.partition_num)]
        self.s7=[None for i in range(self.partition_num)]
        self.r1=None
        self.r2=None
        self.r3=None
        self.r4=None
        self.r5=None
        self.r6=None
        self.r7=None

        self.r8=[None for i in range(self.partition_num)]
        self.r9=[None for i in range(self.partition_num)]
        self.r10=[None for i in range(self.partition_num)]
        self.r11=[None for i in range(self.partition_num)]
        self.r12=[None for i in range(self.partition_num)]
        self.r13=[None for i in range(self.partition_num)]
        self.r14=[None for i in range(self.partition_num)]
        self.s8=None
        self.s9=None
        self.s10=None
        self.s11=None
        self.s12=None
        self.s13=None
        self.s14=None


            
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def partition_to_list(self):
#        print("Repartition parameters!")
        device = torch.device('cuda')
        shuffle(self.temp_hidden_layer_index1)
        self.hidden_layer_index_log1.clear()
        for i in range(self.partition_num):
            current_indexes = torch.tensor(
                self.temp_hidden_layer_index1[i * self.partition_dim:(i + 1) * self.partition_dim]).to(device)
            self.hidden_layer_index_log1.append(current_indexes)

        shuffle(self.temp_hidden_layer_index2)
        self.hidden_layer_index_log2.clear()
        for i in range(self.partition_num):
            current_indexes = torch.tensor(
                self.temp_hidden_layer_index2[i * self.partition_dim:(i + 1) * self.partition_dim]).to(device)
            self.hidden_layer_index_log2.append(current_indexes)

        self.fc1_weight_partition.clear()
        self.bn1_weight_partition.clear()
        self.bn1_bias_partition.clear()
        self.fc2_weight_partition.clear()
        self.bn2_weight_partition.clear()
        self.bn2_bias_partition.clear()
        self.fc3_weight_partition.clear()

        self.fc1_weight_partition = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.hidden_layer_index_log1)
        self.bn1_weight_partition, self.bn1_bias_partition = partition_BN_layer(
            self.bn1.weight, self.bn1.bias, self.hidden_layer_index_log1)
        self.fc2_weight_partition = partition_FC_layer_by_dim_01(
            self.fc2.weight, self.hidden_layer_index_log2, self.hidden_layer_index_log1)
        self.bn2_weight_partition, self.bn2_bias_partition = partition_BN_layer(
            self.bn2.weight, self.bn2.bias, self.hidden_layer_index_log2)
        self.fc3_weight_partition = partition_FC_layer_by_input_dim_1(
            self.fc3.weight, self.hidden_layer_index_log2)

    def flush(self):
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_0(self.bn1.weight.data, self.bn1_weight_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_0(self.bn1.bias.data, self.bn1_bias_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_01(self.fc2.weight.data, self.fc2_weight_partition,
                                             self.hidden_layer_index_log2, self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_0(self.bn2.weight.data, self.bn2_weight_partition,
                                            self.hidden_layer_index_log2)
        update_tensor_by_update_lists_dim_0(self.bn2.bias.data, self.bn2_bias_partition,
                                            self.hidden_layer_index_log2)
        update_tensor_by_update_lists_dim_1(self.fc3.weight.data, self.fc3_weight_partition,
                                            self.hidden_layer_index_log2)


def dispatch_model_to_workers(args, partitioned_model, iter, raw_model=None):
    #print('dispatch_model_to_workers called')
    if args.rank == 0:
        assert(raw_model is not None)
        if(iter==0):
            raw_model.partition_to_list()


        for i in range(1,args.world_size):
            if((raw_model.s1[i] == None or raw_model.s1[i].is_completed()) and
               (raw_model.s2[i] == None or raw_model.s2[i].is_completed()) and
               (raw_model.s3[i] == None or raw_model.s3[i].is_completed()) and
               (raw_model.s4[i] == None or raw_model.s4[i].is_completed()) and
               (raw_model.s5[i] == None or raw_model.s5[i].is_completed()) and
               (raw_model.s6[i] == None or raw_model.s6[i].is_completed()) and
               (raw_model.s7[i] == None or raw_model.s7[i].is_completed())
                    ):
                raw_model.s1[i] = dist.isend(tensor=raw_model.fc1_weight_partition[i], dst=i)
                raw_model.s2[i] = dist.isend(tensor=raw_model.fc2_weight_partition[i], dst=i)
                raw_model.s3[i] = dist.isend(tensor=raw_model.fc3_weight_partition[i], dst=i)
                raw_model.s4[i] = dist.isend(tensor=raw_model.bn1_weight_partition[i], dst=i)
                raw_model.s5[i] = dist.isend(tensor=raw_model.bn1_bias_partition[i], dst=i)
                raw_model.s6[i] = dist.isend(tensor=raw_model.bn2_weight_partition[i], dst=i)
                raw_model.s7[i] = dist.isend(tensor=raw_model.bn2_bias_partition[i], dst=i)
 

    else:

        if((partitioned_model.r1 == None or partitioned_model.r1.is_completed()) and
           (partitioned_model.r2 == None or partitioned_model.r2.is_completed()) and
           (partitioned_model.r3 == None or partitioned_model.r3.is_completed()) and
           (partitioned_model.r4 == None or partitioned_model.r4.is_completed()) and
           (partitioned_model.r5 == None or partitioned_model.r5.is_completed()) and
           (partitioned_model.r6 == None or partitioned_model.r6.is_completed()) and
           (partitioned_model.r7 == None or partitioned_model.r7.is_completed())
             ):

            partitioned_model.r1 = dist.irecv(tensor=partitioned_model.fc1.weight.data, src=0)
            partitioned_model.r2 = dist.irecv(tensor=partitioned_model.fc2.weight.data, src=0)
            partitioned_model.r3 = dist.irecv(tensor=partitioned_model.fc3.weight.data, src=0)
            partitioned_model.r4 = dist.irecv(tensor=partitioned_model.bn1.weight.data, src=0)
            partitioned_model.r5 = dist.irecv(tensor=partitioned_model.bn1.bias.data, src=0)
            partitioned_model.r6 = dist.irecv(tensor=partitioned_model.bn2.weight.data, src=0)
            partitioned_model.r7 = dist.irecv(tensor=partitioned_model.bn2.bias.data, src=0)

        

def push_model_to_parameter_server(args, partitioned_model, raw_model=None):
    #print('push_model_to_parameter_server called!')
    if args.rank == 0:
        assert(raw_model is not None)


        for i in range(1,args.world_size):
            if((raw_model.r8[i] == None or raw_model.r8[i].is_completed()) and
               (raw_model.r9[i] == None or raw_model.r9[i].is_completed()) and
               (raw_model.r10[i] == None or raw_model.r10[i].is_completed()) and
               (raw_model.r11[i] == None or raw_model.r11[i].is_completed()) and
               (raw_model.r12[i] == None or raw_model.r12[i].is_completed()) and
               (raw_model.r13[i] == None or raw_model.r13[i].is_completed()) and
               (raw_model.r14[i] == None or raw_model.r14[i].is_completed())
            ):

                raw_model.r8[i] = dist.irecv(tensor=raw_model.fc1_weight_partition[i], src=i)
                raw_model.r9[i] = dist.irecv(tensor=raw_model.fc2_weight_partition[i], src=i)
                raw_model.r10[i] = dist.irecv(tensor=raw_model.fc3_weight_partition[i], src=i)
                raw_model.r11[i] = dist.irecv(tensor=raw_model.bn1_weight_partition[i], src=i)
                raw_model.r12[i] = dist.irecv(tensor=raw_model.bn1_bias_partition[i], src=i)
                raw_model.r13[i] = dist.irecv(tensor=raw_model.bn2_weight_partition[i], src=i)
                raw_model.r14[i] = dist.irecv(tensor=raw_model.bn2_bias_partition[i], src=i)

        

    else:

        if((partitioned_model.s8 == None or partitioned_model.s8.is_completed()) and
           (partitioned_model.s9 == None or partitioned_model.s9.is_completed()) and
           (partitioned_model.s10 == None or partitioned_model.s10.is_completed()) and
           (partitioned_model.s11 == None or partitioned_model.s11.is_completed()) and
           (partitioned_model.s12 == None or partitioned_model.s12.is_completed()) and
           (partitioned_model.s13 == None or partitioned_model.s13.is_completed()) and
           (partitioned_model.s14 == None or partitioned_model.s14.is_completed()) 
        ):

            partitioned_model.s8 = dist.isend(tensor=partitioned_model.fc1.weight.data, dst=0)
            partitioned_model.s9 = dist.isend(tensor=partitioned_model.fc2.weight.data, dst=0)
            partitioned_model.s10 = dist.isend(tensor=partitioned_model.fc3.weight.data, dst=0)
            partitioned_model.s11 = dist.isend(tensor=partitioned_model.bn1.weight.data, dst=0)
            partitioned_model.s12 = dist.isend(tensor=partitioned_model.bn1.bias.data, dst=0)
            partitioned_model.s13 = dist.isend(tensor=partitioned_model.bn2.weight.data, dst=0)
            partitioned_model.s14 = dist.isend(tensor=partitioned_model.bn2.bias.data, dst=0)



        
def train(args, partitioned_model, raw_model, optimizer, train_loader, epoch, train_time_log):
    start_time = time.time()
    device = torch.device('cuda')
    partitioned_model.train()
    if args.rank == 0:
        raw_model.train()
    for i, batch in enumerate(train_loader):
        if i < len(train_loader) // args.world_size:
#            if i % args.repartition_iter == 0:
            if epoch-1 == 0:
                if args.rank == 0:
                    dispatch_model_to_workers(args, partitioned_model, epoch-1, raw_model)
                else:
                    dispatch_model_to_workers(args, partitioned_model,epoch-1)
            data, target = batch['wav'].float().to(device), batch['label'].to(device)
            optimizer.zero_grad()
            output = partitioned_model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()  # This will just update the local data which reduces communication overhead.
            i += 1
            train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()

            if (i + 1) % args.repartition_iter == 0 or i == len(train_loader) // args.world_size:
                if args.rank == 0:
                    push_model_to_parameter_server(args, partitioned_model, raw_model)
                    raw_model.flush()
                else:
                    push_model_to_parameter_server(args, partitioned_model)
        else:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time



def test(args, raw_model, test_loader, epoch, test_loss_log, test_acc_log):
    # currently only do test on rank0 node.
    assert(args.rank == 0)
    device = torch.device('cuda')
    raw_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            data, target = batch['wav'].float().to(device), batch['label'].to(device)
            output = raw_model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()
            test_total += target.shape[0]
        test_acc = float(test_correct) / float(test_total)
        test_loss /= float(test_total)



    print("Epoch {} Test Loss: {:.6f}; Test Accuracy: {:.2f}.\n".format(epoch, test_loss, test_acc))
     


def worker_process(args):
    assert(args.rank != 0)

    torch.cuda.set_device(idr_torch.local_rank)
    device = torch.device('cuda')
    
    train_set = speech_dataset.train_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    partitioned_model = DNNGoogleSpeechBatchNorm3Layer(partition_num=1,
                                                       model_size=args.model_size // args.world_size).to(device)
    optimizer = torch.optim.SGD(partitioned_model.parameters(), lr=args.lr)
    epochs = args.epochs * args.world_size
    train_time_log = np.zeros(epochs)


    epoch=1
    while(True):
        train(args, partitioned_model, None, optimizer, train_loader, epoch, train_time_log)
        epoch+=1



def parameter_server_process(args):
    assert (args.rank == 0)

    torch.cuda.set_device(idr_torch.local_rank)
    device = torch.device('cuda')
    

    train_set = speech_dataset.train_dataset()
    test_set = speech_dataset.test_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=False)
    model_name = 'DNN_speech_3_layer_BN_' + str(args.epochs) + '_' + str(args.model_size) \
                 + '_cascaded_' + str(args.world_size) + '_' + str(args.repartition_iter)

    raw_model = DNNGoogleSpeechBatchNorm3Layer(partition_num=args.world_size,
                                               model_size=args.model_size).to(device)
    partitioned_model = DNNGoogleSpeechBatchNorm3Layer(partition_num=1,
                                                       model_size=args.model_size//args.world_size).to(device)
    optimizer = torch.optim.SGD(partitioned_model.parameters(), lr=args.lr)
    epochs = args.epochs #* args.world_size
    train_time_log = np.zeros(epochs)
    test_loss_log = np.zeros(epochs)
    test_acc_log = np.zeros(epochs)


    for epoch in range(1, epochs + 1):
        train(args, partitioned_model, raw_model, optimizer, train_loader, epoch, train_time_log)
        if(True):
            test(args, raw_model, test_loader, epoch, test_loss_log, test_acc_log)








def main():
    parser = argparse.ArgumentParser(description='PyTorch 3-layer DNN on google speech dataset (subnet single PS)')
    parser.add_argument('--dist-backend', type=str, default='gloo', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=1, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='partition group (default: 2)')
    parser.add_argument('--model-size', type=int, default=4096, metavar='N',
                        help='model size for intermediate layers (default: 4096)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--repartition-iter', type=int, default=20, metavar='N',
                        help='keep model in local update mode for how many iteration (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01 for BN)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()


    dist.init_process_group(backend="mpi")

    size = dist.get_world_size()
    rank = dist.get_rank()
    args.rank = rank
    args.world_size = size

    print("rank",args.rank)
    print("world",args.world_size)

    
    torch.manual_seed(args.seed)

    if args.rank == 0:
        parameter_server_process(args)
    else:
        worker_process(args)


if __name__ == '__main__':
    main()
