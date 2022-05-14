import time
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
from ist_utilis import *
import sys
from os import environ


class DNNGoogleSpeechBatchNorm2Layer(nn.Module):
    def __init__(self, partition_num=1, sample_size=4096, model_size=4096, label_num=35):
        super(DNNGoogleSpeechBatchNorm2Layer, self).__init__()
        self.partition_num = partition_num
        self.partition_dim = model_size // partition_num
        self.temp_hidden_layer_index = [i for i in range(model_size)]
        self.fc1 = nn.Linear(sample_size, model_size, False)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, track_running_stats=False)
        self.fc2 = nn.Linear(model_size, label_num, False)
        self.bn2 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False)
        self.partition_num=partition_num
        # The following is used for distributed training.
        
        self.hidden_layer_index_log = []
        self.fc1_weight_partition = []
        self.bn1_weight_partition = []
        self.bn1_bias_partition = []
        self.fc2_weight_partition = []

        self.s1=[None for i in range(self.partition_num)]
        self.s2=[None for i in range(self.partition_num)]
        self.s3=[None for i in range(self.partition_num)]
        self.s4=[None for i in range(self.partition_num)]
        self.r1=None
        self.r2=None
        self.r3=None
        self.r4=None

        self.r5=[None for i in range(self.partition_num)]
        self.r6=[None for i in range(self.partition_num)]
        self.r7=[None for i in range(self.partition_num)]
        self.r8=[None for i in range(self.partition_num)]
        self.s5=None
        self.s6=None
        self.s7=None
        self.s8=None

        print("ICICICICICICICCICICICICICICIC",self.partition_num)


    def forward(self, x):
        x = self.fc1(x)
        # print(x[0])
        x = self.bn1(x)
        # print(x[0])
        x = nn.functional.relu(x, inplace=True)
        # print(x[0])
        x = self.fc2(x)
        # print(x[0])
        x = self.bn2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def partition_to_list(self):
        print("Repartition parameters!")
        shuffle(self.temp_hidden_layer_index)
        self.hidden_layer_index_log.clear()
        for i in range(self.partition_num):
            current_indexes = torch.tensor(
                self.temp_hidden_layer_index[i * self.partition_dim:(i + 1) * self.partition_dim])
            self.hidden_layer_index_log.append(current_indexes)

        self.fc1_weight_partition.clear()
        self.bn1_weight_partition.clear()
        self.bn1_bias_partition.clear()
        self.fc2_weight_partition.clear()
        self.fc1_weight_partition = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.hidden_layer_index_log)
        self.bn1_weight_partition, self.bn1_bias_partition = partition_BN_layer(
            self.bn1.weight, self.bn1.bias, self.hidden_layer_index_log)
        self.fc2_weight_partition = partition_FC_layer_by_input_dim_1(
            self.fc2.weight, self.hidden_layer_index_log)



    def flush(self):
        
        # update the model based on the collected parameters.
        # Here we have to get around pytorch variable by use variable.data,
        # since leaf variable disallowing in-place operation
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition,
                                            self.hidden_layer_index_log)
        update_tensor_by_update_lists_dim_0(self.bn1.weight.data, self.bn1_weight_partition,
                                            self.hidden_layer_index_log)
        update_tensor_by_update_lists_dim_0(self.bn1.bias.data, self.bn1_bias_partition,
                                            self.hidden_layer_index_log)
        update_tensor_by_update_lists_dim_1(self.fc2.weight.data, self.fc2_weight_partition,
                                            self.hidden_layer_index_log)


freq_iter=10

def dispatch_model_to_workers(args, partitioned_model, iter,raw_model=None):
    print('dispatch_model_to_workers called!')
    
    if args.rank == 0:
        assert(raw_model is not None)
        if(iter==0):   #a changer
            raw_model.partition_to_list()

        partitioned_model.fc1.weight.data = raw_model.fc1_weight_partition[0].clone()
        partitioned_model.fc2.weight.data = raw_model.fc2_weight_partition[0].clone()
        partitioned_model.bn1.weight.data = raw_model.bn1_weight_partition[0].clone()
        partitioned_model.bn1.bias.data = raw_model.bn1_bias_partition[0].clone()


        print(len(partitioned_model.s4))
        #sys.exit(0)
        for i in range(1,args.world_size):
            """ dist.send(tensor=raw_model.fc1.weight.data[i],dst=i)
            dist.send(tensor=raw_model.fc2.weight.data[i],dst=i)
            dist.send(tensor=raw_model.bn1.weight.data[i],dst=i)
            dist.send(tensor=raw_model.bn1.bias.data[i],dst=i) """
            if(raw_model.s4[i] != None and (iter%freq_iter)==0):
                raw_model.s1[i].wait()
                raw_model.s2[i].wait()
                raw_model.s3[i].wait()
                raw_model.s4[i].wait()
            if((raw_model.s1[i] == None or raw_model.s1[i].is_completed()) and
                (raw_model.s2[i] == None or raw_model.s2[i].is_completed()) and
                (raw_model.s3[i] == None or raw_model.s3[i].is_completed()) and
                (raw_model.s4[i] == None or raw_model.s4[i].is_completed())     ):
                raw_model.s1[i] = dist.isend(tensor=raw_model.fc1_weight_partition[i], dst=i, tag=100)
                raw_model.s2[i] = dist.isend(tensor=raw_model.fc2_weight_partition[i], dst=i, tag=101)
                raw_model.s3[i] = dist.isend(tensor=raw_model.bn1_weight_partition[i], dst=i, tag=102)
                raw_model.s4[i] = dist.isend(tensor=raw_model.bn1_bias_partition[i], dst=i, tag=103)
                print("DDDDDDDDDDIIIIIIIISSSPPPPPPPAAAAAATTTTTTTTTTTTCCCCCCCCCCCCCHHHHHHHH PrOC O send to ",i)



#        dist.scatter(tensor=partitioned_model.fc1.weight.data, scatter_list=raw_model.fc1_weight_partition, src=0)
#        dist.scatter(tensor=partitioned_model.fc2.weight.data, scatter_list=raw_model.fc2_weight_partition, src=0)
#        dist.scatter(tensor=partitioned_model.bn1.weight.data, scatter_list=raw_model.bn1_weight_partition, src=0)
#        dist.scatter(tensor=partitioned_model.bn1.bias.data, scatter_list=raw_model.bn1_bias_partition, src=0)
    else:
        #sys.exit(0)
        if(partitioned_model.r4 != None and (iter%freq_iter)==0):
           partitioned_model.r1.wait()
           partitioned_model.r2.wait()
           partitioned_model.r3.wait()
           partitioned_model.r4.wait()
           
        if((partitioned_model.r1 == None or partitioned_model.r1.is_completed()) and
            (partitioned_model.r2 == None or partitioned_model.r2.is_completed()) and
            (partitioned_model.r3 == None or partitioned_model.r3.is_completed()) and
            (partitioned_model.r4 == None or partitioned_model.r4.is_completed())     ):

            partitioned_model.r1 = dist.irecv(tensor=partitioned_model.fc1.weight.data, src=0, tag=100)
            partitioned_model.r2 = dist.irecv(tensor=partitioned_model.fc2.weight.data, src=0, tag=101)
            partitioned_model.r3 = dist.irecv(tensor=partitioned_model.bn1.weight.data, src=0, tag=102)
            partitioned_model.r4 = dist.irecv(tensor=partitioned_model.bn1.bias.data, src=0, tag=103)
            print("DDDDDDDDDDIIIIIIIISSSPPPPPPPAAAAAATTTTTTTTTTTTCCCCCCCCCCCCCHHHHHHHH",args.rank,"received from 0")
#        dist.scatter(tensor=partitioned_model.fc1.weight.data, scatter_list=[], src=0)
#        dist.scatter(tensor=partitioned_model.fc2.weight.data, scatter_list=[], src=0)
#        dist.scatter(tensor=partitioned_model.bn1.weight.data, scatter_list=[], src=0)
#        dist.scatter(tensor=partitioned_model.bn1.bias.data, scatter_list=[], src=0)
    


def push_model_to_parameter_server(args, partitioned_model, iter, raw_model=None):
    print('push_model_to_parameter_server called!')
    
    if args.rank == 0:
        assert(raw_model is not None)

        raw_model.fc1_weight_partition[0] = partitioned_model.fc1.weight.data.clone()
        raw_model.fc2_weight_partition[0] = partitioned_model.fc2.weight.data.clone()
        raw_model.bn1_weight_partition[0] = partitioned_model.bn1.weight.data.clone()
        raw_model.bn1_bias_partition[0] = partitioned_model.bn1.bias.data.clone()

        for i in range(1,args.world_size):
            if(raw_model.r8[i] != None and (iter%freq_iter)==0):
                raw_model.r5[i].wait()
                raw_model.r6[i].wait()
                raw_model.r7[i].wait()
                raw_model.r8[i].wait()
            if((raw_model.r5[i] == None or raw_model.r5[i].is_completed()) and
                (raw_model.r6[i] == None or raw_model.r6[i].is_completed()) and
                (raw_model.r7[i] == None or raw_model.r7[i].is_completed()) and
                (raw_model.r8[i] == None or raw_model.r8[i].is_completed())         ):

                raw_model.r5[i] = dist.irecv(tensor=raw_model.fc1_weight_partition[i], src=i, tag=200)
                raw_model.r6[i] = dist.irecv(tensor=raw_model.fc2_weight_partition[i], src=i, tag=201)
                raw_model.r7[i] = dist.irecv(tensor=raw_model.bn1_weight_partition[i], src=i, tag=202)
                raw_model.r8[i] = dist.irecv(tensor=raw_model.bn1_bias_partition[i], src=i, tag=203)
                print("PPPPPPPPPUUUUUUUUUUSSSSSSSSSSHHHHHHHHH proc 0 receives from ",i)

        # dist.gather(tensor=partitioned_model.fc1.weight.data, gather_list=raw_model.fc1_weight_partition, dst=0)
        # dist.gather(tensor=partitioned_model.fc2.weight.data, gather_list=raw_model.fc2_weight_partition, dst=0)
        # dist.gather(tensor=partitioned_model.bn1.weight.data, gather_list=raw_model.bn1_weight_partition, dst=0)
        # dist.gather(tensor=partitioned_model.bn1.bias.data, gather_list=raw_model.bn1_bias_partition, dst=0)
    else:
        if(partitioned_model.s8 != None and (iter%freq_iter)==0):
            partitioned_model.s5.wait()
            partitioned_model.s6.wait()
            partitioned_model.s7.wait()
            partitioned_model.s8.wait() 
        if((partitioned_model.s5 == None or partitioned_model.s5.is_completed()) and
            (partitioned_model.s6 == None or partitioned_model.s6.is_completed()) and
            (partitioned_model.s7 == None or partitioned_model.s7.is_completed()) and
            (partitioned_model.s8 == None or partitioned_model.s8.is_completed())    ):

            partitioned_model.s5 = dist.isend(tensor=partitioned_model.fc1.weight.data, dst=0, tag=200)
            partitioned_model.s6 = dist.isend(tensor=partitioned_model.fc2.weight.data, dst=0, tag=201)
            partitioned_model.s7 = dist.isend(tensor=partitioned_model.bn1.weight.data, dst=0, tag=202)
            partitioned_model.s8 = dist.isend(tensor=partitioned_model.bn1.bias.data, dst=0, tag=203)
            print("PPPPPPPPPPPPPUUUUUUUUUUUSSSSSSSSSSSSSHHHHHHHHHH proc ",args.rank,"send to 0")
            


        # dist.gather(tensor=partitioned_model.fc1.weight.data, gather_list=[], dst=0)
        # dist.gather(tensor=partitioned_model.fc2.weight.data, gather_list=[], dst=0)
        # dist.gather(tensor=partitioned_model.bn1.weight.data, gather_list=[], dst=0)
        # dist.gather(tensor=partitioned_model.bn1.bias.data, gather_list=[], dst=0)
    


def train(args, partitioned_model, raw_model, optimizer, train_loader, epoch, train_time_log):
    start_time = time.time()
    partitioned_model.train()
    if args.rank == 0:
        raw_model.train()

    print("LALALA")

    for i, batch in enumerate(train_loader):
        if i < len(train_loader) // args.world_size:
            if i % args.repartition_iter == 0:
                print("AAAAAAAAAAAAAAAAAAAAAAAAAA",epoch)
                if args.rank == 0:
                    dispatch_model_to_workers(args, partitioned_model, epoch-1, raw_model)
                else:
                    dispatch_model_to_workers(args, partitioned_model,epoch-1)
            data, target = batch['wav'].float(), batch['label']
            optimizer.zero_grad()
            output = partitioned_model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()  # This will just update the local data which reduces communication overhead.
            i += 1
            train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
            if i % args.log_interval == 0:
                print('Node {}: Train Epoch {} iter {} <Loss: {:.6f}, Accuracy: {:.2f}%>'.format(
                    args.rank, epoch, i, loss.item(), 100. * train_correct / target.shape[0]))
            if (i + 1) % args.repartition_iter == 0 or i == len(train_loader) // args.world_size:
                if args.rank == 0:
                    push_model_to_parameter_server(args, partitioned_model, epoch-1, raw_model)
                    raw_model.flush()
                else:
                    push_model_to_parameter_server(args, partitioned_model, epoch-1)
        else:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Node {}: Train Epoch {} total time {:3.2f}s'.format(args.rank, epoch, elapsed_time))
    train_time_log[epoch-1] = elapsed_time


def test(args, raw_model, test_loader, epoch, test_loss_log, test_acc_log):
    # currently only do test on rank0 node.

    #assert(args.rank == 0)
    raw_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            data, target = batch['wav'].float(), batch['label']
            output = raw_model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()
            test_total += target.shape[0]
        test_acc = float(test_correct) / float(test_total)
        test_loss /= float(test_total)
    print("Epoch {} Test Loss: {:.6f}; Test Accuracy: {:.2f}.\n".format(epoch, test_loss, test_acc))
    test_loss_log[epoch - 1] = test_loss
    test_acc_log[epoch - 1] = test_acc


def worker_process(args):
    assert(args.rank != 0)
    print("SLAVE")
    #dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                        rank=args.rank, world_size=args.world_size)

    size = dist.get_world_size()
    rank = dist.get_rank()
    args.rank = rank
    args.world_size = size

    print("worker rank",args.rank)
    print("world",args.world_size)


    device = torch.device('cpu')
    train_set = speech_dataset.train_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    partitioned_model = DNNGoogleSpeechBatchNorm2Layer(partition_num=1,
                                                       model_size=args.model_size // args.world_size).to(device)
    print("RAPH")                                                       
    optimizer = torch.optim.SGD(partitioned_model.parameters(), lr=args.lr)
    epochs = args.epochs
    train_time_log = np.zeros(epochs)
    for epoch in range(1, epochs + 1):
        train(args, partitioned_model, None, optimizer, train_loader, epoch, train_time_log)


def parameter_server_process(args):
    assert (args.rank == 0)
    print("MASTER")
    


    device = torch.device('cpu')
    train_set = speech_dataset.train_dataset()
    test_set = speech_dataset.test_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=False)
    model_name = 'DNN_speech_2_layer_BN_' + str(args.epochs) + '_' + str(args.model_size) \
                 + '_cascaded_' + str(args.world_size) + '_' + str(args.repartition_iter)
    print("we are going to train from scratch.")
    raw_model = DNNGoogleSpeechBatchNorm2Layer(partition_num=args.world_size,
                                               model_size=args.model_size).to(device)
    partitioned_model = DNNGoogleSpeechBatchNorm2Layer(partition_num=1,
                                                       model_size=args.model_size//args.world_size).to(device)
    print("RAPH")                                                       
    optimizer = torch.optim.SGD(partitioned_model.parameters(), lr=args.lr)
    epochs = args.epochs
    train_time_log = np.zeros(epochs)
    test_loss_log = np.zeros(epochs)
    test_acc_log = np.zeros(epochs)
    for epoch in range(1, epochs + 1):
        train(args, partitioned_model, raw_model, optimizer, train_loader, epoch, train_time_log)
        if(True): #epoch%5==0):
        #if(epoch==epochs):
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEESSSSSSSSSSSSSSSSSSSSSSSSSSSTTTTTTTTTTTTTTTTTTTTTTT")
            test(args, raw_model, test_loader, epoch, test_loss_log, test_acc_log)
    np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
    np.savetxt('./log/' + model_name + '_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
    np.savetxt('./log/' + model_name + '_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')
    torch.save(raw_model, './trained_models/' + model_name + '.pth')


def main():
    parser = argparse.ArgumentParser(description='PyTorch 2-layer DNN on google speech dataset (subnet single PS)')
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
                        help='keep model in local update mode for how many iteration (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001 for BN)')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    # assert(torch.cuda.is_available())

   
    dist.init_process_group(backend="mpi")

    size = dist.get_world_size()
    rank = dist.get_rank()
    args.rank = rank
    args.world_size = size

    print("rank",args.rank)
    print("world",args.world_size)

#    environ["TOKENIZERS_PARALLELISM"] = "false"
    
    torch.manual_seed(args.seed)

    if args.rank == 0:
        parameter_server_process(args)
    else:
        worker_process(args)


if __name__ == '__main__':
    main()
