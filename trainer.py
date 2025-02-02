import torch
import argparse
import os
from dataloader import create_dataloader
from model import VoiceFilter
from utils import spec2wav, MyWriter
from mir_eval.separation import bss_eval_sources
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel

def validate(model, testloader, criterion, writer, epoch):
    model.eval()
    test_total_loss = 0.0
    total_sdr = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in testloader:
            dvec, target_wav, mixed_wav,target_spect, mixed_spect, mixed_phase = batch[0]
            target_spect, mixed_spect = target_spect.unsqueeze(0).cuda(), mixed_spect.unsqueeze(0).cuda()
            
            dvec = dvec.unsqueeze(0).cuda()
            dvec = dvec.detach()
            
            estimated_mask = model(mixed_spect, dvec)
            estimated_spect = mixed_spect * estimated_mask
            test_loss = criterion(estimated_spect, target_spect).item()
            test_total_loss += test_loss
            batch_count += 1

            mixed_spect = mixed_spect[0].cpu().detach().numpy()
            target_spect = target_spect[0].cpu().detach().numpy()
            estimated_spect = estimated_spect[0].cpu().detach().numpy()
            estimated_mask = estimated_mask[0].cpu().detach().numpy()
            
            estimated_wav = spec2wav(estimated_spect, mixed_phase)
            mini_length = min(len(target_wav), len(estimated_wav))
            target_wav, estimated_wav = target_wav[:mini_length], estimated_wav[:mini_length]
            sdr = bss_eval_sources(target_wav, estimated_wav, False)[0][0]
            total_sdr += sdr
            
        avg_test_loss = test_total_loss / batch_count if batch_count > 0 else float('nan')
        avg_sdr = total_sdr / batch_count if batch_count > 0 else float('nan')
        writer.log_evaluation(avg_test_loss, avg_sdr, mixed_wav, target_wav, estimated_wav, mixed_spect.T, target_spect.T, estimated_spect.T, estimated_mask.T, epoch+1)

def train(trainloader, testloader, model, optimizer, criterion, writer, chkpt_path):
    epoch = 0   
    total_step = 0 
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs!")

    while True:
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for devcs, target_spects, mixed_spects in trainloader:
            target_spects, mixed_spects = target_spects.cuda(), mixed_spects.cuda()
            
            dvec_list = list()
            for dvec in devcs:
                dvec_list.append(dvec)
            dvec = torch.stack(dvec_list, dim=0).cuda()
            dvec = dvec.detach()
            
            mask = model(mixed_spects, dvec)
            output = mixed_spects * mask
            loss = criterion(output, target_spects)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            total_step += 1
            print("Wrote summary at step %d" % total_step)

        avg_loss = running_loss / batch_count
        writer.add_scalar('Training Loss/epoch', avg_loss, epoch+1)
        validate(model, testloader, criterion, writer, epoch)
        print("Saving model at epoch %d" % (epoch+1))
        torch.save(model.state_dict(), os.path.join(chkpt_path, f"model_epoch_{epoch+1}.pt"))
        print("Model saved!")
        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./chkpt')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name') 
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('-l','--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--train_path', type=str, default='./tiny_dataset/train')
    parser.add_argument('--test_path', type=str, default='./tiny_dataset/test')
    args = parser.parse_args()
    
    chkpt_path = os.path.join(args.checkpoint_path, args.model)
    os.makedirs(chkpt_path, exist_ok=True)
    log_dir=os.path.join(chkpt_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    writer = MyWriter(log_dir)
    
    trainloader = create_dataloader(args, train=True)
    testloader = create_dataloader(args, train=False)
    
    # load model
    model = VoiceFilter().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion 
    criterion = torch.nn.MSELoss()

    train(trainloader, testloader, model, optimizer, criterion, writer, chkpt_path)