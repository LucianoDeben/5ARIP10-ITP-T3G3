from src.preprocessing import get_transforms, get_datasets, get_dataloaders, add_vessel_contrast
import torch
from src.model import TACEnet
import random
from src.drr import create_drr
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def loadData():
    transform = get_transforms(resize_shape= [512, 512, 96],contrast_value=1000)
    
    train_ds, val_ds = get_datasets(
    root_dir="../data081",
    collection="HCC-TACE-Seg",
    transform=transform,
    download=False,
    val_frac=0.2,
    download_len=2
    )

    train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=1)

    return train_loader, val_loader


def buildModel():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define model
    model = TACEnet()

    # Move model to device
    model.to(device)

    # Initialize the criterion, DICE loss with cross entropy uses binary cross entropy when there are only two classes
    criterion = torch.nn.MSELoss()

    # Initialize the optimizer, use config to set hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    return model, criterion, optimizer, device

def sampleVolume(train_loader):
    # Sample a batch of data from the dataloader
    batch = next(iter(train_loader))

    # Separate the image and segmentation from the batch
    volumes, target = batch["image"], batch["seg"]
    volume = add_vessel_contrast(volumes, target, contrast_value=4000)

    return volume, target

def trainSingleEpoch(model, criterion, optimizer, Volume, Target, device, length, verbose=False):
    
    model.train()
    for i in range(length):
        rotation = random.uniform(-40,40)
        # Create raw DRR
        drr_raw = create_drr(
            Volume[0],
            Target[0],
            bone_attenuation_multiplier=5.0,
            sdd=1020,
            height=256,
            width=256,
            rotations=torch.tensor([[rotation, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 850.0, 0.0]]),
            mask_to_channels=True,
            device="cpu",
            )
        

        enhancement_factor = random.uniform(0.4,0.6)
        drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor, device)

        prediction, _ = model(Target.to(device), drr_input)
        loss = criterion(prediction, drr_vessels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            print(f"[{i+1}/{length}] - Loss: {loss.item()}")

def testModel(model, criterion, Volume, Target, device):
    model.eval()
    loss = []
    
    for i in range(5):
        rotation = (i*20-40)
        drr_raw = create_drr(
            Volume[0],
            Target[0],
            bone_attenuation_multiplier=5.0,
            sdd=1020,
            height=256,
            width=256,
            rotations=torch.tensor([[rotation, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 850.0, 0.0]]),
            mask_to_channels=True,
            device="cpu",
            )
        drr_body = drr_raw[:,0,:,:]
        drr_vessels = drr_raw[:,1,:,:]

        drr_body = (drr_body/(torch.max(drr_body.flatten()))).to(device)
        drr_vessels = ((drr_vessels/(torch.max(drr_vessels.flatten())))*2).to(device)
        enhancement_factor = 0.5
        drr_combined = (drr_body + enhancement_factor*drr_vessels).unsqueeze(0)

        prediction, _ = model(Target.to(device), drr_combined.to(device))


        loss.append(criterion(prediction+drr_body, drr_vessels+drr_body).item())

    print("-----------------------------------")
    print(f"[TEST] - Loss: {np.mean(loss)}")
    print("-----------------------------------")


def trainModel(epochs,length, save, load, path, verbose=False):
    train_loader, _ = loadData()
    model, criterion, optimizer, device = buildModel()
    volume, target = sampleVolume(train_loader)

    if load:
        model.load_state_dict(torch.load(path))

    for i in range(epochs):
        trainSingleEpoch(model,criterion,optimizer,volume,target,device,length,verbose)
        #testModel(model,criterion,volume,target,device)
        visualizeModel(model,volume,target,device)
    
    if save:
        torch.save(model.state_dict(), path)


def fetchResults(path):
    train_loader, _ = loadData()
    model, criterion, optimizer, device = buildModel()
    volume, target = sampleVolume(train_loader)

    model.load_state_dict(torch.load(path))

    visualizationResult(model, volume, target, device)
    loss = rotationSweep(model, volume, target, device, criterion)

    return loss

def testInferenceSpeed(path):
    train_loader, _ = loadData()
    model, criterion, optimizer, device = buildModel()
    volume, target = sampleVolume(train_loader)

    model.load_state_dict(torch.load(path))

    drr_raw = create_drr(
            volume[0],
            target[0],
            bone_attenuation_multiplier=5.0,
            sdd=1020,
            height=256,
            width=256,
            rotations=torch.tensor([[0.0, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 850.0, 0.0]]),
            mask_to_channels=True,
            device="cpu",
            )
    
    drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor=0.6, device=device)

    start = timer()

    print(target.size())
    print(drr_input.size())
    
    prediction, latent_representation = model(target.to(device), drr_input)
    end = timer()

    return (end-start)



def visualizeModel(model, Volume, Target, device):
    rotation = random.uniform(-40,40)
    drr_raw = create_drr(
            Volume[0],
            Target[0],
            bone_attenuation_multiplier=5.0,
            sdd=1020,
            height=256,
            width=256,
            rotations=torch.tensor([[rotation, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 850.0, 0.0]]),
            mask_to_channels=True,
            device="cpu",
            )
    
    drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor=0.5, device=device)
    
    prediction, latent_representation = model(Target.to(device), drr_input)

    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

    # Plot each image in a subplot
    axs[0].imshow(drr_input.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)
    axs[0].set_title('DRR')

    axs[1].imshow(drr_vessels.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)
    axs[1].set_title('enhanced target')

    axs[2].imshow(prediction.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)
    axs[2].set_title('AI Enhanced')

    axs[3].imshow(latent_representation.cpu().detach().numpy().squeeze(), cmap = 'gray')
    axs[3].set_title('Latent Representation')

    # Hide the axes labels
    for ax in axs:
        ax.axis('off')
    plt.show()

def drr_to_input(drr_raw, enhancement_factor, device):
    drr_body = drr_raw[:,0,:,:]
    drr_vessels = drr_raw[:,1,:,:]

    #drr_body = (drr_body/(torch.max(drr_body.flatten()))).to(device)
    #drr_vessels = ((drr_vessels/(torch.max(drr_vessels.flatten())))*2).to(device)
    drr_input = (drr_body + enhancement_factor*drr_vessels).unsqueeze(0)
    
    drr_vessels = drr_vessels.unsqueeze(0)
    #print(f"input size: {drr_input.size()}")
    #print(f"target size: {drr_vessels.size()}")

    return drr_input.to(device), drr_vessels.to(device)

def visualizationResult(model, Volume, Target, device):
    model.eval()

    rotation = random.uniform(-40,40)
    drr_raw = create_drr(
            Volume[0],
            Target[0],
            bone_attenuation_multiplier=5.0,
            sdd=1020,
            height=256,
            width=256,
            rotations=torch.tensor([[rotation, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 850.0, 0.0]]),
            mask_to_channels=True,
            device="cpu",
            )
    drr_body = drr_raw[:,0,:,:].to(device)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()


    drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor=0.6, device=device)
    prediction, latent_representation = model(Target.to(device), drr_input)
    AI_enhanced = drr_body+prediction

    axs[0].imshow(drr_input.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)
    axs[4].imshow(AI_enhanced.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)

    drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor=0.5, device=device)
    prediction, latent_representation = model(Target.to(device), drr_input)
    AI_enhanced = drr_body+prediction

    axs[1].imshow(drr_input.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)
    axs[5].imshow(AI_enhanced.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)

    drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor=0.4, device=device)
    prediction, latent_representation = model(Target.to(device), drr_input)
    AI_enhanced = drr_body+prediction

    axs[2].imshow(drr_input.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)
    axs[6].imshow(AI_enhanced.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)

    drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor=0.1, device=device)
    prediction, latent_representation = model(Target.to(device), drr_input)
    AI_enhanced = drr_body+prediction

    axs[3].imshow(drr_input.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)
    axs[7].imshow(AI_enhanced.cpu().detach().numpy().squeeze(), cmap='gray', vmax = 40)

    for i in range(8):
        axs[i].axis('off')


    plt.show()


def rotationSweep(model, Volume, Target, device, criterion):
    total_loss = []
    #EF sweep
    for j in range(10):
        
        ef = j/10
        loss = []
        #Rotation sweep
        for i in range(80):
            drr_raw = create_drr(
                Volume[0],
                Target[0],
                bone_attenuation_multiplier=5.0,
                sdd=1020,
                height=256,
                width=256,
                rotations=torch.tensor([[i-40, 0.0, 0.0]]),
                translations=torch.tensor([[0.0, 850.0, 0.0]]),
                mask_to_channels=True,
                device="cpu",
                )
            
            drr_input, drr_vessels = drr_to_input(drr_raw, enhancement_factor=ef, device=device)
            prediction, latent_representation = model(Target.to(device), drr_input)
            loss.append(criterion(prediction, drr_vessels).item())
            print(f"loop: {j}, angle: {i}")
        
        print(f"loss of loop {j}: {np.mean(loss)}")
        total_loss.append(np.mean(loss))

    
    return total_loss