"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image


from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True


# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


if not EVAL:

    # load the features saved from extract_features.py
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=0, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))
    print(features.shape)


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here


    # for each batch, prepare the targets using this torch.nn.utils.rnn function
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    decoder.train()

    for epoch in range(NUM_EPOCHS):

        for batch, (image_features, targets, lenghts) in enumerate(train_loader, 1):

            image_features, targets = image_features.to(device), targets.to(device)

            # forward
            output = decoder(image_features, targets, lenghts)

            # compute loss
            targets = pack_padded_sequence(targets, lenghts, batch_first=True)[0]
            loss = criterion(output, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print info
            print(f'epoch {epoch+1} | batch {batch}/{len(train_loader)} | loss {loss:.3f}')




    # save model after training
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")



# if we already trained, and EVAL == True, reload saved model
else:

    data_transform = transforms.Compose([ 
        transforms.Resize(224),     
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])


    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)


    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm



#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################


    # TODO define decode_caption() function in utils.py
    # predicted_caption = decode_caption(word_ids, vocab)

    ################ prepare features
    #################################

    # remove duplicates, make sure image_ids are identical to the order of reference captions
    test_image_ids = [ image_id for i, image_id in enumerate(test_image_ids) if i%5 == 0]
    # merge references of a image. [['ref1', ..., 'ref5'], ['ref1', ..., 'ref5'], ...] 
    test_cleaned_captions = [ test_cleaned_captions[i: i+5] for i in range(len(test_cleaned_captions)) if i%5 == 0]

    # prepare test images dataset
    test_image_dataset = Flickr8k_Images(
        image_ids=test_image_ids,
        transform=data_transform
    )

    # image and reference dataset to dataloader
    test_loader = torch.utils.data.DataLoader(
        test_image_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    generated_captions = []

    # display progress
    from tqdm import tqdm

    for test_image in tqdm(test_loader, 'captions generating'):

        # no gradient to be updated
        with torch.no_grad():
            test_image = test_image.to(device)
            features = encoder(test_image)   # (batch_size, 2048)

            batch_word_ids = decoder.sample(features).clone().cpu()

            for word_ids in batch_word_ids:

                predicted_caption = decode_caption(word_ids.tolist(), vocab)
                generated_captions.append(predicted_caption)

    import random
    import os

    # randomly choose the index of samples to be shown
    idxes = random.choices(range(len(test_image_ids)), k=3)

    # clear output dir, then copy the images to this dir
    os.system('rm -f output/*')
    for i, idx in enumerate(idxes, 1):

        os.system(f'cp {IMAGE_DIR}/{test_image_ids[idx]}.jpg output/{i}.jpg')

        print(f'predicted: {generated_captions[idx]}')
        print('references:')
        for ref in test_cleaned_captions[idx]:
            print(ref)

        print()





#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report



