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



    #################### to train
    #############################


    # train mode
    decoder.train()

    for epoch in range(NUM_EPOCHS):

        for batch, (image_features, targets, lenghts) in enumerate(train_loader, 1):

            # image_features: size(batch_size, 2048)
            # targets: ['vocab(<start>)', ..., vocab('<end>')]. size(batch_size, seq_len)
            # lengths: len of targets. size(batch_size, 1)
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
    test_image_ids = test_image_ids[::5]
    # merge references of a image. [['ref1', ..., 'ref5'], ['ref1', ..., 'ref5'], ...] 
    test_cleaned_captions = [ test_cleaned_captions[i: i+5] for i in range(0, len(test_cleaned_captions), 5)]

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




    ############### to evaluate
    ###########################

    # store generated captions
    generated_captions = []

    # display progress
    from tqdm import tqdm

    for test_image in tqdm(test_loader, 'captions generating'):
        # no gradient to be updated
        with torch.no_grad():
            test_image = test_image.to(device)
            features = encoder(test_image)   # (batch_size, 2048)

            # there is a bug if computeing all features, so feed just one image feature
            for feature in features:
                feature = feature.unsqueeze(0)  # (1, 2048)

                # conver result tensor to list. (1, max_seq_len)
                word_ids = decoder.sample(feature).clone().cpu().flatten().tolist()
                
                # decode word_ids to caption, and append to generated_captions
                predicted_caption = decode_caption(word_ids, vocab)
                generated_captions.append(predicted_caption)




    ############## print result
    ###########################

    import random
    import os

    # randomly choose the index of samples to be shown
    idxes = random.choices(range(len(test_image_ids)), k=3)

    # clear output dir, then copy the images to this dir
    os.system('rm -f output/*')

    # print three samples
    for i, idx in enumerate(idxes, 1):

        # copy image to output dir to validate
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




    ############## BLEU scores and cosine similarity
    ################################################

    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    from sklearn.metrics.pairwise import cosine_similarity

    # smooth function in calculating bleu in dealing with warning
    smoothie = SmoothingFunction().method4

    # store bleu score of every image's generated caption and its references
    bleu_scores = []
    cosine_scores = []

    for i, (refs, gen) in enumerate(zip(test_cleaned_captions, generated_captions)):
        # bleu4
        bleu_score = sentence_bleu(refs, gen, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

        # convert references and generated caption to vector, where vec_refs is a list of vectors
        vec_refs, vec_gen = texts_to_vecs(refs, gen, vocab)

        # compare generated with each reference
        cosine_score = []
        for vec_ref in vec_refs:
            cosine_score.append(cosine_similarity([vec_ref], [vec_gen]))

        # add them up and calculate average
        cosine_scores.append(np.sum(cosine_score) / len(cosine_score))
    



    ############### print result
    ############################


    # print bleu scores details
    print(f'overall average bleu: {np.sum(bleu_scores)/len(bleu_scores):.3f}')
    print('==============================')

    # index of image_id with highest bleu
    idx = np.argmax(bleu_scores)

    # copy image to output dir
    os.system(f'cp {IMAGE_DIR}/{test_image_ids[idx]}.jpg output/bleu_high.jpg')

    print(f'highest bleu index: {idx}, score: {np.max(bleu_scores):.3f}')
    print(f'predicted: {generated_captions[idx]}')
    print('references:')
    for ref in test_cleaned_captions[idx]:
        print(ref)
    print()


    idx = np.argmin(bleu_scores)
    os.system(f'cp {IMAGE_DIR}/{test_image_ids[idx]}.jpg output/bleu_low.jpg')
    print(f'lowest bleu index: {idx}, socre:{np.min(bleu_scores):.3f}')
    print(f'predicted: {generated_captions[idx]}')
    print('references:')
    for ref in test_cleaned_captions[idx]:
        print(ref)
    print()



    # print cosine similarity scores details
    print(f'overall average cosine similarity: {np.sum(cosine_scores)/len(cosine_scores):.3f}')
    print('==============================')

    idx = np.argmax(cosine_scores)
    os.system(f'cp {IMAGE_DIR}/{test_image_ids[idx]}.jpg output/cosine_high.jpg')
    print(f'highest cosine similarity index: {idx}, score: {np.max(cosine_scores):.3f}')
    print(f'predicted: {generated_captions[idx]}')
    print('references:')
    for ref in test_cleaned_captions[idx]:
        print(ref)
    print()

    idx = np.argmin(cosine_scores)
    os.system(f'cp {IMAGE_DIR}/{test_image_ids[idx]}.jpg output/cosine_low.jpg')
    print(f'lowest cosine similarity index: {idx}, score: {np.min(cosine_scores):.3f}')
    print(f'predicted: {generated_captions[idx]}')
    print('references:')
    for ref in test_cleaned_captions[idx]:
        print(ref)
    print()
