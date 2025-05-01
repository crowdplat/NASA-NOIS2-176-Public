import os
import argparse
import torch

from torchvision import transforms
from model import FastSCNN
from PIL import Image

from utils.transforms import NewPad
from utils.transforms import pred_2_img
from metrics import pixel_accuracy


parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--checkpoint', type=str, default='C:/fast_scnn/checkpoints/FastSCNN01_18',
                    help='which checkpoint to use')
parser.add_argument('--input_image', type=str, default="D:/IR Data Comparison/Extracted_Overwatch_Data/00_12_27_912/frame0-00-58.27.jpg", 
                    help='path to the input picture')
parser.add_argument('--outdir', default='data/test_result', type=str,
                    help='path to save the predict result')
parser.add_argument('--num_classes', type=int, default=2,
                    help='num of classes in model')
parser.add_argument('--eval', default=None, type=str,
                    help='image label for evaluation score')


args = parser.parse_args()


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    """image_transformer = transforms.Compose([
        NewPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.3396, 0.3628, 0.3362], [0.1315, 0.1287, 0.1333])
    ])"""
    image_transformer = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2555], std=[0.1917])  # Use computed values
    ])
    image = Image.open(args.input_image).convert('L')
    image = image_transformer(image).unsqueeze(0).to(device)
    model = FastSCNN(args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # model = torch.load(args.checkpoint, map_location=device)
    print('Finished loading model!')
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).to(device)
    outname = os.path.splitext(os.path.split(args.input_image)[-1])[0] + '.png'
    pred_2_img(pred.cpu(), os.path.join(args.outdir, outname))
    if not args.eval is None:
        # only one label for now, so add batch=1 dim
        label = torch.load(args.eval).unsqueeze(0)
        output = outputs[0]
        print("image score: ", pixel_accuracy(output, label))


if __name__ == '__main__':
    predict()

