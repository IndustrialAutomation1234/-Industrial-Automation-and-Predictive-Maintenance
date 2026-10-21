import numpy as np
import os
import cv2
from keras.src.utils import to_categorical
from numpy import matlib

from AOA import AOA
from AZO import AZO
from CWO import CWO
from Global_Vars import Global_Vars
from HOA import HOA
from Image_Results import *
from Model_Adaboost import Model_AdaBoost
from Model_CSAViT_SeNet import Model_CSAViT_SeNet
from Model_DCNN import Model_DCNN
from Model_DenseUnet import Model_DenseUNet
from Model_FCN import Model_FCN
from Model_SVM import Model_SVM
from Model_Unet import Model_Unet
from Model_YoloV3 import Model_YoloV3
from Model_YoloV9 import Model_YOLOv9
from Plot_Results import *
from Proposed import Proposed
from objfun import objfun

# Read Dataset1
an = 0
if an == 1:
    dir1 = "./Dataset/Dataset 1/Magnetic-Tile-Defect/MT_Blowhole/Imgs"
    images = []
    labels = []
    GT = []
    img_size = (512, 512)
    for root, dirs, files in os.walk(dir1):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                img = cv2.resize(img, img_size)
                images.append(img)
            else:
                file_path = os.path.join(root, file)
                mask = cv2.imread(file_path)
                mask = cv2.resize(mask, img_size)
                GT.append(mask)
                if np.any(np.nonzero(mask) > 0):
                    labels.append(1)
                else:
                    labels.append(0)
    images_np = np.array(images)
    labels_np = np.array(labels)
    gt_np = np.array(GT)
    indices = np.arange(len(images_np))
    np.random.shuffle(indices)
    images_np = images_np[indices]
    labels_np = labels_np[indices]
    gt_np = gt_np[indices]
    np.save("Images_1.npy", images_np)
    np.save("Target_1.npy", labels_np)
    np.save("GT_1.npy", gt_np)

# Read Dataset 2
an = 0
if an == 1:
    base_path = "./Dataset/Dataset 2/KolektorSDD"
    images = []
    labels = []
    GT = []
    img_size = (512, 512)
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jpg"):
                jpg_path = os.path.join(root, file)
                mask_name = file.replace('.jpg', '_label.bmp')
                mask_path = os.path.join(root, mask_name)
                img = cv2.imread(jpg_path)
                img = cv2.resize(img, img_size)
                images.append(img)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, img_size)
                GT.append(mask)
                if np.any(mask > 0):
                    labels.append(1)
                else:
                    labels.append(0)
    images_np = np.array(images)
    labels_np = np.array(labels)
    gt_np = np.array(GT)
    indices = np.arange(len(images_np))
    np.random.shuffle(indices)
    images_np = images_np[indices]
    labels_np = labels_np[indices]
    gt_np = gt_np[indices]
    labels_cat = to_categorical(labels_np, num_classes=2)
    np.save("Images_2.npy", images_np)
    np.save("GT_2.npy", gt_np)
    np.save("Target_2.npy", labels_np)

# Read Dataset3
an = 0
if an == 1:
    base_path = "./Dataset/Dataset 2/KolektorSDD2/Train"
    images = []
    labels = []
    GT = []
    img_size = (512, 512)
    for file in os.listdir(base_path):
        if file.endswith(".png") and not file.endswith("_GT.png"):
            img_path = os.path.join(base_path, file)
            mask_name = file.replace('.png', '_GT.png')
            mask_path = os.path.join(base_path, mask_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            GT.append(mask)
            label = 1 if np.any(mask > 0) else 0
            labels.append(label)
    images_np = np.array(images)
    labels_np = np.array(labels)
    gt_np = np.array(GT)
    indices = np.arange(len(images_np))
    np.random.shuffle(indices)
    images_np = images_np[indices]
    labels_np = labels_np[indices]
    gt_np = gt_np[indices]
    labels_cat = to_categorical(labels_np, num_classes=2)
    np.save("Images_3.npy", images_np)
    np.save("GT_3.npy", gt_np)
    np.save("Target_3.npy", labels_cat)

no_of_dataset = 3

# Segmentation
an = 0
if an == 1:
    EVAL = []
    for n in range(no_of_dataset):
        Image = np.load(f'Images_{n + 1}.npy', allow_pickle=True)
        Target = np.load(f'GT_{n + 1}.npy', allow_pickle=True)
        Eval = np.zeros((5, 16))
        Eval[0, :], SegImage = Model_Unet(Image, Target)
        Eval[1, :], pred1 = Model_YoloV3(Image, Target)
        Eval[2, :], pred2 = Model_FCN(Image, Target)
        Eval[3, :], pred3 = Model_DenseUNet(Image, Target)
        Eval[4, :], pred4 = Model_YOLOv9(Image, Target)
        EVAL.append(Eval)
        np.save(f'UNet_{n + 1}.npy', SegImage)
        np.save(f'YoloV3_{n + 1}.npy', pred1)
        np.save(f'FCN_{n + 1}.npy', pred2)
        np.save(f'Denseunet_{n + 1}.npy', pred3)
        np.save(f'YoloV9_{n + 1}.npy', pred4)
    np.save('Evaluate_Seg_all.npy', EVAL)

# Optimization
an = 0
if an == 1:
    All_BestSolutions = []
    All_Fitness = []
    for i in range(3):
        images = np.load(f'Images_{i + 1}.npy', allow_pickle=True)
        Target = np.load(f'Target_{i + 1}.npy', allow_pickle=True)

        Global_Vars.Images = images
        Global_Vars.Target = Target

        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([5, 0.01, 1], Npop, 1)  # Minimum values
        xmax = matlib.repmat([255, 0.99, 5], Npop, 1)  # Maximum values

        # Initialize solutions
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])

        fname = objfun
        Max_iter = 50

        print("AOA...")
        bestfit1, fitness1, bestsol1, time_taken1 = AOA(initsol, fname, xmin, xmax, Max_iter)

        print("AZO...")
        bestfit2, fitness2, bestsol2, time_taken2 = AZO(initsol, fname, xmin, xmax, Max_iter)

        print("CWO...")
        bestfit3, fitness3, bestsol3, time_taken3 = CWO(initsol, fname, xmin, xmax, Max_iter)

        print("HOA...")
        bestfit4, fitness4, bestsol4, time_taken4 = HOA(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed...")
        bestfit5, fitness5, bestsol5, time_taken5 = Proposed(initsol, fname, xmin, xmax, Max_iter)
        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

        Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]

        All_BestSolutions.append(BestSol)
        All_Fitness.append(Fitness)

    np.save('BestSolutions.npy', All_BestSolutions)
    np.save('Fitness.npy', All_Fitness)

# Classification
an = 0
if an == 1:
    Eval_all = []
    for i in range(no_of_dataset):
        Feature = np.load(f'Images_{i + 1}.npy', allow_pickle=True)
        Target = np.load(f'Target_{i + 1}.npy', allow_pickle=True)
        BestSol = np.load('BestSolutions.npy', allow_pickle=True)[i]
        Data = Feature
        k_fold = 5
        EVAL = []
        for m in range(k_fold):
            Total_Index = np.arange(Data.shape[0])
            Test_index = np.arange(((m - 1) * (Data.shape[0] / k_fold)) + 1, m * (Data.shape[0] / k_fold))
            Train_Index = np.setdiff1d(Total_Index, Test_index)
            Train_Data = Data[Train_Index, :]
            Train_Target = Target[Train_Index]
            Test_Data = Data[Test_index.astype('int'), :]
            Test_Target = Target[Test_index.astype('int')]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                sol = BestSol[j]
                Eval[j, :] = Model_CSAViT_SeNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :] = Model_DCNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :] = Model_AdaBoost(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :] = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_CSAViT_SeNet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
        np.save('Eval_all.npy', Eval_all)


plotConvResults()
Plot_Results()
Plot_ROC_Curve()
Table()
plot_seg_results()
Image_Results()
Sample_Images()