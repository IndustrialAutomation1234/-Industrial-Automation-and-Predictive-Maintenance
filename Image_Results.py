import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def Image_Results():
    I = [11, 16, 21, 36, 81]
    for n in range(3):
        Images = np.load('Images_'+str(n+1)+'.npy', allow_pickle=True)
        GT = np.load('GT_'+str(n+1)+'.npy', allow_pickle=True)
        UNET = np.load('Unet_'+str(n+1)+'.npy', allow_pickle=True)
        UnetPlusPlus = np.load('Denseunet_'+str(n+1)+'.npy', allow_pickle=True)
        FCN = np.load('FCN_'+str(n+1)+'.npy', allow_pickle=True)
        # RESUNET = np.load('Resunet.npy', allow_pickle=True)
        PROPOSED = np.load('YoloV9_'+str(n+1)+'.npy', allow_pickle=True)
        for i in range(len(I)):
            plt.subplot(2, 3, 1)
            plt.title('Original')
            plt.imshow(Images[I[i]])
            plt.subplot(2, 3, 2)
            plt.title('GroundTruth')
            plt.imshow(GT[I[i]])
            plt.subplot(2, 3, 3)
            plt.title('UNET')
            plt.imshow(UNET[I[i]])
            plt.subplot(2, 3, 4)
            plt.title('UnetPlusPlus')
            plt.imshow(UnetPlusPlus[I[i]])
            plt.subplot(2, 3, 5)
            plt.title('FCN')
            plt.imshow(FCN[I[i]])
            plt.subplot(2, 3, 6)
            plt.title('Proposed')
            plt.imshow(PROPOSED[I[i]])
            plt.tight_layout()
            plt.show()
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'orig-' + str(i + 1) + '.png', Images[I[i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'gt-' + str(i + 1) + '.png', GT[I[i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'unet-' + str(i + 1) + '.png', UNET[I[i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'Denseunet-' + str(i + 1) + '.png',
            #            UnetPlusPlus[I[i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'FCN-' + str(i + 1) + '.png',
            #            FCN[I[i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'proposed-' + str(i + 1) + '.png',
            #            PROPOSED[I[i]])


def Sample_Images():
    for n in range(3):
        Orig = np.load('Images_'+str(n+1)+'.npy', allow_pickle=True)
        ind = [10, 20, 30, 40, 50, 60]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from Dataset " + str(n+1))
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[4]])
        # plt.show()
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[5]])
        plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
