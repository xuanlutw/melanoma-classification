import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from math import sqrt
from scipy.ndimage import gaussian_filter, median_filter, binary_fill_holes
from scipy.stats import kurtosis, skew
from skimage import filters, measure, morphology, color, exposure

def segmentation(im_path, name):
    img   = plt.imread(im_path)
    img_h = color.rgb2hsv(img)
    
    # Segmentation
    img_adj = exposure.adjust_gamma(img_h[:,:,1], 2)
    img_gf1 = median_filter(img_adj, size=20)
    img_gf2 = gaussian_filter(img_gf1, sigma=5)
    pre_bw  = img_gf2 > filters.threshold_otsu(img_gf2)
    pre_bw2 = morphology.closing(pre_bw, morphology.square(3))
    #pre_bw3 = morphology.remove_small_holes(pre_bw2, area_threshold = 1024)
    pre_bw3 = binary_fill_holes(pre_bw2)

    # Label
    label = measure.label(pre_bw3)
    max_reg = max(measure.regionprops(label), key = lambda x: x.area)
    pts = max_reg.coords
    bw = np.zeros(pre_bw.shape)
    for item in pts:
        bw[item[0]][item[1]] = 1 
    np.save("./data/%s" % name, pts)
    plt.imsave("./data/%s-bw.bmp" % name, bw, cmap=cm.gray)

    # Show tmp result
    # plt.hist(img_gf2.ravel(), 100)
    # plt.show()
    plt.clf()
    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(img_h)
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(img_h[:,:,1], cmap='Greys')
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow(img_adj, cmap='Greys')
    plt.axis('off')
    plt.subplot(2, 4, 5)
    plt.imshow(img_gf1, cmap='Greys')
    plt.axis('off')
    plt.subplot(2, 4, 6)
    plt.imshow(img_gf2, cmap='Greys')
    plt.axis('off')
    plt.subplot(2, 4, 7)
    plt.imshow(pre_bw, cmap='Greys')
    plt.axis('off')
    plt.subplot(2, 4, 8)
    plt.imshow(bw, cmap='Greys')
    plt.axis('off')
    plt.savefig("./data/%s-flow.png" % name)
    plt.show()
    
def IOU(im1_path, im2_path):
    im1 = plt.imread(im1_path)
    im2 = plt.imread(im2_path)
    I_count = 0
    U_count = 0
    for i in range(0, im1.shape[0]):
        for j in range(0, im1.shape[1]):
            if im1[i][j][0] > 0 and im2[i][j][0] > 0:
                I_count = I_count + 1
            if im1[i][j][0] > 0 or  im2[i][j][0] > 0:
                U_count = U_count + 1
    return I_count / U_count

DIM = 28
def feature_extraction(im_path, bw_path):
    img   = plt.imread(im_path)
    bw    = color.rgb2gray(plt.imread(bw_path)) > 0
    img_h = color.rgb2hsv(img)

    label   = measure.label(bw)
    max_reg = max(measure.regionprops(label), key = lambda x: x.area)
    pts     = max_reg.coords
    
    ret    = np.zeros(DIM)
    ret[0] = (max_reg.area)
    ret[1] = (max_reg.perimeter / sqrt(max_reg.area))
    ret[2] = (max_reg.area / max_reg.convex_area)
    ret[3] = (max_reg.eccentricity)

    # Color Gray feature
    img_g = color.rgb2gray(img)
    intensity = [img_g[x[0], x[1]] for x in pts]
    ret[4]  = kurtosis(intensity)
    ret[5]  = skew(intensity)
    ret[6]  = np.std(intensity)
    ret[7]  = np.mean(intensity)

    # Color Blue feature
    intensity = [img[x[0], x[1], 2] for x in pts]
    ret[8]   = kurtosis(intensity)
    ret[9]   = skew(intensity)
    ret[10]  = np.std(intensity)
    ret[11]  = np.mean(intensity)

    # Color hsv feature
    intensity = [img_h[x[0], x[1], 1] for x in pts]
    ret[12]  = kurtosis(intensity)
    ret[13]  = skew(intensity)
    ret[14]  = np.std(intensity)
    ret[15]  = np.mean(intensity)

    # Center
    kernel = morphology.disk(50)
    bw = morphology.erosion(bw, kernel)
    '''
    plt.imshow(bw, cmap = 'Greys')
    plt.axis('off')
    plt.show()
    '''
    pts_o = pts
    try:
        label = measure.label(bw)
        max_reg = max(measure.regionprops(label), key = lambda x: x.area)
        pts = max_reg.coords
    except:
        print("Fail %s" % im_name)
        pts = pts_o
    
    # Color Gray feature
    img_g = color.rgb2gray(img)
    intensity = [img_g[x[0], x[1]] for x in pts]
    ret[16]  = kurtosis(intensity)
    ret[17]  = skew(intensity)
    ret[18]  = np.std(intensity)
    ret[19]  = np.mean(intensity)

    # Color Blue feature
    intensity = [img[x[0], x[1], 2] for x in pts]
    ret[20]  = kurtosis(intensity)
    ret[21]  = skew(intensity)
    ret[22]  = np.std(intensity)
    ret[23]  = np.mean(intensity)

    # Color hsv feature
    intensity = [img_h[x[0], x[1], 1] for x in pts]
    ret[24]  = kurtosis(intensity)
    ret[25]  = skew(intensity)
    ret[26]  = np.std(intensity)
    ret[27]  = np.mean(intensity)

    return ret

def seg_all():
    file = open('PH2Dataset/PH2.csv')
    data = file.readlines()
    for index, line in enumerate(data):
        im_name = line.split(',')[0]
        segmentation("PH2Dataset/PH2 Dataset images/%s/%s_Dermoscopic_Image/%s.bmp" 
                % (im_name, im_name, im_name), im_name)
        print(im_name, 
                IOU("./data/%s-bw.bmp" % im_name, 
                    "PH2Dataset/PH2 Dataset images/%s/%s_lesion/%s_lesion.bmp" % (im_name, im_name, im_name)))

def feat_all():
    file = open('PH2Dataset/PH2.csv')
    data = file.readlines()
    feat = np.zeros((len(data), DIM))
    for index, line in enumerate(data):
        print(index)
        im_name = line.split(',')[0]
        res = feature_extraction("PH2Dataset/PH2 Dataset images/%s/%s_Dermoscopic_Image/%s.bmp" 
                % (im_name, im_name, im_name), "data/%s-bw.bmp" % im_name)
        feat[index] = res
    np.save('feature', feat)
    return feat

#segmentation("PH2Dataset/PH2 Dataset images/IMD003/IMD003_Dermoscopic_Image/IMD003.bmp", "IMD003")
#print(IOU("./data/IMD421-bw.bmp", "PH2Dataset/PH2 Dataset images/IMD421/IMD421_lesion/IMD421_lesion.bmp"))
#seg_all()
#print(feature_extraction("PH2Dataset/PH2 Dataset images/IMD003/IMD003_Dermoscopic_Image/IMD003.bmp", "data/IMD003-bw.bmp"))
#print(feature_extraction("PH2Dataset/PH2 Dataset images/IMD003/IMD003_Dermoscopic_Image/IMD003.bmp", "data/IMD003-bw.bmp"))
#feat_all()
