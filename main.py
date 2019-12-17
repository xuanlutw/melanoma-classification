import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from skimage import data,filters,segmentation,measure,morphology,color,exposure, io
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from math import sqrt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.stats import kurtosis, skew
from skimage import measure, color

def plti(im, **kwargs):
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')
    plt.show()

def load_img(im_name):
    path = "./PH2Dataset/PH2 Dataset images/%s/%s_Dermoscopic_Image/%s.bmp" % (im_name, im_name, im_name)
    im = plt.imread(path)
    # print(im.shape)
    return im

def load_img2(im_name):
    path = "./PH2Dataset/PH2 Dataset images/%s/%s.tiff" % (im_name, im_name)
    im = plt.imread(path)
    # print(im.shape)
    return im

def save_img(im_name, img):
    path = "./PH2Dataset/PH2 Dataset images/%s/%s.tiff" % (im_name, im_name)
    im = plt.imsave(path, img)
    # print(im.shape)
    return im

def cal_feature(im_name):
    img = load_img(im_name)
    #plti(img[:,:,0], cmap='Greys')
    img_h = color.rgb2hsv(img)
    img_gamma = exposure.adjust_gamma(img_h[:,:,1], 2)
    img_g = color.rgb2gray(img_h)
    #plti(img_h[:,:,0], cmap='Greys')
    #plti(img_h[:,:,1], cmap='Greys')
    #plti(img_h[:,:,2], cmap='Greys')
    #plti(img_h[:,:,1]*0.5 + img_g[:,:,2]*0.5, cmap='Greys')
    #plti(img_g, cmap='Greys')
    
    img_gf1 = median_filter(img_gamma, size=20)
    #plti(img_gf1, cmap='Greys')
    img_gf2 = gaussian_filter(img_gf1, sigma=10)
    #plti(img_gf2, cmap='Greys')
    #img_gf2 = color.rgb2gray(load_img2(im_name))
    save_img(im_name, img_gf2)
    bw = img_gf2 > filters.threshold_otsu(img_gf2)
    bw = morphology.closing(bw, morphology.square(3))
    bw = morphology.remove_small_holes(bw, area_threshold = 10240)
    label = measure.label(bw)

    #plti(np.stack([(1-bw), (1-bw), (1-bw)], axis = 2) * img)
    #plti(label)
    #print(im_name)

    ret = np.zeros(13)
    max_reg = max(measure.regionprops(label), key = lambda x: x.area)
    ret[0] = (max_reg.area)
    ret[1] = (sqrt(max_reg.area) / max_reg.perimeter)
    ret[2] = (max_reg.convex_area / max_reg.area)
    ret[3] = (max_reg.eccentricity)
    ret[4] = (max_reg.equivalent_diameter/ max_reg.major_axis_length)
    ret[5] = (max_reg.equivalent_diameter/ max_reg.minor_axis_length)

    pts = max_reg.coords
    intensity1 = [img_h[x[0], x[1], 1] for x in pts]
    intensity2 = [img_h[x[0], x[1], 2] for x in pts]
    #plt.hist(intensity, bins = range(0,250,5))
    #plt.show()
    #ret[6]  = kurtosis(intensity1)
    ret[6]  = skew(intensity1)
    ret[7]  = np.std(intensity1)
    ret[8]  = np.mean(intensity1) / (np.sum(img_h[:, :, 1]) - np.sum(intensity1)) * (img_h.shape[0] * img_h.shape[1] - len(intensity1))

    intensity3 = [img_g[x[0], x[1]] for x in pts]
    #ret[9]   = kurtosis(intensity3)
    ret[9]   = skew(intensity3)
    ret[10]  = np.std(intensity3)
    ret[11]  = np.mean(intensity3) / (np.sum(img_g[:, :]) - np.sum(intensity3)) * (img_g.shape[0] * img_g.shape[1] - len(intensity3))

    ret[12] = np.mean(intensity1) / np.mean(intensity2)
    return ret

def std(arr):
    z   = np.mean(arr)
    std = np.std(arr)
    return [(x - z) / std for x in arr]

def get_feature():
    '''
    file = open('PH2Dataset/PH2.csv')
    data = file.readlines()
    x_data = np.zeros((len(data), 13))
    y_data = np.zeros(len(data))
    for index, line in enumerate(data):
        x_data[index] = cal_feature(line.split(',')[0])
        if (line.split(',')[1] == 'X'):
            y_data[index] = 1
        if (line.split(',')[2] == 'X'):
            y_data[index] = 2
        if (line.split(',')[3] == 'X\n'):
            y_data[index] = 3
        if index % 10 == 0:
            print(index)
        
    np.save('x.data', x_data)
    np.save('y.data', y_data)
    '''
    x_data = np.load('x.data.npy')
    y_data = np.load('y.data.npy')
    # print(x_data)
    # print(y_data)

    #x_data = np.append(x_data[0:40][:], x_data[-80:][:], axis = 0)
    #y_data = np.append(y_data[0:40][:], y_data[-80:][:], axis = 0)
    x_data = np.append(x_data, x_data[-40:][:], axis = 0)
    y_data = np.append(y_data, y_data[-40:][:], axis = 0)
    for i in range(x_data.shape[1]):
        x_data[:, i] = std(x_data[:, i])

    num_data = x_data.shape[0]
    rng = np.arange(num_data)
    np.random.shuffle(rng)
    x_valid, y_valid = x_data[rng[:num_data//3]], y_data[rng[:num_data//3]]
    x_train, y_train = x_data[rng[num_data//3:]], y_data[rng[num_data//3:]]
    
    #print(x_train)
    #print(y_train)
    c1 = LogisticRegression()
    c1.fit(x_train, y_train)
    print('The result of logisitic reg : {:.4f}'.format(c1.score(x_valid, y_valid)))
    #print('The result of logisitic reg : {:.4f}'.format(c1.score(x_train, y_train)))
    c2 = RandomForestClassifier(n_estimators = 150, max_features = 6, max_depth = 3, min_samples_leaf = 5)
    c2.fit(x_train, y_train)
    print('The result of reandom forest: {:.4f}'.format(c2.score(x_valid, y_valid)))
    #print('The result of reandom forest: {:.4f}'.format(c2.score(x_train, y_train)))
    c3 = GradientBoostingClassifier()
    c3.fit(x_train, y_train)
    print('The result of gradient boost: {:.4f}'.format(c3.score(x_valid, y_valid)))
    #print('The result of gradient boost: {:.4f}'.format(c3.score(x_train, y_train)))
    c4 = SVC(gamma='auto')
    c4.fit(x_train, y_train)
    print('The result of SVM: {:.4f}'.format(c4.score(x_valid, y_valid)))
    #print('The result of SVM: {:.4f}'.format(c4.score(x_train, y_train)))
get_feature()
#print(cal_feature("IMD002"))
#print(cal_feature("IMD035"))
    



# red color
# ring and center
# PR 95 color

