import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import datasets
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.transform import resize

digits = datasets.load_digits()  # 8x8 숫자 데이터 불러옴

# Training
clf = svm.SVC(gamma=0.0001, C=100)  #감마값이 클수록 데이터 하나하나에 민감해져서 에러가 날 가능성이 높음, 적정한 값으로 조절해야함
feature = digits.data[:-10]  # 숫자 이미지 중 뒤부터 10개 뺀 것 (Feature)
label = digits.target[:-10]  # 숫자 이미지 정답 중 뒤부터 10개 뺀 것 (Label)
print(len(digits.data[:-10])) # 학습한 데이터의 갯수
clf.fit(feature,label)  # digits data[:-10] 까지 학습 10개 빼고 학습

# Validation
# 학습 모델을 검증 하는 부분, 테스트 결과에 따라 gamma 값을 조정함
a=[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
for i in a: # 뒤에서 10개 뺀 데이터를 순차적으로 예측하는 구문 = Validation Data를 이용해 학습 모델을 평가
    print('Prediction:', clf.predict(digits.data[[i]]))  # 학습 이후 주어진 이미지에 대해 예측한 결과 값 출력
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r , interpolation="nearest") #최근린 보간법(nearest)을 이용하여 데이터를 표현
    plt.xlabel("It's the Prcoess that validates the model")  # 이미지 설명
    plt.show() # 숫자 이미지 표시

# Test
# 학습 결과물을 외부 HandWritten Digit image로 테스트함.
for i in [1,2,3]:
    img = resize(imread(os.path.join("TestData", "testdata{}.jpg".format(i))), (8, 8)) # TestData 폴더 내에 있는 testdata1,2,3.jpg 를 불러옴
    img = rescale_intensity(img, out_range=(0, 16))  # 이미지의 대비를 확실하게 표현
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest",data="test")
    plt.xlabel("It's the Prcoess that tests the model")  # 이미지 설명
    testimg = [sum(pixel) / 3.0 for row in img for pixel in row] # 3차원 배열로 표현된 데이터를 1차원 배열로 평균내서 변환
    print(testimg) #디버깅
    print("The predict digit is {}".format(clf.predict([testimg]))) #예측 결과
    plt.show()  # 이미지 표시


