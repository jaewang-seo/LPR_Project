####################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
# import tensorflow as tf
# import tensorflow_datasets
import os
from cv2 import cv2


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# plt.style.use('dark_background') 

####################################################
#Read Input Image
Im_list = []
for i in range(1,4):
    im_num = i
    img_ori = cv2.imread('{}.jpg'.format(im_num))
    Im_list.append(img_ori)

def imwrite(filename, img, params=None): 
        try: 
            ext = os.path.splitext(filename)[1] 
            result, n = cv2.imencode(ext, img, params) 

            if result: 
                with open(filename, mode='w+b') as f: 
                  n.tofile(f) 
                return True 
            else: 
                return False 
        except Exception as e: 
            print(e) 
            return False

def main():
    for image in Im_list:
        dkjdkjd(image)


def dkjdkjd(img_ori):

    # print('{}번째 이미지 '.format(im_num))

    # img_ori = cv2.imread('{}.jpg'.format(im_num))   #이미지 불러오기


    height, width, channel = img_ori.shape  #높이, 너비, 채널 확보

    ####################################################
    #Convert Image to Grayscale
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    # plt.figure(figsize=(12, 10))
    # plt.imshow(gray,cmap = 'gray')

    ####################################################
    #Adaptive Threshholding
    img_blurred = cv2.GaussianBlur(gray,ksize=(9,9),sigmaX=0)       #노이즈 제거 
    
    plt.figure(figsize=(12, 10))
    # plt.imshow(img_blurred,cmap = 'gray')
    # plt.show()

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,                               #12 통과 15,20
        C=9
    )
    # plt.figure(figsize=(12, 10))
    # plt.imshow(img_thresh,cmap = 'gray')

    ####################################################
    #Contours(윤곽선)
    contours, _ = cv2.findContours(      #윤곽선을 찾기
        img_thresh,                     #    
        mode=cv2.RETR_LIST,             #
        method=cv2.CHAIN_APPROX_SIMPLE  #
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(
        temp_result,    # 원본이미지
        contours=contours,  #contours 정보
        contourIdx=-1, # -1 : 전체
        color=(255,255,255),
        thickness=1)

    # plt.figure(figsize=(12, 10))
    # plt.imshow(temp_result)

    ####################################################
    #Prepare Data(사각형)
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)    #temp_result 이미지 초기화

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  #컴투어를 감싸는 사각형
        cv2.rectangle(
            temp_result, 
            pt1=(x,y), pt2=(x+w,y+h),
            color=(255,255,255),
            thickness=2
        )

        contours_dict.append({
            'contour':contour,
            'x':x,
            'y':y,
            'w':w,
            'h':h,
            'cx':x + (w / 2),   #중심 좌표
            'cy':y + (h / 2)
        })

    # plt.figure(figsize=(12, 10))
    # plt.imshow(temp_result,cmap = 'gray')

    ####################################################
    #번호판 사각형만 골라오기 



    MIN_AREA = 80                       #boundingRect(사각형)의 최소넓이 
    MIN_WIDTH, MIN_HEIGHT = 2, 8        #boundingRect의 최소 넓이, 높이 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0    #boundingRect의 가로 세로 비율

    possible_contours = [ ]              #위의 조건의 만족하는 사각형


    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']          #가로 * 세로 = 면적
        ratio = d['w'] / d['h']         #가로 / 세로 = 비율

        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt                  #조건의 맞는 값을 idx에 저장한다.
            cnt += 1
            possible_contours.append(d)     #possible_contour를 업데이트한다. 

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        # cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')        #위의 조건을 만족하는 사각형을 그린다. 

    ####################################################
    MAX_DIAG_MULTIPLYER = 5                 #대각선의 5배 안에 있어야함
    MAX_ANGLE_DIFF = 12.0 # 12.0            #세타의 최대값
    MAX_AREA_DIFF = 0.5 # 0.5               #면적의 차이
    MAX_WIDTH_DIFF = 0.8                    #너비차이
    MAX_HEIGHT_DIFF = 0.2                   #높이차이
    MIN_N_MATCHED = 3 # 3                   #위의 조건이 3개 미만이면 뺀다

    def find_chars(contour_list):                   #
        matched_result_idx = []                     #idx 값 저장

        for d1 in contour_list:                     
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:          #d1 과 d2가 같으면 컨티뉴 
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) 

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))  #대각선 길이 
                if dx == 0:
                    angle_diff = 90                                                             #0일 때 각도 90도 (예외처리)
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))                                 #세타 구하기
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])    #면적 비율
                width_diff = abs(d1['w'] - d2['w']) / d1['w']                                   #너비비율
                height_diff = abs(d1['h'] - d2['h']) / d1['h']                                  #높이비율
                                                                                                #조건들
                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])                                      #d2만 넣었기 때문에 마지막으로 d1을 넣은다 

            # append this contour
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:                                       #3개 이하이면 번호판 x
                continue

            matched_result_idx.append(matched_contours_idx)                                     #최종 후보군

            unmatched_contour_idx = []                                                          #최종후보군이 아닌 애들
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)               

            # recursive
            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)                                                  #번호판 이외의 값을 재정의

            break 

        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
    #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.show()


    #################################################### 
    #번호판을 똑바로 정렬
    PLATE_WIDTH_PADDING = 1.3 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])             #x방향으로 순차적으로 정렬

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2         #센터 좌표 구하기
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING  #너비

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)                                   #높이

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']                                            #
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

        # plt.subplot(len(matched_result), 1, i+1)
        # plt.imshow(img_cropped, cmap='gray')


    #################################################### 
    longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        # np.scipy.ndimage.morphology.binary_fill_holes()

        # im_last = ndimage.binary_fill_holes(img_result).astype(int)

        kernel = np.ones((3,3), np.uint8)
        ABC = cv2.erode(img_result, kernel, iterations=1)
        

        chars = pytesseract.image_to_string(ABC, lang='kor', config='--psm 7 --oem 0')

        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c

        print(result_chars)
        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

        plt.subplot(len(plate_imgs), 1, i+1)
        plt.imshow(ABC, cmap='gray')

    ####################################################
    
    # cv2.imwrite(chars + '.jpg', img_result)
    info = plate_infos[longest_idx]
    chars = plate_chars[longest_idx]

    print(chars)
    # print(type(chars))
    img_out = img_ori.copy()

    # filename = '{}.jpg'.format(chars)
    # cv2.imwrite(filename, img_result)
    cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

    # cv2.imwrite('{}.jpg'.format(chars), img_result)
    imwrite('{}.jpg'.format(chars), ABC)

    # plt.figure(figsize=(12, 10))
    # plt.imshow(img_out)
    # plt.show()


if __name__ == "__main__":
    main()
