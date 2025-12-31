import copy

import find_marker
import numpy as np
import cv2
import time
import marker_detection_white
import sys
import setting
#import matplotlib.pyplot as plt
import pickle as pk
import os

from gelsight import gsdevice


def find_cameras():
    # checks the first 10 indexes.
    if os.name == 'nt':
        return [1]
    index = 0
    arr = []
    i = 10
    while i >= 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            command = 'v4l2-ctl -d ' + str(index) + ' --info'
            is_arducam = os.popen(command).read()
            if is_arducam.find('Arducam') != -1 or is_arducam.find('Mini') != -1:
                arr.append(index)
            cap.release()
        index += 1
        i -= 1

    return arr


# def resize_crop_mini(img, imgw, imgh):
#     # resize, crop and resize back
#     img = cv2.resize(img, (895, 672))  # size suggested by janos to maintain aspect ratio
#     border_size_x, border_size_y = int(img.shape[0] * (1 / 9)), int(np.floor(img.shape[1] * (1 / 9)))  # remove 1/7th of border from each size
#     img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
#     img = img[:, :-1]  # remove last column to get a popular image resolution
#     img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
#     return img

def cresize_crop_mini(img, fraction1=1/9, fraction2=1/9):
    """
    Crops the image from the bottom, left, and right, and returns the
    cropped image along with its new width and height.
    """
    h_original, w_original = img.shape[:2]

    # 垂直方向（高）
    crop_top = 0
    # 底部裁剪
    crop_bottom = int(h_original * fraction1)

    # 水平方向（宽）
    # 左边裁剪
    crop_left = int(w_original * fraction2)
    # 右边裁剪
    crop_right = int(w_original * fraction2)

    # 2. 定义裁剪的起始和结束行列索引
    start_row = crop_top
    end_row = h_original - crop_bottom
    start_col = crop_left
    end_col = w_original - crop_right

    # 4. 执行裁剪
    img = img[start_row:end_row, start_col:end_col]

    # 5. 获取新的尺寸
    new_h, new_w = img.shape[:2]

    # 6. 返回裁剪后的图像和新的宽度、高度
    return img, new_w, new_h


def trim(img):
    img[img<0] = 0
    img[img>255] = 255


def compute_tracker_gel_stats(thresh): # 计算marker的半径
    numcircles = 5 * 5;
    mmpp = .063;
    true_radius_mm = .5;
    true_radius_pixels = true_radius_mm / mmpp; # 计算真实半径对应的像素值
    circles = np.where(thresh)[0].shape[0]
    circlearea = circles / numcircles;
    radius = np.sqrt(circlearea / np.pi);
    radius_in_mm = radius * mmpp;
    percent_coverage = circlearea / (np.pi * (true_radius_pixels) ** 2);
    return radius_in_mm, percent_coverage*100.


def main(argv):
    # --- MODIFIED: imgw and imgh are now determined dynamically after cropping ---
    # imgw = 1435
    # imgh = 1440 
    USE_LIVE_R1 = False
    calibrate = False

    outdir = './TEST/'

    SAVE_VIDEO_FLAG = True
    SAVE_DATA_FLAG = True
    SAVE_ONE_IMG_FLAG = False

    if SAVE_ONE_IMG_FLAG:
        sn = input('Please enter the serial number of the gel \n')
        #sn = str(5)
        viddir = outdir + 'vids/'
        imgdir = outdir + 'imgs/'
        resultsfile = outdir + 'marker_qc_results.txt'
        vidfile = viddir + sn + '.avi'
        imgonlyfile = imgdir + sn + '.png'
        maskfile = imgdir + 'mask_' + sn + '.png'
        # check to see if the directory exists, if not create it
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not os.path.exists(viddir):
            os.mkdir(viddir)
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)

    if SAVE_DATA_FLAG:
        datadir = outdir + 'data'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        count = 0
        needcount = True;
        while needcount:
            outfile = datadir + '/marker_locations_' + str(count) + '.csv'
            if os.path.isfile(outfile):
                count = count + 1;
            else:
                needcount = False;
        datafile = open(outfile,"w")
        # 添加面积列到表头
        datafile.write("frameno, row, col, Ox, Oy, Cx, Cy, major_axis, minor_axis, angle\n")
        vidfile = datadir + '/video_' + str(count) + '.avi'

    if len(sys.argv) > 1:
        if sys.argv[1] == 'calibrate':
            calibrate = True

    dev_id = 0
    cap = cv2.VideoCapture(dev_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024) # set the resolution to 1280x1024
    WHILE_COND = cap.isOpened()

    setting.init()
    RESCALE = setting.RESCALE
    
    # --- MODIFIED: Initialize VideoWriter object to None. It will be created later. ---
    out = None

    frame0 = None
    counter = 0
    while 1:
        if counter < 50:
            ret, frame = cap.read()
            print('flush black imgs')
    
            if counter == 48:
                ret, frame = cap.read()
                # 畸变矫正
                # frame, new_cam_mat = marker_detection_white.init_HSR(frame)
                # output_folder = os.path.join('MarkerCalibration', 'Results', 'data', 'PreprocessPara')
                # output_filepath = os.path.join(output_folder, 'NewIntrinsic.xlsx') 
                # marker_detection_white.save_new_intrinsics_to_excel(new_cam_mat, output_filepath)
                
                # --- MODIFIED: Call the new version of cresize_crop_mini ---
                # This now returns the cropped image AND its new dimensions.
                # Make sure your cresize_crop_mini function is updated to return (image, width, height)
                frame, imgw, imgh = cresize_crop_mini(frame, fraction1=1/9, fraction2=1/9)
                print(f"Frame cropped. New resolution for video: {imgw}x{imgh}")

                # --- MODIFIED: Create the VideoWriter object HERE, with the correct dimensions ---
                if SAVE_VIDEO_FLAG:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(vidfile, fourcc, 25, (imgw, imgh), isColor=True)
                    print(f"Video writer initialized for '{vidfile}' with size {imgw}x{imgh}.")

                # ### find marker masks
                # mask = marker_detection_white.find_marker(frame)
                # ### find marker centers
                # mc, areas = marker_detection_white.marker_center(mask, frame)
                # 在你的主循环中

                # 在你的主循环中
                # 新的调用:
                mask, area_mask = marker_detection_white.find_marker(frame)
                # 【修改】接收新的数据结构
                marker_data_list = marker_detection_white.marker_center(mask, area_mask, frame)
                # 【修改】增加健壮性检查
                if not marker_data_list:
                    print("Error: No markers found during initialization. Exiting.")
                    sys.exit(1) # 如果初始化失败则退出
                # 【修改】从新的数据结构中提取 mc 用于匹配
                mc = np.array([d['center'] for d in marker_data_list])


                break

            counter += 1

    counter = 0

    mccopy = mc
    
    mc_sorted1 = mc[mc[:,0].argsort()]
    mc1 = mc_sorted1[:setting.N_]
    mc1 = mc1[mc1[:,1].argsort()]

    mc_sorted2 = mc[mc[:,1].argsort()]
    mc2 = mc_sorted2[:setting.M_]
    mc2 = mc2[mc2[:,0].argsort()]

    N_= setting.N_
    M_= setting.M_
    fps_ = setting.fps_
    x0_ = np.round(mc1[0][0])
    y0_ = np.round(mc1[0][1])
    dx_ = mc2[1, 0] - mc2[0, 0]
    dy_ = mc1[1, 1] - mc1[0, 1]

    print ('x0:',x0_,'\n', 'y0:', y0_,'\n', 'dx:',dx_,'\n', 'dy:', dy_)

    radius ,coverage = compute_tracker_gel_stats(mask)

    if SAVE_ONE_IMG_FLAG:
        fresults = open(resultsfile, "a")
        fresults.write(f"{sn} {float(f'{dx_:.2f}')} {float(f'{dy_:.2f}')} {float(f'{radius*2:.2f}')} {float(f'{coverage:.2f}')}\n")

    m = find_marker.Matching(N_,M_,fps_,x0_,y0_,dx_,dy_)

    frameno = 0
    #  在主循环开始前，初始化计时器和帧计数器
    start_time = time.time()
    saved_frame_count = 0
    
    try:
        while (WHILE_COND):
            ret, frame = cap.read()
            if not(ret):
                break
            # frame, _ = marker_detection_white.init_HSR(frame)
            # --- MODIFIED: Crop each subsequent frame. We don't need the dimensions anymore, so use _. ---
            frame, _, _ = cresize_crop_mini(frame, fraction1=1/9, fraction2=1/9)
            raw_img = copy.deepcopy(frame)

            # ### find marker masks
            # mask = marker_detection_white.find_marker(frame)
            # ### find marker centers
            # mc,areas = marker_detection_white.marker_center(mask, frame)
            mask, area_mask = marker_detection_white.find_marker(frame)
            # 【修改】接收新的数据结构
            marker_data_list = marker_detection_white.marker_center(mask, area_mask, frame)
            # 【修改】增加健-壮性检查
            if not marker_data_list:
                print(f"Warning: No markers found in frame {frameno}. Skipping frame.")
                # 如果这一帧没找到，跳过后续处理
                if SAVE_VIDEO_FLAG and out is not None:
                    out.write(frame) # 仍然保存原始帧
                cv2.imshow('frame', frame) # 仍然显示
                continue
            # 【修改】从新的数据结构中提取 mc
            mc = np.array([d['center'] for d in marker_data_list])

            if calibrate == False:

                tm = time.time()
                m.init(mc)
                m.run()
                flow = m.get_flow()

                if frame0 is None:
                    frame0 = frame.copy()
                    frame0 = cv2.GaussianBlur(frame0, (int(63), int(63)), 0)

                marker_detection_white.draw_flow(frame, flow)
                frameno = frameno + 1
                
                if SAVE_DATA_FLAG:
                    Ox, Oy, Cx, Cy, Occupied = flow
                    # 【修改】将 marker_data_list 转换为一个以坐标为键的查找表，方便快速查找
                    lookup_table = {d['center']: d for d in marker_data_list}

                    for i in range(len(Ox)):
                        for j in range(len(Ox[i])):
                            # 我们信任的坐标
                            current_pos = (Cx[i][j], Cy[i][j])
                            
                            # 【修改】反向查询：在所有检测到的标记点中，找到与当前网格坐标最接近的那个
                            # np.linalg.norm 计算欧氏距离
                            # 使用 try-except 避免 lookup_table 为空时出错
                            try:
                                best_match_center = min(lookup_table.keys(), key=lambda k: np.linalg.norm(np.array(k) - np.array(current_pos)))
                                
                                # 从查询结果中获取正确的椭圆数据
                                matched_data = lookup_table[best_match_center]
                                major_axis_length = matched_data['major_axis']
                                minor_axis_length = matched_data['minor_axis']
                                angle_val = matched_data['angle']
                            except ValueError:
                                # 如果 lookup_table 为空，则无法匹配，使用默认值
                                major_axis_length = 0.0
                                minor_axis_length = 0.0
                                angle_val = 0.0

                            # 写入正确关联的数据
                            datafile.write(
                                f"{frameno:6d}, {i:3d}, {j:3d}, {Ox[i][j]:6.2f}, {Oy[i][j]:6.2f}, {Cx[i][j]:6.2f}, {Cy[i][j]:6.2f}, {major_axis_length:6.2f}, {minor_axis_length:6.2f}, {angle_val:6.2f}\n")
                    # 每次成功写入数据后，计数器加一
                    saved_frame_count += 1
            mask_img = np.asarray(mask)

            bigframe = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
            cv2.imshow('frame', bigframe)
            bigmask = cv2.resize(mask_img*255, (int(mask_img.shape[1]*0.5), int(mask_img.shape[0]*0.5)))
            cv2.imshow('mask', bigmask)

            if SAVE_ONE_IMG_FLAG:
                cv2.imwrite(imgonlyfile, raw_img)
                cv2.imwrite(maskfile, mask*255)
                SAVE_ONE_IMG_FLAG = False

            if calibrate:
                cv2.imshow('mask',mask_img*255)
            
            # --- MODIFIED: Check if the VideoWriter object has been created before writing to it ---
            if SAVE_VIDEO_FLAG and out is not None:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print('Interrupted!')
        
    # 计算并打印平均帧率
    finally:
        end_time = time.time()
        total_duration = end_time - start_time
        if total_duration > 0 and saved_frame_count > 0:
            effective_fps = saved_frame_count / total_duration
            print("\n-----------------------------------------")
            print(f"Data logging finished.")
            print(f"Total frames saved to CSV: {saved_frame_count}")
            print(f"Total duration: {total_duration:.2f} seconds")
            print(f"Actual average data saving FPS: {effective_fps:.2f} Hz")
            print("-----------------------------------------")

    ### release the capture and other stuff
    cap.release()
    cv2.destroyAllWindows()
    
    # --- MODIFIED: Check if the VideoWriter object exists before trying to release it ---
    if SAVE_VIDEO_FLAG and out is not None:
        out.release()

if __name__ == '__main__':
    main(sys.argv[1:])