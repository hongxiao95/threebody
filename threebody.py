# -*- coding:utf-8 -*-
# @Author HONG, Xiao

import scipy.constants as CONSTANTS
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os,sys
import numba
import time
import inputimeout
from datetime import datetime
from PIL import ImageDraw, ImageFont, Image

class MP:
    '''
    定义质点
    '''
    def __init__(self, pos:np.array, m:np.float64, v:np.array = np.zeros(2), name:str = "None", a:np.array = np.zeros(2), dtype=np.float32):
        self.dtype = dtype
        self.pos, self.m, self.v, self.name, self.a = np.array(pos, dtype=self.dtype), self.dtype(m), np.copy(v).astype(self.dtype), name, np.copy(a).astype(self.dtype)
        self.ori_pos, self.ori_m, self.ori_v = self.pos.copy(), self.m.copy(), self.v.copy()

    def _clear_a(self):
        self.a = np.zeros(2, dtype=self.dtype)

    def _clac_distance(self, other:"MP"):
        return np.linalg.norm(other.pos - self.pos)

    
    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:], numba.float64, numba.float64, numba.float64[:], numba.float64[:]))
    def __calc_a(olda, m, distance, otherpos, selfpos):
        return olda + (CONSTANTS.G * m / np.power(distance, 3) * (otherpos - selfpos))

    def calc_a(self, mps:list["MP"]):
        '''
        计算合加速度
        '''
        self._clear_a()
        for mp in mps:
            if mp == self:
                continue
            distance = self._clac_distance(mp)
            # self.a += CONSTANTS.G * mp.m / np.power(distance, 3) * (mp.pos - self.pos)
            self.a = self.__calc_a(self.a, mp.m, distance, mp.pos, self.pos)

    @staticmethod
    @numba.njit()
    def __update_move(pos, v, dlt_t, a):
        dlt_s = v * dlt_t + a * np.square(dlt_t) / 2
        new_v = v + a * dlt_t
        new_pos = pos + dlt_s
        return [new_v, new_pos]

    def update_move(self, dlt_t:np.int32):
        '''
        根据算出的a更新pos和速度,务必全部更新完加速度再更新距离,防止循环计算
        '''
        # dlt_s = self.v * dlt_t + self.a * np.square(dlt_t) / 2
        # self.v += self.a * dlt_t
        # self.pos += dlt_s
        res = self.__update_move(self.pos, self.v, dlt_t, self.a)
        self.v = res[0]
        self.pos = res[1]

class MultiBody:
    '''
    多体模拟系统
    '''
    video_root_dir = "video"
    colors = [(240,50,50),(50,240,50),(50,50,240),(240,240,50),(240,50,240),(50,240,240),(255,255,255)]
    def __init__(self, mps:list[MP], dlt_t:np.int32 = 10, iter_round:np.int32 = 8640, history_count:int = 100, sub_dir:str = "common"):
        self.mps, self.dlt_t, self.iter_round, self.sub_dir = mps, dlt_t, iter_round, sub_dir
        self.whole_path = f"{self.video_root_dir}{os.sep}{self.sub_dir}"
        self.history_count = history_count
        self.historys = [np.zeros((self.history_count, 2), dtype=np.float64) for j in range(len(self.mps))]
        self.current_round = 0
        for i in range(len(self.mps)):
            self.historys[i][0] = np.copy(self.mps[i].pos)

    def calc_round(self):
        self.current_round += 1
        for i in range(self.iter_round):
            for mp in self.mps:
                mp.calc_a(self.mps)
            for j, mp in enumerate(self.mps):
                mp.update_move(self.dlt_t)
                # 记录历史移动轨迹
                self.historys[j][self.current_round % self.history_count] = np.copy(mp.pos)

    def _calc_canvas_pos(self, ori_pos:np.ndarray, ori_opoint:np.ndarray, div_times:int, width:int, height:int):
        relative_pos = ori_pos - ori_opoint
        canvas_pos = (np.array([width, height]) / 2 + np.array(relative_pos / div_times)).astype(np.int32)

        # 注意调整画布关系,y轴需要倒置
        canvas_pos[1] = height - canvas_pos[1]
        return canvas_pos

    def _in_canvas(self, pos, width, height):
        if np.min(pos) < 0 or pos[0] > width or pos [1] > height:
            return False
        else:
            return True
    
    @staticmethod
    def __put_text_chinese(img:np.ndarray, text:str, pos:list | tuple, color:tuple, font_size:float):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "Deng.ttf"
        font = ImageFont.truetype(font_path, font_size)
        draw.text(pos, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def gen_video(self, file_name:str="threebody", width:int=1920, height:int=1080, max_tail:int=300, padding:float = 0.3, fps:int=30):
        print("\n")
        if os.path.exists(self.whole_path) == False:
            os.makedirs(self.whole_path)

        dot_scale = [mp.m / np.min([other.m for other in self.mps]) for mp in self.mps]
        
        # 模版画面
        tpl_img = np.zeros((height, width, 3), dtype=np.uint8)
        p_count = len(self.mps)
        fourcc = cv2.VideoWriter.fourcc(*"avc1")
        file_type = ".mp4"
        if "--x264" in sys.argv:
            fourcc = cv2.VideoWriter.fourcc(*"X264")
            file_type = ".mp4"
        elif "--xvid" in sys.argv:
            fourcc = cv2.VideoWriter.fourcc(*"XVID")
            file_type = ".avi"
        elif "--h264" in sys.argv:
            fourcc = cv2.VideoWriter.fourcc(*"H264")
            file_type = ".mp4"
        
        video_witer = cv2.VideoWriter(full_file_name:=f"{self.whole_path}{os.sep}{file_name}{file_type}",fourcc=fourcc,fps=30,frameSize=(width, height))
        for i in range(self.history_count):
            print(f"正在写入{full_file_name}第{i}帧\r", end="")
            current_img = tpl_img.copy()
            # 对于每一步，首先计算相对坐标
            # 锚定minx对应左15%，miny对应下15%,maxx右15%，maxy上15%
            sorted_ori_xs = np.sort([mp_his[i][0] for mp_his in self.historys])
            ori_min_x, ori_max_x = sorted_ori_xs[0], sorted_ori_xs[-1]

            sorted_ori_ys = np.sort([mp_his[i][1] for mp_his in self.historys])
            ori_min_y, ori_max_y = sorted_ori_ys[0], sorted_ori_ys[-1]

            # 计算物理坐标中点位置，该位置应该贴合画面中间
            ori_opoint = np.array([(ori_min_x + ori_max_x) / 2, (ori_min_y + ori_max_y) / 2])

            # 计算坐标缩放倍数
            div_times = int((ori_max_y - ori_min_y) / (height * (1 - 2 * padding)))
            if (x_div_times:=int((ori_max_x - ori_min_x) / (width * (1 - 2 * padding)))) > div_times:
                # 如果按横坐标缩放，代表的范围更广，应采纳横坐标缩放倍率，反之亦然
                div_times = x_div_times
            
            #计算当前点的画布坐标，并绘制当前点
            for j, current_mp in enumerate(self.mps):
                # 对于每个质点，最多追溯max_tail个元素
                canvas_pos = self._calc_canvas_pos(self.historys[j][i], ori_opoint, div_times, width, height)

                eng_font = cv2.FONT_HERSHEY_SIMPLEX

                # 绘制比例尺
                line_pix = int(width * 0.1)
                line_st = (int(width * 0.1), int(height * (1 - padding + 0.1)))
                line_end = (line_st[0] + line_pix, line_st[1])
                half_line_thickness = 2

                # 主比例尺
                rate_color = (0,255,255)
                cv2.line(current_img, line_st, line_end, color=rate_color, thickness=half_line_thickness * 2)
                # 比例尺两边
                cv2.line(current_img, (line_st[0] + half_line_thickness, line_st[1] - int(line_pix * 0.05)), (line_st[0] + half_line_thickness, line_st[1] + int(line_pix * 0.05)),color=rate_color, thickness=half_line_thickness * 2)
                cv2.line(current_img, (line_end[0] - half_line_thickness, line_end[1] - int(line_pix * 0.05)), (line_end[0] - half_line_thickness, line_end[1] + int(line_pix * 0.05)),color=rate_color, thickness=half_line_thickness * 2)
                # 比例尺尺寸和推演时间
                # 计算推演时间
                total_sec = i * self.dlt_t * self.iter_round
                day = total_sec // (3600 * 24)
                hour  = (total_sec % (3600 * 24)) // 3600
                min = (total_sec % 3600) // 60
                sec = total_sec % 60
                text_pos = (line_end[0] + 20, line_end[1] + 10)
                cv2.putText(current_img, f"{np.round(line_pix * div_times / 1000, 1)} KM", text_pos, eng_font, 1, color=(0,255,255), thickness=2)

                # 时间
                text_pos = (line_st[0], line_st[1] + int(height * 0.05))
                # cv2.putText(current_img, f"Day {day}, {hour:02}:{min:02}:{sec:02}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0,255,255), thickness=2)
                chn_font_size = int(width * 0.02)
                current_img = self.__put_text_chinese(current_img, f"Day {day}, {hour:02}:{min:02}:{sec:02}  推演步长:{self.dlt_t} 秒", pos=text_pos, color=(0,255,255), font_size=chn_font_size)

                text_pos = (text_pos[0], text_pos[1] + chn_font_size)

                # 星球信息，首先计算剩余行高，平均除以星球数，计算当前星球所在的行
                remain_height = int((height - text_pos[1]) * 0.8)
                each_height = remain_height // len(self.mps)
                text_pos = (text_pos[0], text_pos[1] + each_height * (j + 1))
                # 一个scale对应的像素数
                font_rate = cv2.getTextSize("mNgP", eng_font, 1, 1)[0][1]
                font_size = (each_height / font_rate) * 0.6

                p_info_text = f"Name: {current_mp.name}, Mass: {current_mp.m:.4e} Kg, Pos: ({self.historys[j][i][0]:.5e}, {self.historys[j][i][1]:.5e}) m"

                cv2.putText(current_img, p_info_text, text_pos, eng_font, font_size, color=self.colors[j], thickness=2)

                cv2.circle(current_img, canvas_pos, radius=int(8*dot_scale[j]), color=self.colors[j % len(self.colors)], thickness=-1)
                tail_i = i - 1
                while tail_i >= 0 and i - tail_i - 1 < max_tail:
                    canvas_pos = self._calc_canvas_pos(self.historys[j][tail_i], ori_opoint, div_times, width, height)
                    if self._in_canvas(canvas_pos, width, height):
                        cv2.circle(current_img, canvas_pos, radius=2, color=self.colors[j % len(self.colors)], thickness=-1)
                    tail_i -= 1
                    
            video_witer.write(current_img)
        video_witer.release()
        print("写入完成")


def main():
    p1 = MP(pos = [200000000,0], m = 2e24, v = np.array([0,500]), name="p1", dtype=np.float64)
    p2 = MP(pos = [0,200000000], m = 1.5e24, v = np.array([-500,0]), name="p2", dtype=np.float64)
    p3 = MP(pos = [-200000000,0], m = 1.8e24, v = np.array([0,-500]), name="p3", dtype=np.float64)

    sub_dir = datetime.now().strftime("%m_%d_%H%M%S")
    system = MultiBody([p1, p2, p3], 2, 3600, 12*30*10, sub_dir=sub_dir)
    go_calc = True
    is_first = True
    video_no = 1
    
    while go_calc:
        st = time.time()
        for i in range(system.history_count - (1 if is_first else 0)):
            print(f"calc round {i} / {system.history_count}\r", end="")
            system.calc_round()
        end = time.time()
        print(f"calc use {end - st}s")

        system.gen_video(f"threebody_{video_no}", max_tail=1500)
        try:
            go_calc = inputimeout.inputimeout(("go? (y/n)"), 2).lower().strip() == "y"
        except inputimeout.TimeoutOccurred:
            go_calc = True
        is_first  =False
        video_no += 1

if __name__ == "__main__":
    main()