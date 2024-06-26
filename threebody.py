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
from threading import Thread
from concurrent import futures

class MP:
    '''
    定义质点
    '''
    def __init__(self, pos:np.array, m:np.float64, v:np.array = np.zeros(2), name:str = "None", a:np.array = np.zeros(2), dtype=np.float32):
        self.dtype = dtype
        self.pos, self.m, self.v, self.name, self.a = np.array(pos, dtype=self.dtype), self.dtype(m), np.copy(v).astype(self.dtype), name, np.copy(a).astype(self.dtype)
        self.ori_pos, self.ori_m, self.ori_v = self.pos.copy(), self.m.copy(), self.v.copy()
        self.lorentz = self.calc_lorentz()

    def _clear_a(self):
        self.a = np.zeros(2, dtype=self.dtype)

    def _clac_distance(self, other:"MP"):
        return np.linalg.norm(other.pos - self.pos)

    
    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:], numba.float64, numba.float64, numba.float64[:], numba.float64[:], numba.float64))
    def __calc_a(olda, m, distance, otherpos, selfpos, selflorentz):
        return olda + (m / np.power(distance, 3) / selflorentz * (otherpos - selfpos))
    
    @staticmethod
    @numba.njit(numba.float64(numba.float64[:]))
    def __calc_lorentz(v):
        return 1 / np.sqrt(1 - (v[0] ** 2 + v[1] ** 2)/(CONSTANTS.c ** 2))

    def calc_lorentz(self):
        return self.__calc_lorentz(self.v)

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
            # self.a = self.__calc_a(self.a, mp.m, distance, mp.pos, self.pos)

            # 考虑相对论，对方质量需要乘以对方洛伦兹因子，我方加速度需要除以我方洛伦兹因子
            self.a = self.__calc_a(self.a, mp.m * mp.lorentz, distance, mp.pos, self.pos, self.lorentz)

        self.a *= CONSTANTS.G

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
        self.lorentz = self.calc_lorentz()

class MultiBody:
    '''
    多体模拟系统
    '''
    video_root_dir = "video"
    colors = [(240,50,50),(50,240,50),(50,50,240),(240,240,50),(240,50,240),(50,240,240),(50,140,240),(50,240,140),(140,50,240),(140,240,50),(240,140,50),(240,50,140),(255,255,255)]
    def __init__(self, mps:list[MP], dlt_t:np.int32 = 10, iter_round:np.int32 = 8640, history_count:int = 100, sub_dir:str = "common"):
        self.mps, self.dlt_t, self.iter_round, self.sub_dir = mps, dlt_t, iter_round, sub_dir
        self.whole_path = f"{self.video_root_dir}{os.sep}{self.sub_dir}"
        self.history_count = history_count
        self.historys = [np.zeros((self.history_count, 2), dtype=np.float64) for j in range(len(self.mps))]
        self.v_history = [np.zeros((self.history_count, 2), dtype=np.float64) for j in range(len(self.mps))]
        self.current_round = 0
        for i in range(len(self.mps)):
            self.historys[i][0] = np.copy(self.mps[i].pos)
            self.v_history[i][0] = np.copy(self.mps[i].v)

    def calc_round(self):
        self.current_round += 1
        for i in range(self.iter_round):
            for mp in self.mps:
                mp.calc_a(self.mps)
            for j, mp in enumerate(self.mps):
                mp.update_move(self.dlt_t)
                # 记录历史移动轨迹
                self.historys[j][self.current_round % self.history_count] = np.copy(mp.pos)
                self.v_history[j][self.current_round % self.history_count] = np.copy(mp.v)

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
        scale_max = 2
        if max(dot_scale) > 1:
            second_scale = (max(dot_scale) - 1) / (scale_max - 1)
            new_dot_scale = [1 + (s - 1) / second_scale for s in dot_scale]
            dot_scale = new_dot_scale
        else:
            dot_scale = [1 + (scale_max - 1) / 2 for _ in range(len(self.mps))]
        
        # 模版画面
        tpl_img = np.zeros((height, width, 3), dtype=np.uint8)

        # 绘制比例尺
        line_pix = int(width * 0.1)
        line_st = (int(width * 0.1), int(height * (1 - padding + 0.1)))
        line_end = (line_st[0] + line_pix, line_st[1])
        half_line_thickness = 2

        # 主比例尺
        rate_color = (0,255,255)
        cv2.line(tpl_img, line_st, line_end, color=rate_color, thickness=half_line_thickness * 2)
        # 比例尺两边
        cv2.line(tpl_img, (line_st[0] + half_line_thickness, line_st[1] - int(line_pix * 0.05)), (line_st[0] + half_line_thickness, line_st[1] + int(line_pix * 0.05)),color=rate_color, thickness=half_line_thickness * 2)
        cv2.line(tpl_img, (line_end[0] - half_line_thickness, line_end[1] - int(line_pix * 0.05)), (line_end[0] - half_line_thickness, line_end[1] + int(line_pix * 0.05)),color=rate_color, thickness=half_line_thickness * 2)

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

        
        video_witer = cv2.VideoWriter(full_file_name:=f"{self.whole_path}{os.sep}{file_name}{file_type}",fourcc=fourcc,fps=fps,frameSize=(width, height))
        tpl_imgs = []
        img_buffer_size = 100
        print()
        last_div_times = None
        # 质点占画面的幅比
        max_cross_rate = 1 - padding * 2 + 0.05
        min_cross_rate = 1 - padding * 2 - 0.05
        mid_cross_rate = 1 - padding * 2
        # print(f"crossrate :max:{max_cross_rate}, min:{min_cross_rate}, mid:{mid_cross_rate}")
        # input()
        for i in range(self.history_count):
            print(f"正在写入{full_file_name}第{i + 1} / {self.history_count}帧 \r", end="")
            if (buffer_index:=i % img_buffer_size) == 0:
                tpl_imgs = [tpl_img.copy() for __ in range(img_buffer_size)]
            current_img = tpl_imgs[buffer_index]
            # 对于每一步，首先计算相对坐标
            
            sorted_ori_xs = np.sort([mp_his[i][0] for mp_his in self.historys])
            ori_min_x, ori_max_x = sorted_ori_xs[0], sorted_ori_xs[-1]

            sorted_ori_ys = np.sort([mp_his[i][1] for mp_his in self.historys])
            ori_min_y, ori_max_y = sorted_ori_ys[0], sorted_ori_ys[-1]

            # 计算物理坐标中点位置，该位置应该贴合画面中间
            ori_opoint = np.array([(ori_min_x + ori_max_x) / 2, (ori_min_y + ori_max_y) / 2])

            # 计算坐标缩放倍数
            x_cross = ori_max_x - ori_min_x
            y_cross = ori_max_y - ori_min_y
            
            # 为了防止画面抖动，允许倍数左右缩放0.05个padding，也就是
            # 本次使用的画面幅比控制
            using_cross_rate = mid_cross_rate
            need_recalc_rate = True
            if last_div_times != None:
                # 计算沿用上次的倍数是否符合
                # 沿用上次，y方向实际占据屏幕比例
                y_occupy = y_cross / last_div_times / height
                x_occupy = x_cross / last_div_times / width
                # print(f"frame{i}: y_occ_rate {y_occupy}, x_occ_rate {x_occupy}",end="")

                # 分情况讨论：任意一个占据比例大于最大，则需要重新计算，贴合上限
                if max(y_occupy, x_occupy) > max_cross_rate:
                    using_cross_rate = max_cross_rate
                # 都没有超过最大比例，但都小于最小，重新计算，贴合下限
                elif max(y_occupy, x_occupy) < min_cross_rate:
                    using_cross_rate = min_cross_rate
                # 一个在比例内，一个小，不变，全部在比例内，不变,flag设为0
                else:
                    using_cross_rate = mid_cross_rate
                    need_recalc_rate = False
            # print(f"frane:{i}, using_rate{using_cross_rate},need_recalc:{need_recalc_rate}")
            div_times = last_div_times
            if need_recalc_rate:
                div_times = int(y_cross / (height * using_cross_rate))
                if (x_div_times:=int(x_cross / (width * using_cross_rate))) > div_times:
                    # 如果按横坐标缩放，代表的范围更广，应采纳横坐标缩放倍率，反之亦然
                    div_times = x_div_times
                last_div_times = div_times
            
            #计算当前点的画布坐标，并绘制当前点
            for j, current_mp in enumerate(self.mps):
                # 对于每个质点，最多追溯max_tail个元素
                canvas_pos = self._calc_canvas_pos(self.historys[j][i], ori_opoint, div_times, width, height)

                eng_font = cv2.FONT_HERSHEY_SIMPLEX

                # 计算推演时间
                total_sec = i * self.dlt_t * self.iter_round
                day = total_sec // (3600 * 24)
                hour  = (total_sec % (3600 * 24)) // 3600
                minutes = (total_sec % 3600) // 60
                sec = total_sec % 60
                text_pos = (line_end[0] + 20, line_end[1] + 10)
                cv2.putText(current_img, f"{np.round(line_pix * div_times / 1000, 1)} KM", text_pos, eng_font, 1, color=rate_color, thickness=2)

                # 时间
                text_pos = (line_st[0], line_st[1] + int(height * 0.05))
                # cv2.putText(current_img, f"Day {day}, {hour:02}:{minutes:02}:{sec:02}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0,255,255), thickness=2)
                chn_font_size = int(width * 0.02)
                # current_img = self.__put_text_chinese(current_img, f"Day {day}, {hour:02}:{minutes:02}:{sec:02}  推演步长:{self.dlt_t} 秒, 推演总时长 {int(self.history_count / (86400 /(self.dlt_t * self.iter_round)))} 天", pos=text_pos, color=(0,255,255), font_size=chn_font_size)
                cv2.putText(current_img, f"Day {day}, {hour:02}:{minutes:02}:{sec:02}  Simulation Step:{self.dlt_t} sec, Total Simulation Time: {int(self.history_count / (86400 /(self.dlt_t * self.iter_round)))} day", text_pos, eng_font, 1, color = rate_color, thickness = 2)

                text_pos = (text_pos[0], text_pos[1] + chn_font_size)

                # 星球信息，首先计算剩余行高，平均除以星球数，计算当前星球所在的行
                remain_height = int((height - text_pos[1]) * 0.8)
                each_height = remain_height // len(self.mps)
                text_pos = (text_pos[0], text_pos[1] + each_height * (j + 1))
                # 一个scale对应的像素数
                font_rate = cv2.getTextSize("mNgP", eng_font, 1, 1)[0][1]
                font_size = (each_height / font_rate) * 0.5

                p_info_text = f"Name: {current_mp.name}, M: {current_mp.m:.2e} Kg, Pos: ({self.historys[j][i][0]:.2e}, {self.historys[j][i][1]:.2e}) m, V: ({self.v_history[j][i][0]:.2}, {self.v_history[j][i][1]:.2}) m/s"

                cv2.putText(current_img, p_info_text, text_pos, eng_font, font_size, color=self.colors[j], thickness=2)

                # 绘制拖影
                tail_i = max(0, i - max_tail)
                tail_color_base = [min(rgb * 1.2, 255) for rgb in self.colors[j % len(self.colors)]]
                while tail_i < i:
                    # 计算该拖影在不在画面内，不在就不画，节省性能
                    canvas_pos = self._calc_canvas_pos(self.historys[j][tail_i], ori_opoint, div_times, width, height)
                    if self._in_canvas(canvas_pos, width, height):
                        cv2.circle(current_img, canvas_pos, radius=2, color=[int(rgb *  (1 - (i - tail_i) / max_tail)) for rgb in tail_color_base], thickness=-1)
                    tail_i += 1
                # 绘制星球本体
                cv2.circle(current_img, canvas_pos, radius=int(8*dot_scale[j]), color=self.colors[j % len(self.colors)], thickness=-1)
                    
            video_witer.write(current_img)
            tpl_imgs[buffer_index] = None
            current_img = None
        video_witer.release()
        print("写入完成")

def gen_simulation_video(mps:list[MP], calc_step_s:int = 2, frame_steps_interval:int = 3600, video_fps:int = 30, max_tail:int = 1000, video_sec:int = 30, total_frames_cover_sec:int = None):
    sub_dir = datetime.now().strftime("%m_%d_%H%M%S")
    total_frames = video_sec * video_fps
    if total_frames_cover_sec is not None:
        total_frames = total_frames_cover_sec
        video_sec = total_frames // video_fps

    virtual_days = calc_step_s * frame_steps_interval * total_frames // 86400

    simulation_info = f"准备生成 {len(mps)} 体运动模拟动画，计算步长 {calc_step_s} 秒, 每帧 {frame_steps_interval} 步， 视频帧率: {video_fps} fps, 拖尾长度:{max_tail} 帧, 视频时长:{video_sec}秒, 每秒相当时长:{calc_step_s * frame_steps_interval * video_fps / 86400}天, 模拟相当时长:{virtual_days}天，存储在文件夹:{sub_dir}。是否继续？(y/n)"

    ask_continue = input(simulation_info).strip().lower()
    if ask_continue != "y":
        sys.exit(0)

    system = MultiBody(mps, calc_step_s, frame_steps_interval, total_frames, sub_dir=sub_dir)
    go_calc = True
    is_first = True
    video_no = 1
    
    while go_calc:
        st = time.time()
        for i in range(system.history_count - (1 if is_first else 0)):
            print(f"calc round {i} / {system.history_count}\r", end="")
            system.calc_round()
        
        end = time.time()
        print(f"\ncalc use {end - st}s")

        system.gen_video(f"threebody_{video_no}", max_tail=max_tail, fps=video_fps)
        try:
            go_calc = inputimeout.inputimeout(("go? (y/n)"), 2).lower().strip() == "y"
        except inputimeout.TimeoutOccurred:
            go_calc = True
        is_first  =False
        video_no += 1
    pass
def main():
    # # # 模拟三体
    # stable1 = MP(pos = [0, 1.5e11], m = 1.6e30, v = np.array([25000,0]), name="sun", dtype=np.float64)
    # stable2 = MP(pos = [8.66e10, 0], m = 1.2e30, v = np.array([-12500,-21650]), name="earth", dtype=np.float64)
    # stable3 = MP(pos = [-8.66e10, 0], m = 0.6e30, v = np.array([-12500,21650]), name="moon", dtype=np.float64)
    # non_stable = [stable1, stable2, stable3]
    # gen_simulation_video(non_stable, 180, 600, 60, 1000, 90)

    # # 模拟双行星系
    # sun1 = MP(pos = [3e11, 0], m = 2e30, v = np.array([0,15000]), name="S1", dtype=np.float64)
    # sun2 = MP(pos = [-3e11, 0], m = 1.5e30, v = np.array([0,-15000]), name="S2", dtype=np.float64)
    # earth1 = MP(pos = [4e11, 0], m = 0.4e30, v = np.array([0,-24000]), name="E1", dtype=np.float64)
    # earth2 = MP(pos = [-4e11, 0], m = 0.4e30, v = np.array([0,16000]), name="E2", dtype=np.float64)

    # double_solar = [sun1, sun2, earth1, earth2]
    # gen_simulation_video(double_solar, 45, 2400, 60, 1000, 90)

    # # 模拟双行星系2
    sun1 = MP(pos = [2.5e11, 0], m = 2e30, v = np.array([0,17000]), name="S1", dtype=np.float64)
    sun2 = MP(pos = [-2.5e11, 0], m = 1.5e30, v = np.array([0,-17000]), name="S2", dtype=np.float64)
    earth1 = MP(pos = [3.5e11, 0], m = 0.4e30, v = np.array([0,-20000]), name="E1", dtype=np.float64)
    earth2 = MP(pos = [-4e11, 0], m = 0.4e30, v = np.array([0,15000]), name="E2", dtype=np.float64)
    earth3 = MP(pos = [-2.5e11, 0.5e11], m = 0.3e30, v = np.array([-35000,-17000]), name="E3", dtype=np.float64)

    double_solar = [sun1, sun2, earth1, earth2, earth3]
    gen_simulation_video(double_solar, 45, 2400, 60, 1000, 90)

    # 模拟四体
    # stable1 = MP(pos = [1.5e11,1.5e11], m = 1.61e30, v = np.array([-12500,12500]), name="sun1", dtype=np.float64)
    # stable2 = MP(pos = [-1.5e11,1.5e11], m = 1.61e30, v = np.array([-12500,-12500]), name="sun2", dtype=np.float64)
    # stable3 = MP(pos = [-1.5e11,-1.5e11], m = 1.61e30, v = np.array([12500,-12500]), name="sun3", dtype=np.float64)
    # stable4 = MP(pos = [1.5e11,-1.5e11], m = 1.61e30, v = np.array([12500,12500]), name="sun3", dtype=np.float64)
    # non_stable = [stable1, stable2, stable3, stable4]
    # gen_simulation_video(non_stable, 9, 12000, 60, 1000, 120)

    # # 模拟20体
    # plants = [MP(pos = [(np.random.rand() - 0.5) * 3e11, (np.random.rand() - 0.5) * 3e11], m = 0.4e30 + np.random.rand() * 1.6e30, v = np.array([(np.random.rand() - 0.5) * 20000, (np.random.rand() - 0.5) * 20000]), name=f"P{i}",dtype=np.float64) for i in range(8)]
    # gen_simulation_video(plants, 30, 1200, 60, 400, 90)

    # 模拟双星系统
    # sun1 = MP(pos = [-2e11,0.5e11], m = 1.5e30, v = np.array([10000,0]), name="sun", dtype=np.float64)
    # sun2 = MP(pos = [2e11,-0.5e11], m = 1.5e30, v = np.array([-10000,0]), name="sun", dtype=np.float64)
    # double_sun = [sun1, sun2]
    # gen_simulation_video(double_sun, 60, 960, 60, 1500, 45)

if __name__ == "__main__":
    main()