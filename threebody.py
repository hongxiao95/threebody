# -*- coding:utf-8 -*-
# @Author HONG, Xiao

import scipy.constants as CONSTANTS
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


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

    def calc_a(self, mps:list["MP"]):
        '''
        计算合加速度
        '''
        self._clear_a()
        for mp in mps:
            if mp == self:
                continue
            distance = self._clac_distance(mp)
            self.a += CONSTANTS.G * mp.m / np.power(distance, 3) * (mp.pos - self.pos)
    
    def update_move(self, dlt_t:np.int32):
        '''
        根据算出的a更新pos和速度,务必全部更新完加速度再更新距离,防止循环计算
        '''
        dlt_s = self.v * dlt_t + self.a * np.square(dlt_t) / 2
        self.v += self.a * dlt_t
        self.pos += dlt_s

class MultiBody:
    '''
    多体模拟系统
    '''
    video_dir = "video"
    colors = [(240,50,50),(50,240,50),(50,50,240),(240,240,50),(240,50,240),(50,240,240),(255,255,255)]
    def __init__(self, mps:list[MP], dlt_t:np.int32 = 10, iter_round:np.int32 = 8640, history_count:int = 100):
        self.mps, self.dlt_t, self.iter_round = mps, dlt_t, iter_round
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



    def gen_video(self, file_name:str="threebody", width:int=1920, height:int=1080, max_tail:int=300, padding:float = 0.3, fps:int=30):
        print("\n")
        if os.path.exists(self.video_dir) == False:
            os.mkdir(self.video_dir)

        # 模版画面
        tpl_img = np.zeros((height, width, 3), dtype=np.uint8)
        p_count = len(self.mps)
        fourcc = cv2.VideoWriter.fourcc(*"H264")
        video_witer = cv2.VideoWriter(f"{self.video_dir}{os.sep}{file_name}.mp4",fourcc=fourcc,fps=30,frameSize=(width, height))
        for i in range(self.history_count):
            print(f"正在写入第{i}帧\r", end="")
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
            for j in range(p_count):
                # 对于每个质点，最多追溯max_tail个元素
                canvas_pos = self._calc_canvas_pos(self.historys[j][i], ori_opoint, div_times, width, height)
                
                cv2.circle(current_img, (canvas_pos), radius=8, color=self.colors[j % len(self.colors)], thickness=-1)
                tail_i = i - 1
                while tail_i >= 0 and i - tail_i - 1 < max_tail:
                    canvas_pos = self._calc_canvas_pos(self.historys[j][tail_i], ori_opoint, div_times, width, height)
                    if self._in_canvas(canvas_pos, width, height):
                        cv2.circle(current_img, (canvas_pos), radius=2, color=self.colors[j % len(self.colors)], thickness=-1)
                    tail_i -= 1
                    
            video_witer.write(current_img)

        video_witer.release()
        print("写入完成")


def main():
    p1 = MP(pos = [500000000,0], m = 2e24, v=np.array([100,-850]), name="p1", dtype=np.float64)
    p2 = MP(pos = [0,280000000], m = 1.5e24, v = np.array([-300,380]), name="p2", dtype=np.float64)
    p3 = MP(pos = [-200000000,0], m = 1.8e24, v = np.array([-450,-150]), name="p3", dtype=np.float64)

    system = MultiBody([p1, p2, p3], 10, 720, 12*30)
    for i in range(system.history_count - 1):
        print(f"calc round {i} / {system.history_count}\r", end="")
        system.calc_round()

    plt.figure()
    for i, mp in enumerate(system.mps):
        plt.scatter(system.historys[i][:,0], system.historys[i][:,1], s=2,label=f"{mp.name}, pos:{np.round(mp.ori_pos, 2)}m, M:{mp.ori_m}kg, v:{mp.ori_v}m/s")
    plt.title("Threebody")
    plt.axis("equal")
    plt.legend()
    # plt.show()
    system.gen_video()
    


if __name__ == "__main__":
    main()