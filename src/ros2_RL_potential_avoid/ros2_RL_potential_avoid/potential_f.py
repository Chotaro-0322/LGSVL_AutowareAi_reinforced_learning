import math
import numpy as np

class Potential_avoid():
    """https://qiita.com/koichi_baseball/items/0a6eef85be8d700f6125を参考"""
    def __init__(self, delt, speed, weight_goal):
        self.delt = delt
        self.speed = speed
        self.potential_min = -1
        self.potential_max = 1
        self.weight_obst = 0.1
        self.weight_goal = weight_goal
    
    def cal_potential(self, x, y, goal, obst_target, actions):
        tmp_pot = 0
        for obst in obst_target:
            # 障害物がないとき(Noneがはいっている)
            if obst[0] == None or obst[1] == None:
                obst_pot = 0

            # 障害物の座標のpotentialはmax
            elif obst[1] == y and obst[0] == x:
                obst_pot = self.potential_max
            else:
                if obst[2] == 50: # ただの壁(costmapを作成したときに適当に決めた値)
                    obst_weight = actions[0]
                elif obst[2] == 75: # 車だった
                    obst_weight = actions[1]
                elif obst[2] == 100: # 歩行者だった
                    obst_weight = actions[2] 
                # print("obst_weight : ", obst_weight)
                obst_pot =  1 / math.sqrt(pow((y - obst[1]), 2) + pow((x - obst[0]), 2)) * obst_weight # obst[2]にはその物体のobst_weightが入力されている 
                # obst_pot += obst_pot * self.weight_obst

            tmp_pot += obst_pot

        # ゴールの座標はpotentialはmin
        if goal[1] == y and goal[0] == x:
            goal_pot = self.potential_min
        else:
            goal_pot = -1 / math.sqrt(pow((y - goal[1]),  2) + pow((x - goal[0]),  2))

        pot_all = tmp_pot + self.weight_goal * goal_pot

        return pot_all

    def calculation(self, start, goal, actions, grid_map, yaw, velocity, change_flag):
        # grid_map内の障害物の位置をまとめる
        grid_height, grid_width = grid_map.shape[0], grid_map.shape[1]
        # print("grid_map : ", grid_map.shape)
        obst_grid = np.array(list(zip(*np.where(grid_map[:, :, 2] > 0)))) # 障害物の場所(grid場所)を計算
        # print("obst_grid : ", obst_grid)
        obst_position = np.array([grid_map[position[0], position[1]] for position in obst_grid]) # 障害物のgrid番号と実際のmapから障害物の座標を計算
        # print("obst_position : ", obst_position.shape)
        xy_pos = start
        output_route = np.array([start[0], start[1], 0, yaw, velocity, change_flag])[np.newaxis, :] # ルートをここに追加していく
        # print("first_waypoint : ", output_route.shape)
        # print("start : ", start)
        
        count = 0

        while True:
            count += 1
            # print("yx_pos : ", yx_pos)
            error2obst = np.sum(np.abs(obst_position[:, :2] - xy_pos), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
            # print("error2obst_sort", error2obst.shape)
            error2obst_sort = np.argsort(error2obst)
            nearest_obstacle = np.zeros((50, 3))
            
            
            for i in range(50): #近い障害物の座標をn個得る
                # print("error2obst_sort : ", np.where(error2obst_sort == i)[0][0])
                nearest_obstacle[i] = obst_position[np.where(error2obst_sort == i)[0][0]]
            # print("nearest_obstacle : ", nearest_obstacle.shape)

            vx = -(self.cal_potential(xy_pos[0] + self.delt, xy_pos[1], goal, nearest_obstacle, actions) - self.cal_potential(xy_pos[0], xy_pos[1], goal, nearest_obstacle, actions)) / self.delt
            vy = -(self.cal_potential(xy_pos[0], xy_pos[1] + self.delt, goal, nearest_obstacle, actions) - self.cal_potential(xy_pos[0], xy_pos[1], goal, nearest_obstacle, actions)) / self.delt
            
            v = np.sqrt(vx * vx + vy * vy)

            # 正規化
            vy /= v / self.speed
            vx /= v / self.speed

            xy_pos[0] += vx
            xy_pos[1] += vy

            # print("output_route : ", output_route.shape)
            # print("yx_pos : ", yx_pos.shape)
            # print(np.array([yx_pos[1], yx_pos[0], 0, yaw, velocity, change_flag]).shape)

            output_route = np.append(output_route, np.array([xy_pos[0], xy_pos[1], 0, yaw, velocity, change_flag])[np.newaxis, :], axis=0)

            # ゴールに近づいた場合，10,000回ループした場合，終了
            if np.abs(goal[0] - xy_pos[0]) < self.speed and np.abs(goal[1] - xy_pos[1]) < self.speed:
                # print("output_route  : \n", output_route.shape)
                return True, output_route
            if count > 500:
                print("count over !!!")
                return False, output_route