import time
import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('robots/summit_xl_description/summit_xls.xml')
d = mujoco.MjData(m)

mobile_kv = 200.0  # 控制增益

mobile_dot = np.zeros(4)
command = np.zeros(4)

# 在循环外部添加参数定义
radius = 1.5  # 圆形轨迹半径（米）
linear_speed = 8  # 线速度（米/秒）
wheel_base = 0.35  # 轮距（左右轮中心距离，根据实际机器人调整）


t = 0  # 时间变量，用来模拟圆形运动
delay = 1000  # 初始延迟
 
with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        
        # 获取当前轮速
        mobile_dot[0] = d.qvel[19]  # 前左
        mobile_dot[1] = d.qvel[6]   # 前右
        mobile_dot[2] = d.qvel[45]  # 后左
        mobile_dot[3] = d.qvel[32]  # 后右

        # 圆形轨迹速度计算
        if t < delay:
            target_vel = np.zeros(4)
        else:
            # 计算差速转向参数
            angular_speed = linear_speed / radius
            delta_speed = angular_speed * wheel_base
            
            # 设置左右轮速度（保持线速度恒定）
            left_speed = linear_speed - delta_speed/2
            right_speed = linear_speed + delta_speed/2
            
            # 设置四轮目标速度（前左，前右，后左，后右）
            target_vel = np.array([left_speed, right_speed, 
                                  left_speed, right_speed])

        # 保持原有控制逻辑
        command = (target_vel - mobile_dot) * mobile_kv
        d.ctrl[0] = command[1]  # 前右
        d.ctrl[1] = command[0]  # 前左
        d.ctrl[2] = command[3]  # 后右
        d.ctrl[3] = command[2]  # 后左

        viewer.sync()

        # 时间管理，避免与墙时钟漂移
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # 每100步打印当前的速度和控制命令
        if t % 100 == 0:
            print("current velocity:")
            print(mobile_dot)
            print("control force:")
            print(command)
            print("\n\n")

        # 更新时间变量t，增加圆形运动的进度
        t += 1
