from turtle import *  # 导入turtle库

def draw_simple_circle():
    speed(6)  
    bgcolor("white")  # 白色背景，简洁

    # 黄色圆形
    color("orange", "yellow")  # 线条橙，填充黄
    penup()
    goto(0, 0)  # 从中心开始
    pendown()
    begin_fill()
    circle(40)  # 半径40的圆
    end_fill()


    done()  # 保持窗口

# 调用函数画图
draw_simple_circle()