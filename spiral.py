from turtle import *
from random import randint

def draw_spiral_variant():
    bgcolor('black')  # 黑色背景突出彩色，比原红色更显层次
    x = 1
    speed(10)
    colormode(255)  # 扩展颜色范围到0-255，色彩更丰富（原200基础上优化）
    
    while x < 380:  # 循环次数
        r = randint(50, 255)  # 避免过暗，最低亮度50
        g = randint(50, 255)
        b = randint(50, 255)
        
        pencolor(r, g, b)
        fd(40 + x) 
        rt(120.2)  # 螺旋呈正三角式扩展
        x += 1  # 步长递增
    
    exitonclick()  # 点击关闭窗口的交互

# 调用函数绘制
draw_spiral_variant()