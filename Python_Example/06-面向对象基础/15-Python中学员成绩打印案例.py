'''
需求：定义学员信息类，包含姓名、成绩属性，定义成绩打印方法（90分及以上显示优秀，80分及以上显示良好，70分及以上显示中等，60分及以上显示合格，60分以下显示不及格）

思路分析：面向对象程序设计
① 找对象 => 学生对象
② 找属性和方法 => 属性：姓名、成绩；方法：成绩打印
③ 让对象执行对应的方法 => 实例化对象，调用成绩打印方法
'''
# 1、定义一个学生信息类
class Student(object):
    # 2、定义成员属性，name、score
    def __init__(self, name, score):
        self.name = name
        self.score = score
    # 3、定义成员方法，print_grade()
    def print_grade(self):
        if self.score >= 90:
            print('优秀')
        elif self.score >= 80:
            print('良好')
        elif self.score >= 70:
            print('中等')
        elif self.score >= 60:
            print('及格')
        else:
            print('不及格')

# 4、实例化对象
s1 = Student('Jack', 95)
# 5、调用成绩打印方法，输出对应的信息
print(s1.name)
print(s1.score)
s1.print_grade()