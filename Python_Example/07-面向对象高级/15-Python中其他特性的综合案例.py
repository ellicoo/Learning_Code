'''
① 成员方法（实例方法、对象方法）：每个对象都会拥有的方法
② 类方法：类自身方法，作用：调用自身类属性
③ 静态方法：既不需要调用自身属性，也不需要调用自身方法
案例：定义一个Game游戏类，定义一个show_help方法，用于显示游戏的开始菜单；定义一个show_top_score方法，用于显示游戏最高分；定义一个start_game()
方法，用于开始游戏，每个游戏也有自己的名称等等自身属性。

show_help()：静态方法，既不需要调用自身属性也不需要调用自身方法
show_top_score()：类方法
start_game()：对象方法，每个游戏都应该具有方法
'''
class Game(object):
    # 类属性
    score = 0
    # 定义一个静态方法show_help()
    @staticmethod
    def show_help():
        print('欢迎来到游戏世界')
        print('游戏规则如下:...')

    # 类方法
    @classmethod
    def show_top_score(cls):
        print(f'目前游戏最高分为：{cls.score}')

    # 定义对象方法
    def start_game(self):
        print('开始游戏')

    # 为游戏定义相关属性
    def __init__(self, name):
        self.name = name