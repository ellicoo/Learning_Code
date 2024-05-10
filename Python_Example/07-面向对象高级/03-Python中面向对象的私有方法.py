'''
私有属性：明确区分内外，控制外部对私有属性的操作行为，起到保护数据目的
私有方法：简化调用程序，降低程序的复杂度

案例：编写一个ATM机系统，要求功能包含：① 插卡 ② 验证 ③ 输入取款金额 ④ 取款 ⑤ 打印取款凭证

小结：简化调用程序，降低程序的复杂度
'''
# 1、定义一个ATM类
class Atm():
    # 2、编写对应的相关方法
    def __card(self):
        print('插卡')
    def __auth(self):
        print('授权')
    def __input(self):
        print('输入')
    def __take_money(self):
        print('取款')
    def __print_bill(self):
        print('凭证')

    # 定义一个公共接口，取款操作必须通过这个接口来完成
    def withdraw(self):
        self.__card()
        self.__auth()
        self.__input()
        self.__take_money()
        self.__print_bill()

# 3、实例化类生成对象
atm = Atm()
atm.withdraw()
