'''
Debug代码调试：① 了解程序执行流程 ② 运行结果异常，通过Debug工具了解出入情况
Debug三步走：
① 下断点 => if/while/for关键字所在位置打断点
② 启动Debug调试
③ 通过Step Over按钮，一步一步调试代码
'''
i = 1
while i <= 100:
    if i % 2 == 0:
        print(i)
    i += 1