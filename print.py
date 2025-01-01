# # import torch
# # from model import MattingNetwork

# # # 实例化模型
# # model = MattingNetwork('mobilenetv3').eval()

# # def print_named_children(module, prefix=''):
# #     print(f"\nInspecting: {module.__class__.__name__}")
# #     for name, child in module.named_children():
# #         # 打印模块名和模块的类型
# #         print(f"{prefix}{name}: {child.__class__.__name__}")
# #         # 递归调用，打印子模块的子模块
# #         print_named_children(child, prefix + '  ')

# # # 打印模型的所有子模块
# # print_named_children(model)


# import torch
# from model import MattingNetwork

# # 实例化模型
# model = MattingNetwork('mobilenetv3').eval()

# # 打印 model 的所有直接子模块
# for name, child in model.named_children():
#     print(f"{name}: {child.__class__.__name__}")
import torch
from model import MattingNetwork

model = MattingNetwork('mobilenetv3').eval()

def check_special_modules(module):
    for name, child_module in module.named_children():
        # 打印结构
        print(name, child_module)
        


        # 如果模块还包含子模块，递归调用
        # check_special_modules(child_module)

# 调用函数开始检查
check_special_modules(model)
