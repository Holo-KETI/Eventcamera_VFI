import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.Expv8_large.runExpv8_large import Expv8_large
# from params.GOPRO_release.params_trainOurs_mix import trainGOPRO_Ours
from params.bsergb.params_traintest_x4AdamwithLPIPS import traintest_BSERGB_x4AdamwithLPIPS
from params.bsergb.params_traintest_x5AdamwithLPIPS_vali import  traintest_BSERGB_x5AdamwithLPIPS_vali
from easydict import EasyDict as ED
import time

import torch.nn as nn
import torch.optim as optim


# 텐서 크기 및 메모리 측정
def get_model_memory(model):
    # GPU가 있을 경우 GPU 메모리 사용량 측정
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 캐시 초기화
        mem_alloc = torch.cuda.memory_allocated()
        mem_cached = torch.cuda.memory_reserved()
        return mem_alloc, mem_cached
    else:
        return "CUDA를 사용할 수 없습니다."

args = ED()
args.model_name = 'Expv8_large'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
args.calc_flops = True
# args.param_name = 'traintest_BSERGB_x4AdamwithLPIPS'

params = traintest_BSERGB_x5AdamwithLPIPS_vali(args)
# params = args.param_name
records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
# params.training_config.crop_size = 64


params.training_config.interp_ratio = 20
params.real_interp = 5
# params.real_interp = 4
params.save_flow = False
params.save_images = False

params.debug = False
datashape_h, datashape_w = 576, 928
# datashape_h, datashape_w = params.training_config.crop_size, params.training_config.crop_size

net = Expv8_large(params).cuda()
left_frame = torch.randn(1, 3, datashape_h, datashape_w).float().cuda()
right_frame = torch.randn(1, 3, datashape_h, datashape_w).float().cuda()
events = torch.randn(1, 125, datashape_h, datashape_w).float().cuda()

# 메모리 사용량 출력
alloc, cached = get_model_memory(net.net.events_encoder)
print(f"메모리 할당량: {alloc} bytes")
print(f"메모리 예약량: {cached} bytes")

# 예시로 하나의 배치 실행
inputs = torch.randn(32, 256).cuda()  # 가상 입력 데이터 (32배치 크기, 256 특성)
model = net.eval()  # 모델을 GPU로 이동

# 모델 실행
# outputs = net(left_frame, right_frame, events)

# inputs=(left_frame, right_frame, events)
# print(left_frame.shape, right_frame.shape, events.shape)
# from thop import profile
inputs =[left_frame, right_frame, events]
macs, params = profile(model,inputs)
print(macs, params)
# import torchinfo

# 346 x 260
# torchinfo.summary(model, input_size=[(1,3, 256, 256),(1,3, 256, 256),(1,125, 256, 256)])

# import torch
# import torch.fx
# import csv

# ⚠️ 모델과 입력 설정
# 모델 정의 또는 불러오기
# model.eval()  # 반드시 evaluation 모드로 설정

# # torch.fx로 모델 트레이싱
# traced = torch.fx.symbolic_trace(model)

# # 연산자 개수 세기
# op_count = {}
# for node in traced.graph.nodes:
#     if node.op in ['call_function', 'call_module', 'call_method']:
#         op_name = str(node.target)
#         op_count[op_name] = op_count.get(op_name, 0) + 1

# # 결과를 CSV 파일로 저장
# csv_filename = 'operator_counts.csv'
# with open(csv_filename, mode='w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(['Operator', 'Count'])
#     for op, count in sorted(op_count.items(), key=lambda x: x[1], reverse=True):
#         writer.writerow([op, count])

# print(f"✅ 연산자 통계가 '{csv_filename}' 파일에 저장되었습니다.")
# torch.onnx.export(
#     model,
#     inputs,  # tuple로 전달
#     "model_512_x4.onnx",
#     export_params=True,
#     opset_version=16,
#     input_names=['input1', 'input2','input3','input4'],
#     output_names=['output']
# )

# torch.onnx.export(model, inputs, "multi_input_model_512.onnx")



# # 다시 메모리 사용량 출력
# alloc, cached = get_model_memory(model)
# print(f"메모리 할당량: {alloc} bytes") #0.605945344
# print(f"메모리 예약량: {cached} bytes")

# macs, model_params = 0, 0

# outprofile = profile(net, inputs=(left_frame, right_frame, events, 2))
# macs += outprofile[0]
# model_params += outprofile[1]
# content = f'[MODEL NAME] {args.model_name} '\
#           f'[INPUT INFO] {datashape_h}x{datashape_w}x{params.training_config.interp_ratio} '\
#           f'[MACs]       {macs/1e9:.3f} G MACs [AVERAGE MACs]: {macs/1e9/(params.real_interp-1):.3f} GMACs'\
#           f'[PARAMs]     {model_params/1e9:.3f} G'
# print('-'*20)
# print(content)

# with torch.no_grad():
#     res = net(left_frame, right_frame, events, params.training_config.interp_ratio)
#     t = time.time()
#     for i in range(10):
#         res = net(left_frame, right_frame, events, params.training_config.interp_ratio)
#     print((time.time()-t)/10/(params.real_interp-1))

# records.write(content+'\n')







